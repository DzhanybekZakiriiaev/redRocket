"""Re-emit the Gemma trace with the model's INPUT attached to every decision.

Why this exists
---------------
`sim/trace.py` records what the advisor concluded -- posterior, action, ground
truth, rationale -- but not what it was asked. The evidence bullets are built in
`GemmaAdvisor._prompt` and thrown away after the call, so the UI can show a
diagnosis and cannot show the reasoning behind it.

How it works without an API key
-------------------------------
The obvious fix, re-running `python -m sim.trace gemma`, needs live credentials
and would spend a few hundred LLM calls. It also risks a *different* run.

So this replays instead. `_ReplayAdvisor` builds the prompt through the real
`GemmaAdvisor._prompt` -- the same code path, no reimplementation -- but returns
the decision already recorded in the baked trace rather than calling anything.
Because every returned Policy is identical to the original run's, the engine
follows the identical trajectory, so the observations feeding the prompt builder
are the identical observations. The prompts recorded here are therefore the
prompts that were actually sent, not a reconstruction of them.

Two consequences worth stating plainly:
  * no network, no .env, no cost, and it cannot alter the run;
  * if the replay ever diverges from the baked decisions, that is a real error
    and this script fails loudly rather than emitting a plausible-looking trace.

Usage
-----
    python -m tools.replay_evidence
    python -m tools.replay_evidence --in web/public/trace_gemma.json \
                                    --out web/public/trace_gemma.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sim.config import DEFAULT
from sim.types import Action, Cause, CAUSES, Outcome, Policy
from sim.advisor.gemma import GemmaAdvisor
from sim import trace as T

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "web" / "public" / "trace_gemma.json"


def _prompt_builder() -> GemmaAdvisor:
    """A GemmaAdvisor for prompt construction only -- no credentials needed.

    __init__ demands a live SPUR endpoint, so it is bypassed and only the
    feature-state attributes that observe()/receive()/_prompt() touch are set.
    The network path (_call) is never reached. Same trick as
    tools/distill_dataset.py:_new_prompt_builder.
    """
    g = GemmaAdvisor.__new__(GemmaAdvisor)
    g._last_obs = {}
    g._recent_outcome = defaultdict(dict)
    g._beacon = {}
    g._peer_reports = defaultdict(list)
    g._seen = set()
    return g


class _ReplayAdvisor:
    """Rebuilds each prompt; returns the decision the baked trace already holds."""

    def __init__(self, baked: dict) -> None:
        self.name = baked["meta"]["advisor"]          # keep the trace self-describing
        self.gemma = _prompt_builder()
        self._order = [c.value for c in CAUSES]
        # (node, t) -> the recorded belief sample
        self._baked: dict[tuple[str, int], dict] = {}
        for node, series in baked["beliefs"].items():
            for b in series:
                self._baked[(node, int(b["t"]))] = b
        self.records: dict[tuple[str, int], dict] = {}
        self.missing = 0

    # -- advisor protocol ---------------------------------------------------

    def observe(self, obs) -> None:
        self.gemma.observe(obs)

    def receive(self, evidence) -> None:
        self.gemma.receive(evidence)

    def decide(self, node: str, t: int) -> Policy:
        # Mirror GemmaAdvisor.decide's link selection exactly: the most recent
        # adverse primary observation is the one that got diagnosed.
        adverse = [(p, o) for (n, p), o in self.gemma._last_obs.items()
                   if n == node and o.outcome is not Outcome.OK]
        if adverse:
            peer, obs = max(adverse, key=lambda kv: kv[1].t)
            self.records[(node, t)] = {
                "peer": peer,
                "prompt": self.gemma._prompt(obs),
                "f": _fields(obs, self.gemma),
            }

        b = self._baked.get((node, t))
        if b is None:
            # No recorded decision at this instant -- the replay has drifted off
            # the original run. Count it; main() treats any drift as fatal.
            self.missing += 1
            return Policy({Cause.NOMINAL.value: 1.0}, Action.NONE, "", t, 1.0, "")

        belief = {k: float(v) for k, v in zip(self._order, b["d"])}
        return Policy(belief, Action(b["a"]), b.get("tg", ""), t,
                      float(max(b["d"])), b.get("r", ""))


def _fields(obs, g: GemmaAdvisor) -> dict:
    """The evidence bullets, split out so the UI can table them."""
    def tri(x):
        return "unknown" if x is None else ("yes" if x else "no")
    return {
        "outcome": obs.outcome.value,
        "measured_rate": obs.measured_rate,
        "expected_rate": obs.expected_rate,
        "ms_since_ok": obs.ms_since_ok,
        "self_cofailure": tri(g._self_cofail(obs)),
        "peer_cofailure_gossiped": tri(g._peer_cofail(obs)),
        "beacon_healthy": tri(g._beacon_ok(obs)),
        "queue_bytes": obs.queue_bytes,
        "shift_observed": "yes" if obs.shift_observed else "no",
        "self_degree": obs.self_degree,
        "peer_degree": obs.peer_degree,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default=str(DEFAULT_IN),
                    help="baked gemma trace to replay")
    ap.add_argument("--out", dest="dst", default=None,
                    help="where to write the enriched trace (default: in place)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src
    if not src.exists():
        print(f"[replay] no such trace: {src}")
        return 2

    baked = json.loads(src.read_text(encoding="utf-8"))
    if baked["meta"]["config_hash"] != DEFAULT.hash:
        print(f"[replay] config mismatch: trace={baked['meta']['config_hash']} "
              f"current={DEFAULT.hash}. The scenario changed since this trace "
              f"was baked; replaying it would attach the wrong evidence.")
        return 2

    adv = _ReplayAdvisor(baked)
    blob = T.build(DEFAULT, advisor=adv)

    if adv.missing:
        print(f"[replay] FAILED: {adv.missing} decisions had no baked "
              f"counterpart -- the replay diverged from the original run. "
              f"Refusing to write a trace whose evidence may not match.")
        return 1

    # Merge the recorded input into each belief sample.
    attached = 0
    for node, series in blob["beliefs"].items():
        for b in series:
            rec = adv.records.get((node, int(b["t"])))
            if rec is not None:
                b["ev"] = rec
                attached += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(blob, separators=(",", ":")), encoding="utf-8")

    total = sum(len(s) for s in blob["beliefs"].values())
    kb = dst.stat().st_size / 1024
    print(f"[replay] {dst}  {kb:.0f} KB")
    print(f"[replay] evidence attached to {attached} of {total} belief samples "
          f"({attached / total * 100:.1f}%)")
    print(f"[replay] samples without evidence are the ones where nothing was "
          f"adverse, so no prompt was ever built -- matching gemma.py's "
          f"'conclude nothing, spend no call' path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
