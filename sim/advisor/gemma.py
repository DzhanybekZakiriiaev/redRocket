"""Gemma advisor -- the LLM arm. PLAN.md section 5.4 + RESKIN-BRIEF.md section 5.

Drops into the same swap point as BayesAdvisor / NullAdvisor (base.py): same
observations in, same Policy out, same code path around it. The only difference
is *how the belief is formed* -- here, by asking Gemma 4 E4B (edge-sized,
onboard-deployable; hosted on Spur's compute for the demo) to name the most
likely cause from the local evidence, then reusing the SHARED optimal decision
rule `bayes_action` so the comparison against the Bayes filter stays honest.

The advisor NEVER sees FaultTrace, the engine, or the contact plan -- only what
arrives in an Observation and gossiped Evidence. A test enforces this.

Config is read from the repo-root `.env` (git-ignored). Set:

    SPUR-GEMMA4-API-KEY   = <your key>
    SPUR-BASE-URL         = https://ai.spuric.com/v1   (OpenAI-compatible base)
    SPUR-MODEL            = spur-gemma4                 (served model id)

Then run:  python -m sim.trace  (with advisor_name="gemma"), or the smoke test
`python -m tools.gemma_smoke` first to confirm the endpoint answers.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from ..types import (
    Observation, Evidence, Policy, Action, Cause, Outcome, Channel, CAUSES,
)
from .base import bayes_action, nominal_belief


# --------------------------------------------------------------------------
# Config -- env var NAMES (hyphenated, matching the existing key). Values live
# in .env, never in code, never in git.
# --------------------------------------------------------------------------

ENV_KEY = "SPUR-GEMMA4-API-KEY"
ENV_BASE_URL = "SPUR-BASE-URL"
ENV_MODEL = "SPUR-MODEL"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env(path: Path) -> dict[str, str]:
    """Minimal .env reader. Handles `NAME=value`, surrounding quotes, comments,
    and hyphenated names (which os.environ/shells choke on but a dict does not).
    Deliberately dependency-free."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        env[name.strip()] = val
    return env


# --------------------------------------------------------------------------
# The cause menu the model chooses from. Letters map 1:1 to CAUSES order.
# Framed in deep-space language (RESKIN-BRIEF section 4.1); the internal enum is
# unchanged. Descriptions carry the DISCRIMINATING signal for each cause so the
# model has something to reason with -- the same features the Bayes filter uses.
# --------------------------------------------------------------------------

LETTERS = "ABCDEF"
assert len(LETTERS) == len(CAUSES)
_LETTER_TO_CAUSE = {LETTERS[i]: c for i, c in enumerate(CAUSES)}

_CAUSE_MENU = """\
A. NOMINAL - link is healthy; no fault.
B. DUST STORM at the surface site - dust degrades EVERY orbiter pass over that
   site and self-heals over time. Tell: OTHER orbiters over the SAME site also
   fail right now (peer co-failure).
C. SAFE-MODE / BUS FAULT on the peer orbiter - total loss on every link
   INCLUDING the low-gain beacon; does not self-heal. Tell: the beacon is also
   silent, and the silence is long.
D. ANTENNA MISPOINT / attitude drift on THIS orbiter - primary channel only;
   the low-gain beacon stays healthy; it is per-orbiter, so peers at the same
   site are FINE. Tell: beacon OK, no peer co-failure.
E. RECORDER FULL - onboard storage saturated; correlates with a deep local
   queue; clears if traffic is throttled. Tell: queue is deep.
F. STALE SEQUENCE / EPHEMERIS DRIFT - the contact is SHIFTED in time, not
   absent; carrier seen outside the scheduled window. Tell: a nonzero time
   shift was observed."""


def _bucket_silence(ms: int) -> str:
    if ms < 5 * 60_000:
        return "short (<5 min)"
    if ms < 30 * 60_000:
        return "medium (5-30 min)"
    return "long (>30 min)"


class GemmaAdvisor:
    """LLM advisor. One diagnosis per adverse link, on demand, cached."""

    name = "gemma"

    def __init__(self, *, cache_path: str | None = None, timeout_s: float = 30.0,
                 raise_on_error: bool = False):
        env = _load_env(_REPO_ROOT / ".env")
        # os.environ (if exported) wins over the file; either is fine.
        import os
        self.key = os.environ.get(ENV_KEY) or env.get(ENV_KEY, "")
        self.base_url = (os.environ.get(ENV_BASE_URL) or env.get(ENV_BASE_URL, "")).rstrip("/")
        self.model = os.environ.get(ENV_MODEL) or env.get(ENV_MODEL, "")
        self.timeout_s = timeout_s
        self.raise_on_error = raise_on_error

        missing = [n for n, v in
                   ((ENV_KEY, self.key), (ENV_BASE_URL, self.base_url), (ENV_MODEL, self.model))
                   if not v]
        if missing:
            raise RuntimeError(
                "GemmaAdvisor is not configured. Add these to .env:\n  "
                + "\n  ".join(f"{n}=..." for n in missing)
                + "\n(The API key you already have; BASE-URL is the OpenAI-compatible "
                  "endpoint on Spur's compute, e.g. https://host/v1; MODEL is the "
                  "served model id.)")

        # Per-link state, mirroring BayesAdvisor's inputs (base.py contract).
        self._last_obs: dict[tuple[str, str], Observation] = {}
        self._recent_outcome: dict[str, dict[str, tuple[int, Outcome]]] = defaultdict(dict)
        self._beacon: dict[tuple[str, str], tuple[int, bool]] = {}
        self._peer_reports: dict[str, list[Evidence]] = defaultdict(list)
        self._seen: set[tuple[str, int]] = set()

        # Response cache -> replay determinism + cheap reruns (PLAN T3).
        self._cache_path = Path(cache_path) if cache_path else (_REPO_ROOT / "out" / "gemma_cache.json")
        self._cache: dict[str, dict] = {}
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}
        self._warned = False

    # -- inputs (identical contract to BayesAdvisor) -----------------------

    def observe(self, obs: Observation) -> None:
        key = (obs.node, obs.peer)
        if obs.channel is Channel.BEACON:
            self._beacon[key] = (obs.t, obs.outcome is Outcome.OK)
            return
        self._last_obs[key] = obs
        self._recent_outcome[obs.node][obs.peer] = (obs.t, obs.outcome)

    def receive(self, evidence: list[Evidence]) -> None:
        for e in evidence:
            if e.key in self._seen:
                continue  # (origin, seq) dedup -- repeats must not inflate confidence
            self._seen.add(e.key)
            self._peer_reports[e.peer].append(e)

    # -- co-failure features (same questions base.py/bayes.py ask) ---------

    def _self_cofail(self, obs: Observation) -> bool | None:
        recent = self._recent_outcome.get(obs.node, {})
        window = 15 * 60_000
        others = [o for p, (t, o) in recent.items()
                  if p != obs.peer and 0 <= (obs.t - t) <= window]
        if not others:
            return None
        return sum(o is not Outcome.OK for o in others) / len(others) >= 0.5

    def _peer_cofail(self, obs: Observation) -> bool | None:
        window = 4 * 3600 * 1000
        reports = [e for e in self._peer_reports.get(obs.peer, ())
                   if e.origin != obs.node and 0 <= (obs.t - e.t) <= window]
        if not reports:
            return None
        return sum(e.outcome is not Outcome.OK for e in reports) / len(reports) >= 0.5

    def _beacon_ok(self, obs: Observation) -> bool | None:
        b = self._beacon.get((obs.node, obs.peer))
        if b and 0 <= (obs.t - b[0]) <= 15 * 60_000:
            return b[1]
        return None

    # -- prompt ------------------------------------------------------------

    def _prompt(self, obs: Observation) -> str:
        self_cf, peer_cf, beacon = self._self_cofail(obs), self._peer_cofail(obs), self._beacon_ok(obs)

        def tri(x: bool | None) -> str:
            return "unknown" if x is None else ("yes" if x else "no")

        rate = ("silent" if obs.measured_rate == 0
                else f"{obs.measured_rate}/{obs.expected_rate} bps")
        evidence = "\n".join([
            f"- observed outcome: {obs.outcome.value.upper()} ({rate})",
            f"- elapsed silence on this link: {_bucket_silence(obs.ms_since_ok)}",
            f"- other links of THIS orbiter failing now (self co-failure): {tri(self_cf)}",
            f"- OTHER orbiters failing at this SAME peer (gossiped): {tri(peer_cf)}",
            f"- low-gain beacon healthy: {tri(beacon)}",
            f"- local send queue deep: {'yes' if obs.queue_bytes >= 1_000_000 else 'no'}",
            f"- carrier seen OUTSIDE the scheduled window (time shift): "
            f"{'yes' if obs.shift_observed else 'no'}",
            f"- concurrent contacts: this orbiter {obs.self_degree}, peer {obs.peer_degree}",
        ])
        return (
            "You are the onboard fault-diagnosis model on a Mars relay orbiter. "
            "A scheduled contact under-performed. From the local evidence alone, "
            "name the single most likely cause.\n\n"
            f"CAUSES:\n{_CAUSE_MENU}\n\n"
            f"EVIDENCE for orbiter {obs.node} <-> {obs.peer}:\n{evidence}\n\n"
            'Reply with ONLY a JSON object, no prose:\n'
            '{"cause": "<one letter A-F>", "confidence": <0..1>, '
            '"rationale": "<one short sentence>"}'
        )

    # -- LLM call (OpenAI-compatible chat completions) ---------------------

    def _call(self, prompt: str) -> dict | None:
        ck = hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()
        if ck in self._cache:
            return self._cache[ck]

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 200,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                resp = json.loads(r.read().decode())
            content = resp["choices"][0]["message"]["content"]
            parsed = _parse(content)
            if parsed is None:
                raise ValueError(f"could not parse cause from: {content!r}")
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
            if self.raise_on_error:
                raise
            if not self._warned:
                print(f"[gemma] call failed ({e}); falling back to NOMINAL for "
                      f"failed calls. Fix .env / endpoint and rerun.", file=sys.stderr)
                self._warned = True
            return None

        self._cache[ck] = parsed
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except Exception:
            pass
        return parsed

    # -- output ------------------------------------------------------------

    def decide(self, node: str, t: int) -> Policy:
        # Diagnose the node's most recently adverse link. Nothing wrong -> stay
        # nominal and DON'T spend an LLM call (matches "conclude nothing" cost).
        adverse = [(p, o) for (n, p), o in self._last_obs.items()
                   if n == node and o.outcome is not Outcome.OK]
        if not adverse:
            b = nominal_belief()
            return Policy(b, Action.NONE, "", t, 1.0, "")

        peer, obs = max(adverse, key=lambda kv: kv[1].t)
        parsed = self._call(self._prompt(obs))

        if parsed is None:                       # call failed -> honest fallback
            b = nominal_belief()
            return Policy(b, *bayes_action(b)[:1], target=peer, until=t,
                          confidence=0.0, rationale="[gemma unavailable]")

        cause = _LETTER_TO_CAUSE.get(str(parsed.get("cause", "")).strip()[:1].upper(),
                                     Cause.NOMINAL)
        conf = float(parsed.get("confidence", 0.6))
        conf = min(0.98, max(0.2, conf))
        belief = _belief_from(cause, conf)
        action, _ = bayes_action(belief)
        return Policy(
            belief=belief, action=action, target=peer,
            until=t + 30 * 60_000, confidence=conf,
            rationale=str(parsed.get("rationale", ""))[:160],
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _parse(content: str) -> dict | None:
    """Pull the JSON object out of the model's reply; tolerate a stray letter."""
    i, j = content.find("{"), content.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            obj = json.loads(content[i:j + 1])
            if isinstance(obj, dict) and "cause" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    for ch in content.strip():
        if ch.upper() in LETTERS:
            return {"cause": ch.upper(), "confidence": 0.6, "rationale": ""}
    return None


def _belief_from(cause: Cause, conf: float) -> dict[str, float]:
    """A simplex point: `conf` on the named cause, the rest spread uniformly.
    Guarantees sum == 1 without the model ever emitting a raw probability vector
    (which it cannot be trusted to normalise -- PLAN 5.4)."""
    rest = (1.0 - conf) / (len(CAUSES) - 1)
    b = {c.value: rest for c in CAUSES}
    b[cause.value] = conf
    return b
