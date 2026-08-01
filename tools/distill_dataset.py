"""Distillation dataset generator -- turn the optimal Bayes filter into training
data for a small student LLM.

    python -m tools.distill_dataset            # 30 scenarios, default settings
    python -m tools.distill_dataset --seeds 50 --faults 12

CONCEPT
-------
`sim/advisor/bayes.py:BayesAdvisor` is the decision-theoretically optimal fault
diagnoser: given its belief and the shared cost matrix it is provably the best
you can do, and it costs nothing to query. `sim/advisor/gemma.py:GemmaAdvisor`
asks a *language model* the same question, phrased as a natural-language evidence
prompt (`GemmaAdvisor._prompt`). Those two arms see the SAME observations through
the SAME advisor interface.

That symmetry is what makes distillation possible. For every adverse link the
simulator produces, we can build the exact prompt the LLM would be asked AND read
off the exact cause the Bayes filter concludes. Pair them up across thousands of
simulated situations and you get a supervised fine-tuning set that teaches a
small model to reproduce the optimal filter's judgments -- deployable onboard
where running an actual Bayes filter over gossiped evidence is awkward but a
frozen LLM is not.

    (gemma-style evidence prompt)  ->  (optimal cause label from Bayes)

HOW THE ROWS ARE CAPTURED
-------------------------
`_RecordingAdvisor` implements the same observe()/receive()/decide() contract the
engine drives, and internally holds BOTH:

  * a real `BayesAdvisor`   -- its per-link belief argmax is the TEACHER LABEL,
  * a real `GemmaAdvisor`   -- reused ONLY for its prompt/feature machinery
                              (_prompt, _self_cofail, _peer_cofail, _beacon_ok),
                              so the student trains on byte-for-byte the same
                              input the deployed LLM would receive.

`GemmaAdvisor.__init__` refuses to construct without SPUR endpoint credentials in
.env (it is built to actually call an LLM). We do not want or need a network call
here -- only the prompt builder -- so we allocate it via ``__new__`` and populate
exactly the feature-state attributes its prompt methods read. Nothing in gemma.py
is modified; we reuse its code verbatim.

On each decide(), for every link of the node whose latest primary outcome is not
OK, we emit one training row:

    {"messages": [
        {"role": "user",      "content": <gemma evidence prompt>},
        {"role": "assistant", "content": <{"cause": "<letter>",
                                           "confidence": <bayes top prob>,
                                           "rationale": ""} as JSON>}]}

The label letter is the Bayes argmax mapped through gemma's A-F cause menu
(`_LETTER_TO_CAUSE`), so the student's output space matches what gemma.py already
parses. `rationale` is left empty: the Bayes filter has no natural language, and
teaching the student to invent one would be teaching it to hallucinate.

Scenarios are diversified by `ScenarioConfig.seed`, which reseeds both fault
generation and background traffic. The contact plan is pure geometry and seed
independent, so we build it once and reuse it.

RUNS LOCALLY. Needs only numpy + skyfield (already installed). No torch, no GPU,
no network.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sim.config import DEFAULT, ScenarioConfig               # noqa: E402
from sim.types import Outcome, CAUSES                          # noqa: E402
from sim.advisor.bayes import BayesAdvisor                     # noqa: E402
from sim.advisor.gemma import GemmaAdvisor, LETTERS, _LETTER_TO_CAUSE  # noqa: E402
from sim import contacts as C                                  # noqa: E402
from sim import faults as F                                    # noqa: E402
from sim.engine import Engine                                  # noqa: E402


# cause.value -> menu letter. Inverse of gemma's _LETTER_TO_CAUSE, keyed by the
# string value the Bayes belief dict uses.
_CAUSE_VALUE_TO_LETTER = {c.value: L for L, c in _LETTER_TO_CAUSE.items()}
assert len(_CAUSE_VALUE_TO_LETTER) == len(CAUSES) == len(LETTERS)


def _new_prompt_builder() -> GemmaAdvisor:
    """A GemmaAdvisor usable for prompt construction WITHOUT .env credentials.

    We deliberately bypass __init__ (which requires a live SPUR endpoint) and set
    only the feature-state attributes that observe()/receive()/_prompt() touch.
    Everything else on the instance stays untouched because we never call the
    network path (_call / decide).
    """
    g = GemmaAdvisor.__new__(GemmaAdvisor)
    g._last_obs = {}
    g._recent_outcome = defaultdict(dict)
    g._beacon = {}
    g._peer_reports = defaultdict(list)
    g._seen = set()
    return g


class _RecordingAdvisor:
    """Runs the standard advisor contract; logs one row per adverse diagnosis.

    Holds a Bayes filter (for the optimal label) and a Gemma prompt builder (for
    the input), feeding both every observation/evidence the engine delivers so
    their per-link state stays in lockstep. The Policy it returns to the engine is
    the Bayes policy, so the simulation's action dynamics match a real Bayes run.
    """

    name = "record"

    def __init__(self) -> None:
        self.bayes = BayesAdvisor()
        self.gemma = _new_prompt_builder()
        self.rows: list[dict] = []

    def observe(self, obs) -> None:
        self.bayes.observe(obs)
        self.gemma.observe(obs)

    def receive(self, evidence) -> None:
        self.bayes.receive(evidence)
        self.gemma.receive(evidence)

    def decide(self, node: str, t: int):
        # One row per link of this node whose most recent PRIMARY outcome is
        # adverse. gemma._last_obs holds exactly those latest primary obs (beacon
        # observations are stored separately and never land in _last_obs).
        for (n, peer), obs in self.gemma._last_obs.items():
            if n != node or obs.outcome is Outcome.OK:
                continue

            prompt = self.gemma._prompt(obs)               # verbatim gemma input

            belief = self.bayes.belief_for(node, peer)     # optimal posterior
            cause_value = max(belief, key=belief.__getitem__)
            letter = _CAUSE_VALUE_TO_LETTER[cause_value]
            confidence = round(float(belief[cause_value]), 2)

            label = json.dumps({"cause": letter,
                                 "confidence": confidence,
                                 "rationale": ""})
            self.rows.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": label},
                ],
            })

        # Hand the engine the Bayes policy so downstream dynamics are realistic.
        return self.bayes.decide(node, t)


def generate_rows(n_seeds: int, n_faults: int, start_seed: int) -> list[dict]:
    """Run the pipeline over many seeds and collect deduplicated training rows."""
    # Contact plan is pure geometry (seed independent) -- build once, reuse.
    print(f"[distill] building contact plan (geometry, shared across seeds)...")
    contacts = C.build(DEFAULT)
    print(f"[distill] {len(contacts)} directed contacts")

    all_rows: list[dict] = []
    for i in range(n_seeds):
        seed = start_seed + i
        cfg: ScenarioConfig = replace(DEFAULT, seed=seed, tag=f"distill-{seed}")
        faults = F.generate(contacts, cfg, n_faults=n_faults)

        advisor = _RecordingAdvisor()
        Engine(contacts, faults, advisor, cfg).run()
        all_rows.extend(advisor.rows)
        print(f"[distill] seed {seed:4d}: {len(faults):2d} faults, "
              f"+{len(advisor.rows):5d} rows (running total {len(all_rows)})")

    # Dedup exact (prompt, label) pairs. A persistent fault re-diagnosed at many
    # decide ticks with identical evidence would otherwise flood the set with
    # copies and skew the label balance; distinct evidence states are kept.
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for row in all_rows:
        key = (row["messages"][0]["content"], row["messages"][1]["content"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    print(f"[distill] {len(all_rows)} raw rows -> {len(deduped)} after "
          f"exact-duplicate removal")
    return deduped


def _label_distribution(rows: list[dict]) -> Counter:
    dist: Counter = Counter()
    letter_to_name = {L: c.value for L, c in _LETTER_TO_CAUSE.items()}
    for row in rows:
        letter = json.loads(row["messages"][1]["content"])["cause"]
        dist[f"{letter} ({letter_to_name.get(letter, '?')})"] += 1
    return dist


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=30,
                    help="number of scenarios / distinct seeds (default 30)")
    ap.add_argument("--faults", type=int, default=12,
                    help="faults injected per scenario (default 12)")
    ap.add_argument("--start-seed", type=int, default=1000,
                    help="first seed; scenarios use start-seed .. start-seed+seeds-1")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction held out for validation (default 0.1)")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="RNG seed for the train/val shuffle (default 0)")
    args = ap.parse_args()

    rows = generate_rows(args.seeds, args.faults, args.start_seed)
    if not rows:
        print("[distill] no rows produced -- aborting", file=sys.stderr)
        raise SystemExit(1)

    out_dir = DEFAULT.out_path("distill_dataset.jsonl").parent
    full_path = out_dir / "distill_dataset.jsonl"
    train_path = out_dir / "distill_train.jsonl"
    val_path = out_dir / "distill_val.jsonl"

    # Deterministic shuffle + split.
    rng = random.Random(args.shuffle_seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_frac))
    val_rows = shuffled[:n_val]
    train_rows = shuffled[n_val:]

    _write_jsonl(rows, full_path)
    _write_jsonl(train_rows, train_path)
    _write_jsonl(val_rows, val_path)

    dist = _label_distribution(rows)
    total = len(rows)

    print()
    print("=" * 60)
    print(f"  DISTILLATION DATASET")
    print("=" * 60)
    print(f"  total rows        {total}")
    print(f"  train / val       {len(train_rows)} / {len(val_rows)}")
    print(f"  files")
    print(f"    {full_path}")
    print(f"    {train_path}")
    print(f"    {val_path}")
    print(f"  label (cause) distribution:")
    for label, count in dist.most_common():
        bar = "#" * int(40 * count / total)
        print(f"    {label:22s} {count:6d}  {100*count/total:5.1f}%  {bar}")
    print()
    print("  example row:")
    print(json.dumps(rows[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
