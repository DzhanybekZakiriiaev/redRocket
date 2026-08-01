"""Unmodeled-fault demo: the one thing an LLM does that the Bayes filter cannot.

    python -m tools.unmodeled_demo

The Bayes filter (`sim/advisor/bayes.py`) carries a fixed likelihood row per
cause. Its posterior is a distribution over exactly six hardcoded causes, so a
fault OUTSIDE those six has nowhere to go -- it is forced into the nearest
hardcoded bin, with confidence. There is no "none of the above" for it to pick.

The open-world Gemma (`sim/advisor/gemma_openworld.py`) is handed one extra
option -- G. UNKNOWN / ANOMALY -- and nothing else changes. This script builds a
handful of hand-constructed Observations:

  * two CONTROLS that match a known cause (both advisors should agree), and
  * two NOVEL faults with no Cause enum value -- RFI/uplink jamming and an
    SEU frame-corruption event -- whose evidence matches no known tell.

For each case it prints, side by side, what the open-world Gemma says vs what the
Bayes filter is forced to say, and writes the full transcript to
out/unmodeled_demo.json.

The novel faults have NO Cause value on purpose. Their truth is represented by
the literal string "unknown" only inside this script -- that absence is the point.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sim.types import Observation, Outcome, Channel, Cause  # noqa: E402
from sim.advisor.bayes import BayesAdvisor  # noqa: E402
from sim.advisor.gemma_openworld import OpenWorldGemmaAdvisor, UNKNOWN  # noqa: E402

MIN = 60_000  # ms


@dataclass
class Case:
    name: str
    truth: str            # Cause value string, or the sentinel "unknown"
    blurb: str            # human framing of the physical fault
    outcome: Outcome
    measured_rate: int
    expected_rate: int
    queue_bytes: int
    ms_since_ok: int
    shift_observed: int
    beacon_ok: bool       # is the low-gain beacon still healthy?
    elevation_deg: float = 35.0


CASES: list[Case] = [
    # ---- CONTROL 1: a known cause both advisors should get right ----------
    Case(
        name="CONTROL - SAFE-MODE / BUS FAULT (known: node_down)",
        truth=Cause.NODE_DOWN.value,
        blurb="Peer orbiter dropped to safe mode: total loss on every link, "
              "beacon included, and it does not self-heal.",
        outcome=Outcome.SILENT, measured_rate=0, expected_rate=8_000_000,
        queue_bytes=32 * 1024, ms_since_ok=55 * MIN, shift_observed=0,
        beacon_ok=False,
    ),
    # ---- CONTROL 2: a known cause both advisors should get right ----------
    Case(
        name="CONTROL - RECORDER FULL (known: buffer)",
        truth=Cause.BUFFER.value,
        blurb="Onboard recorder saturated: link degrades and the local send "
              "queue is backed up deep; clears under throttle.",
        outcome=Outcome.DEGRADED, measured_rate=900_000, expected_rate=8_000_000,
        queue_bytes=3 * 1024 * 1024, ms_since_ok=8 * MIN, shift_observed=0,
        beacon_ok=True,
    ),
    # ---- NOVEL 1: outside the six modeled causes --------------------------
    Case(
        name="NOVEL - RFI / UPLINK JAMMING (unmodeled)",
        truth=UNKNOWN,
        blurb="Narrowband RFI on the payload uplink corrupts the high-gain "
              "channel. Beacon is untouched, no other orbiter is affected, the "
              "queue is shallow, and nothing is time-shifted -- so NONE of the "
              "six signatures fire cleanly.",
        outcome=Outcome.DEGRADED, measured_rate=1_100_000, expected_rate=8_000_000,
        queue_bytes=16 * 1024, ms_since_ok=6 * MIN, shift_observed=0,
        beacon_ok=True,
    ),
    # ---- NOVEL 2: outside the six modeled causes --------------------------
    Case(
        name="NOVEL - SEU FRAME CORRUPTION (unmodeled)",
        truth=UNKNOWN,
        blurb="A single-event upset flips bits in the framer; the decoder drops "
              "the garbled frames so the pass reads silent, yet the beacon is "
              "healthy, no peer is co-failing, and the queue is shallow.",
        outcome=Outcome.SILENT, measured_rate=0, expected_rate=8_000_000,
        queue_bytes=20 * 1024, ms_since_ok=12 * MIN, shift_observed=0,
        beacon_ok=True,
    ),
]

NODE, PEER, T0 = "MRO-2", "DSN-Goldstone", 1_700_000_000_000

# How many times a persistent fault is observed before we read the belief. A
# real fault recurs across several contacts; one observation leaves the Bayes
# filter dominated by its strong NOMINAL prior. Several drive it to commit.
N_OBS = 8


def _obs(case: Case, t: int, channel: Channel) -> Observation:
    return Observation(
        t=t, node=NODE, peer=PEER, contact_id=f"c-{t}",
        outcome=(Outcome.OK if channel is Channel.BEACON and case.beacon_ok
                 else (Outcome.SILENT if channel is Channel.BEACON
                       else case.outcome)),
        measured_rate=case.measured_rate if channel is Channel.PRIMARY else 0,
        expected_rate=case.expected_rate,
        elevation_deg=case.elevation_deg,
        channel=channel,
        queue_bytes=case.queue_bytes,
        ms_since_ok=case.ms_since_ok,
        shift_observed=case.shift_observed,
        peer_degree=1, self_degree=1,
    )


def run_bayes(case: Case) -> dict:
    """Drive a FRESH Bayes filter with this fault, observed N_OBS times."""
    adv = BayesAdvisor()
    for k in range(N_OBS):
        t = T0 + k * MIN
        # Beacon reading arrives just before the primary pass, so its health is
        # in scope for the primary link's likelihood.
        adv.observe(_obs(case, t - 1000, Channel.BEACON))
        adv.observe(_obs(case, t, Channel.PRIMARY))
    belief = adv.belief_for(NODE, PEER)
    argmax = max(belief, key=belief.__getitem__)
    return {"argmax": argmax, "prob": belief[argmax], "belief": belief}


def run_gemma(case: Case) -> dict:
    """Ask the open-world Gemma on a single representative observation."""
    adv = OpenWorldGemmaAdvisor(cache_path=str(_ROOT / "out" / "gemma_openworld_cache.json"))
    t = T0 + (N_OBS - 1) * MIN
    adv.observe(_obs(case, t - 1000, Channel.BEACON))
    obs = _obs(case, t, Channel.PRIMARY)
    adv.observe(obs)
    return adv.diagnose(obs)


def main() -> int:
    transcript = []
    print("=" * 78)
    print("UNMODELED-FAULT DEMO -- open-world Gemma vs the closed Bayes filter")
    print("The Bayes filter has 6 hardcoded causes and NO way to say "
          "'none of the above'.")
    print("=" * 78)

    for case in CASES:
        b = run_bayes(case)
        g = run_gemma(case)

        novel = case.truth == UNKNOWN
        gemma_str = (f"G  UNKNOWN/ANOMALY  (conf {g['confidence']:.2f})"
                     if g["flagged"]
                     else f"{g['letter']}  {g['cause'].upper()}  "
                          f"(conf {g['confidence']:.2f})")
        bayes_str = f"{b['argmax'].upper()}  (p={b['prob']:.2f})"

        # Is Bayes forced into a WRONG known cause on a novel fault?
        bayes_wrong = novel and b["argmax"] != Cause.NOMINAL.value
        mark = "  <-- FORCED WRONG" if bayes_wrong else ""

        print()
        print("-" * 78)
        print(f"CASE: {case.name}")
        print(f"  truth: {case.truth}"
              + ("   (no Cause enum value -- unmodeled)" if novel else ""))
        print(f"  physical: {case.blurb}")
        print(f"  evidence: outcome={case.outcome.value} beacon_ok={case.beacon_ok} "
              f"queue={'deep' if case.queue_bytes >= 1_000_000 else 'shallow'} "
              f"shift={case.shift_observed} peer_cofail=none self_cofail=none")
        print()
        print(f"  OPEN-WORLD GEMMA -> {gemma_str}")
        if g.get("hypothesis"):
            print(f"                     hypothesis: {g['hypothesis']}")
        print(f"                     rationale: {g['rationale']}")
        print(f"  BAYES FILTER     -> {bayes_str}{mark}")
        if novel:
            if g["flagged"] and bayes_wrong:
                print(f"  ==> Gemma flagged the anomaly; Bayes was forced to call it "
                      f"'{b['argmax']}'. That bin does not exist for this fault.")
            elif g["flagged"]:
                print(f"  ==> Gemma flagged the anomaly; Bayes landed on "
                      f"'{b['argmax']}'.")
        else:
            agree = g["cause"] == case.truth
            print(f"  ==> control: Gemma {'agrees' if agree else 'differs'}, "
                  f"Bayes argmax = '{b['argmax']}'.")

        transcript.append({
            "case": case.name,
            "truth": case.truth,
            "novel": novel,
            "physical": case.blurb,
            "evidence": {
                "outcome": case.outcome.value,
                "beacon_ok": case.beacon_ok,
                "queue_bytes": case.queue_bytes,
                "queue_deep": case.queue_bytes >= 1_000_000,
                "ms_since_ok": case.ms_since_ok,
                "shift_observed": case.shift_observed,
            },
            "gemma_openworld": g,
            "bayes": b,
            "bayes_forced_wrong": bayes_wrong,
        })

    out = _ROOT / "out" / "unmodeled_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    print()
    print("=" * 78)
    print(f"Full transcript written to {out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
