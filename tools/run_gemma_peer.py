"""Run the peer cross-check Gemma arm and score its CALIBRATION, not just its accuracy.

Writes web/public/trace_gemma_peer.json -- a NEW file. trace_gemma.json and
trace_gemma_cot.json are never touched, so a worse result costs nothing but the
API calls.

The headline number here is deliberately NOT accuracy. Both existing Gemma arms
sit at zero belief samples below the frontend's UNCERTAIN threshold
(CONF = 0.65 in web/src/trace.ts) across all 2304 samples -- the model never
reports doubt, including where the diagnosability strip proves the evidence
cannot decide. This arm exists to produce honest doubt when peers disagree. If
its sub-threshold count is still zero, it failed at the thing it was built for,
and this script says so in as many words rather than quietly reporting accuracy
instead.

    python -m tools.run_gemma_peer
"""

from __future__ import annotations

import json
from pathlib import Path

from sim.config import DEFAULT
from sim.advisor.gemma_peer import GemmaPeerAdvisor
from sim import trace as T

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "web" / "public"
ARMS = [("gemma", PUB / "trace_gemma.json"),
        ("gemma_cot", PUB / "trace_gemma_cot.json")]
OUT = PUB / "trace_gemma_peer.json"

# The frontend's UNCERTAIN threshold. Hardcoded there as CONF in
# web/src/trace.ts; a belief whose top probability sits below it renders the
# node as UNCERTAIN instead of a confident diagnosis.
CONF = 0.65


def accuracy(blob: dict) -> tuple[int, int]:
    """Correct / total over samples whose link was genuinely faulted."""
    correct = total = 0
    for series in blob["beliefs"].values():
        for b in series:
            if b["truth"] != "nominal":
                total += 1
                if b["x"] == b["truth"]:
                    correct += 1
    return correct, total


def uncertain(blob: dict) -> tuple[int, int, int, int]:
    """(sub-threshold, all samples, sub-threshold on faulted, faulted)."""
    sub = tot = sub_f = tot_f = 0
    for series in blob["beliefs"].values():
        for b in series:
            low = max(b["d"]) < CONF
            tot += 1
            sub += low
            if b["truth"] != "nominal":
                tot_f += 1
                sub_f += low
    return sub, tot, sub_f, tot_f


def _pct(n: int, d: int) -> str:
    return f"{n / d * 100:5.1f}%" if d else "    -"


def main() -> int:
    adv = GemmaPeerAdvisor()
    print(f"[peer] running the peer cross-check arm against {adv.model} ...")
    blob = T.build(DEFAULT, advisor=adv)

    attached = 0
    for node, series in blob["beliefs"].items():
        for b in series:
            rec = adv.records.get((node, int(b["t"])))
            if rec is None:
                continue
            b["ev"] = {
                "peer": rec["peer"],
                "prompt": rec["prompt"],
                "raw": rec["raw"],
                "steps": rec["steps"],
                "peers": rec["peers"],
                "consensus": rec["consensus"],
            }
            attached += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(blob, separators=(",", ":")), encoding="utf-8")
    print(f"[peer] wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"[peer] reasoning + peer reports attached to {attached} decisions")

    # ---- load the other arms for comparison -------------------------------
    loaded: list[tuple[str, dict]] = []
    for name, path in ARMS:
        if path.exists():
            loaded.append((name, json.loads(path.read_text(encoding="utf-8"))))
        else:
            print(f"[peer] note: {path.name} missing -- {name} omitted from the table")
    loaded.append(("gemma_peer", blob))

    # ---- accuracy ---------------------------------------------------------
    print()
    print("  ARM            ACCURACY ON FAULTED SAMPLES")
    accs: dict[str, tuple[int, int]] = {}
    for name, b in loaded:
        c, t = accuracy(b)
        accs[name] = (c, t)
        print(f"  {name:<13}  {_pct(c, t)}   ({c}/{t})")

    cp, tp = accs["gemma_peer"]
    print()
    for name in ("gemma", "gemma_cot"):
        if name not in accs:
            continue
        c0, t0 = accs[name]
        d = (cp / tp - c0 / t0) * 100 if tp and t0 else 0.0
        if abs(cp - c0) <= 2:
            print(f"  vs {name:<10} delta {d:+.1f} pts on {abs(cp - c0)} sample(s) "
                  f"-- a wash, not a result. Do not claim an improvement from this.")
        else:
            print(f"  vs {name:<10} delta {d:+.1f} pts ({cp - c0:+d} samples).")

    # ---- THE metric -------------------------------------------------------
    print()
    print("  " + "=" * 68)
    print(f"  CRITICAL METRIC -- belief samples below the UNCERTAIN threshold "
          f"({CONF})")
    print("  A node only displays doubt when its top probability falls under it.")
    print("  " + "=" * 68)
    print("  ARM             ALL SAMPLES              FAULTED SAMPLES")
    unc: dict[str, tuple[int, int, int, int]] = {}
    for name, b in loaded:
        s, t, sf, tf = uncertain(b)
        unc[name] = (s, t, sf, tf)
        print(f"  {name:<13}  {s:5d} / {t:<5d} {_pct(s, t)}      "
              f"{sf:4d} / {tf:<4d} {_pct(sf, tf)}")

    # ---- peer cross-check activity ----------------------------------------
    recs = list(adv.records.values())
    with_peers = [r for r in recs if r["peers"]]
    disagreed = [r for r in with_peers if not r["consensus"]]
    print()
    print(f"  peer cross-check: {len(recs)} diagnoses, {len(with_peers)} had at "
          f"least one peer report "
          f"({_pct(len(with_peers), len(recs)).strip()} of diagnoses)")
    print(f"                    of those, {len(disagreed)} disagreed "
          f"({_pct(len(disagreed), len(with_peers)).strip()} disagreement rate)")
    if with_peers:
        n_reports = sum(len(r["peers"]) for r in with_peers)
        print(f"                    {n_reports} peer reports quoted in total, "
              f"{n_reports / len(with_peers):.1f} per consulted decision")

    # ---- verdict ----------------------------------------------------------
    s, t, sf, tf = unc["gemma_peer"]
    print()
    if s == 0:
        print("  *** THE FEATURE FAILED ITS PURPOSE. ***")
        print("  gemma_peer produced ZERO belief samples below the threshold, "
              "same as the")
        print("  arms it was meant to improve on. No node will ever render "
              "UNCERTAIN on this")
        print("  trace. Do not present this as calibrated. The cause is one of:")
        print("    - peers rarely diagnose the same target within the window "
              f"({len(with_peers)}/{len(recs)} consulted),")
        print(f"    - or they agree when they do ({len(disagreed)} "
              "disagreements),")
        print("    - or the consensus spread is not wide enough to cross 0.65.")
        print("  The counts above say which. Do NOT tune the floor or the "
              "threshold to")
        print("  manufacture a number -- that would be fitting the metric, not "
              "the model.")
    else:
        base = max((unc[n][0] for n, _ in loaded if n != "gemma_peer"), default=0)
        print(f"  gemma_peer put {s} belief sample(s) below {CONF}; the other arms "
              f"put {base}.")
        if len(disagreed) <= 2:
            print(f"  WARNING: this rests on {len(disagreed)} disagreeing "
                  f"decision(s) -- a wash, not a result.")
            print("  Report the disagreement rate, do not claim calibration was "
                  "fixed.")
        else:
            print(f"  It comes from {len(disagreed)} decisions where peers "
                  f"reached incompatible")
            print("  conclusions about the same target. Uncertainty is produced "
                  "by disagreement")
            print("  between conclusions -- never by merging peer evidence into "
                  "the posterior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
