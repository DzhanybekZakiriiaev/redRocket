"""Run the reasoning-first Gemma arm and score it against the baseline.

Writes web/public/trace_gemma_cot.json -- a NEW file. The tuned demo trace
(trace_gemma.json) is never touched, so a worse result costs nothing but the
API calls and you can look at the number before deciding to swap.

Each decision carries its verbatim prompt, the raw reply, and the elimination
steps the model produced BEFORE naming a cause, so the UI can show reasoning
that genuinely conditioned the answer rather than a caption written after it.

    python -m tools.run_gemma_cot
"""

from __future__ import annotations

import json
from pathlib import Path

from sim.config import DEFAULT
from sim.advisor.gemma_cot import GemmaCotAdvisor
from sim import trace as T
from tools.replay_evidence import _fields

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "web" / "public" / "trace_gemma.json"
OUT = ROOT / "web" / "public" / "trace_gemma_cot.json"


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


def main() -> int:
    adv = GemmaCotAdvisor()
    print(f"[cot] running the reasoning-first arm against {adv.model} ...")
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
            }
            attached += 1

    # Re-derive the evidence field table from the same observations. Cheap and
    # keeps the panel's layout identical across both arms.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(blob, separators=(",", ":")), encoding="utf-8")

    c1, t1 = accuracy(blob)
    print(f"[cot] wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"[cot] reasoning attached to {attached} decisions")
    print()
    print("  ARM            ACCURACY ON FAULTED SAMPLES")
    if BASELINE.exists():
        base = json.loads(BASELINE.read_text(encoding="utf-8"))
        c0, t0 = accuracy(base)
        print(f"  gemma          {c0/t0*100:5.1f}%   ({c0}/{t0})")
        print(f"  gemma_cot      {c1/t1*100:5.1f}%   ({c1}/{t1})")
        d = (c1 / t1 - c0 / t0) * 100
        print()
        if abs(c1 - c0) <= 2:
            print(f"  delta {d:+.1f} pts on {abs(c1-c0)} sample(s) -- a wash, not a result. "
                  f"Do not claim an improvement from this.")
        else:
            print(f"  delta {d:+.1f} pts ({c1-c0:+d} samples).")
    else:
        print(f"  gemma_cot      {c1/t1*100:5.1f}%   ({c1}/{t1})   (no baseline to compare)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
