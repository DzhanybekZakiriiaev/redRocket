"""Run the full default 24h scenario with GemmaPlusAdvisor and emit
out/trace_gemma_plus.json -- the same trace shape sim/trace.py build() produces,
so the frontend can load it with zero changes.

    python -m tools.run_gemma_plus

sim.trace.build() only accepts a string advisor_name and has no branch for
'gemma_plus', so rather than modify it we replicate its body here and inject a
GemmaPlusAdvisor instance. Every non-advisor step (contacts, faults, strip,
positions, node list, belief series, gossip/fails, summary) is reused verbatim
from sim.trace / sim.contacts / sim.diagnosability.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sim.config import DEFAULT, ScenarioConfig            # noqa: E402
from sim.types import CAUSES                               # noqa: E402
from sim import contacts as C                              # noqa: E402
from sim import diagnosability as D                        # noqa: E402
from sim import faults as F                                # noqa: E402
from sim.engine import Engine                              # noqa: E402
from sim.trace import _positions                           # noqa: E402  (reused)
from sim.advisor.gemma_plus import GemmaPlusAdvisor        # noqa: E402


def build_gemma_plus(cfg: ScenarioConfig = DEFAULT) -> dict:
    """A copy of sim.trace.build() with a GemmaPlusAdvisor instance injected.

    Kept byte-for-byte compatible with build()'s output so the frontend needs
    no branch of its own.
    """
    cs = C.read(cfg)
    fs = F.generate(cs, cfg)
    strip = D.compute(cs, cfg)
    grid = strip.t_grid

    advisor = GemmaPlusAdvisor()          # <-- the only substantive change
    eng = Engine(cs, fs, advisor, cfg)
    summary = eng.run()

    stations = {s["id"]: s for s in C.load_stations(cfg)}
    all_stations = json.loads(
        cfg.data_path(cfg.stations_file).read_text(encoding="utf-8"))["stations"]

    nodes = [{"id": C.sat_id(i), "type": "sat"} for i in range(cfg.n_sats)]
    for s in all_stations:
        nodes.append({
            "id": s["id"], "name": s["name"], "type": "gs",
            "lat": s["lat"], "lon": s["lon"],
            "operational": s["id"] in stations,
        })

    order = [c.value for c in CAUSES]
    by_node: dict[str, list] = {}
    for b in eng.beliefs:
        by_node.setdefault(b["node"], []).append({
            "t": b["t"],
            "d": [round(b["belief"].get(k, 0.0), 4) for k in order],
            "a": b["action"], "tg": b["target"],
            "x": b["argmax"], "truth": b["truth"],
            "r": b.get("rationale", ""),
        })

    gossip = [e for e in eng.events if e["type"] == "gossip"]
    fails = [e for e in eng.events if e["type"] == "contact_fail"]

    return {
        "meta": {
            "epoch_ms": grid[0],
            "grid_ms": cfg.grid_s * 1000,
            "n_frames": len(grid),
            "horizon_ms": cfg.horizon_ms,
            "causes": order,
            "advisor": summary["advisor"],
            "config_hash": cfg.hash,
        },
        "nodes": nodes,
        "positions": _positions(cfg, grid),
        "contacts": [
            {"a": c.src, "b": c.dst, "t0": c.t_open, "t1": c.t_close, "k": c.kind}
            for c in cs if c.src < c.dst
        ],
        "faults": [f.to_dict() for f in fs],
        "beliefs": by_node,
        "gossip": [{"t": g["t"], "f": g["from"], "to": g["to"], "n": g["n"]}
                   for g in gossip],
        "fails": [{"t": f["t"], "a": f["src"], "b": f["dst"], "m": f["mode"]}
                  for f in fails],
        "strip": {k: v for k, v in strip.to_dict()["per_link"].items()},
        "summary": summary,
    }


def main() -> int:
    cfg = DEFAULT
    t = time.time()
    blob = build_gemma_plus(cfg)
    path = cfg.out_path("trace_gemma_plus.json")
    path.write_text(json.dumps(blob, separators=(",", ":")), encoding="utf-8")
    kb = path.stat().st_size / 1024
    print(f"{path}  {kb:.0f} KB  in {time.time()-t:.1f}s")
    print(f"  frames    {blob['meta']['n_frames']}")
    print(f"  advisor   {blob['meta']['advisor']}")
    print(f"  faults    {len(blob['faults'])}")
    print(f"  fails     {len(blob['fails'])}")
    print(f"  strip     {len(blob['strip'])} links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
