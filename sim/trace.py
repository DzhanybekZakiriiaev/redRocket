"""Trace emission -- the one file the frontend reads. PLAN.md section 4.6.

Everything the demo needs is baked in here: satellite positions on the 60 s
grid, contact windows, faults, per-node belief series, gossip exchanges, and the
diagnosability strip. No TLE parsing in the browser, no SGP4, no second fetch,
nothing to go wrong on demo day.

Positions are lat/lon/alt because that is what a globe wants. The frontend
interpolates between frames; at a 60 s grid an Iridium satellite moves ~3.8 deg
of arc, and linear interpolation across that is visually indistinguishable from
the true arc at render scale.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from skyfield.api import load, wgs84

from .config import ScenarioConfig, DEFAULT
from .types import Contact, FaultEvent, CAUSES
from . import contacts as C
from . import diagnosability as D


def _positions(cfg: ScenarioConfig, grid: list[int]) -> dict[str, list]:
    """lat/lon/alt_km per satellite at every grid point."""
    ts = load.timescale(builtin=True)
    sats = C.load_satellites(cfg, ts)

    t0 = datetime.fromtimestamp(grid[0] / 1000, tz=timezone.utc)
    base = ts.utc(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)
    offs = np.array([(g - grid[0]) / 86400_000.0 for g in grid])
    times = ts.tt_jd(base.tt + offs)

    out: dict[str, list] = {}
    for i, sat in enumerate(sats):
        sp = wgs84.subpoint(sat.at(times))
        lat = np.round(sp.latitude.degrees, 3)
        lon = np.round(sp.longitude.degrees, 3)
        alt = np.round(sp.elevation.km, 1)
        out[C.sat_id(i)] = [
            [float(a), float(b), float(c)] for a, b, c in zip(lat, lon, alt)
        ]
    return out


def build(cfg: ScenarioConfig = DEFAULT, *, advisor_name: str = "bayes"):
    """Run the full pipeline and assemble the trace."""
    from .engine import Engine
    from . import faults as F
    from .advisor.bayes import BayesAdvisor, NullAdvisor

    cs = C.read(cfg)
    fs = F.generate(cs, cfg)
    strip = D.compute(cs, cfg)
    grid = strip.t_grid

    if advisor_name == "bayes":
        advisor = BayesAdvisor()
    elif advisor_name == "gemma":
        from .advisor.gemma import GemmaAdvisor
        advisor = GemmaAdvisor()
    elif advisor_name == "null":
        advisor = NullAdvisor()
    else:
        raise ValueError(f"unknown advisor: {advisor_name!r} (use bayes|gemma|null)")
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
            # Stations beyond n_stations exist for visual density only -- the
            # globe stays populated without inflating network coverage, which
            # is what quietly killed the delay-tolerance premise once already
            # (FINDINGS F-017).
            "operational": s["id"] in stations,
        })

    # Belief series keyed by node, aligned to the grid so the frontend can index
    # by frame rather than searching by timestamp.
    order = [c.value for c in CAUSES]
    by_node: dict[str, list] = {}
    for b in eng.beliefs:
        by_node.setdefault(b["node"], []).append({
            "t": b["t"],
            "d": [round(b["belief"].get(k, 0.0), 4) for k in order],
            "a": b["action"], "tg": b["target"],
            "x": b["argmax"], "truth": b["truth"],
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
            {"a": c.src, "b": c.dst, "t0": c.t_open, "t1": c.t_close,
             "k": c.kind}
            for c in cs if c.src < c.dst  # one record per undirected window
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


def write(cfg: ScenarioConfig = DEFAULT, name: str = "trace.json",
          *, advisor_name: str = "bayes") -> str:
    blob = build(cfg, advisor_name=advisor_name)
    path = cfg.out_path(name)
    path.write_text(json.dumps(blob, separators=(",", ":")), encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    import sys, time, os
    advisor_name = sys.argv[1] if len(sys.argv) > 1 else "bayes"
    name = "trace.json" if advisor_name == "bayes" else f"trace_{advisor_name}.json"
    t = time.time()
    p = write(name=name, advisor_name=advisor_name)
    kb = os.path.getsize(p) / 1024
    blob = json.loads(open(p, encoding="utf-8").read())
    print(f"{p}  {kb:.0f} KB  in {time.time()-t:.1f}s")
    print(f"  frames    {blob['meta']['n_frames']}")
    print(f"  nodes     {len(blob['nodes'])} "
          f"({sum(1 for n in blob['nodes'] if n['type']=='sat')} sats, "
          f"{sum(1 for n in blob['nodes'] if n.get('operational'))} operational GS)")
    print(f"  contacts  {len(blob['contacts'])}")
    print(f"  faults    {len(blob['faults'])}")
    print(f"  gossip    {len(blob['gossip'])}")
    print(f"  fails     {len(blob['fails'])}")
    print(f"  strip     {len(blob['strip'])} links")
