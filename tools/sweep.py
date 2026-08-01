"""Topology tuning sweep. PLAN.md section 5.1.

Sweeps {8,12,16} sats x {1,2,3} adjacent planes x {3000,4500,6000} km ISL.

THE TWO AXES ARE INDEPENDENT, AND THAT IS A RESULT, NOT AN ASSUMPTION.
Since diagnosability scopes concurrent degree WITHIN link kind (FINDINGS
F-001 -- a healthy crosslink cannot discriminate a downlink fault, different
antenna), `downlink_identifiable_in_window` is computed from downlink windows
alone, and downlink windows come out of find_events, which never sees
isl_max_km. So ISL range cannot move strip structure by construction, and the
sweep is really two separate one-dimensional questions:

    strip structure   = f(n_sats, n_planes)              -- sparsity
    gossip propagation = f(n_sats, n_planes, isl_max_km) -- connectivity

Section 3 below tests the invariance claim adversarially over a much wider
range than the sweep grid (2000 - 20000 km) rather than taking it on faith.

Gossip reachability: treating ISL contacts as time-respecting edges, what
fraction of satellite pairs are joined by a temporally ordered path inside a
6 h window from the epoch?

Everything is computed IN MEMORY. Nothing is written -- out/ holds the current
locked artifacts and this script must not clobber them.

Run from the project root:

    python tools/sweep.py
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import contacts as C            # noqa: E402
from sim import diagnosability as D      # noqa: E402
from sim.config import DEFAULT, ScenarioConfig  # noqa: E402
from sim.types import Contact            # noqa: E402

# --- the grid -------------------------------------------------------------
N_SATS = (8, 12, 16)
N_PLANES = (1, 2, 3)
ISL_MAX_KM = (3000.0, 4500.0, 6000.0)

# Section 3 probe: deliberately far outside the grid in both directions.
PROBE_KM = (2000.0, 3000.0, 4500.0, 6000.0, 8000.0, 12000.0, 20000.0)

GOSSIP_WINDOW_MS = 6 * 3600 * 1000     # PLAN 5.1: "within the scenario window"
SLOW_CONFIG_S = 20.0                   # flag anything above this

# A hop costs nothing in the literal time-respecting-path definition, and on a
# near-continuous in-plane chain that makes every well-connected configuration
# report t50 = 0: the whole chain is already open at the epoch, so one message
# crosses all of it instantaneously. True, but it discriminates nothing. The
# second figure charges one gossip exchange per hop (60 s, the belief-emission
# grid) and has to fit inside the contact window, which is what actually
# separates "the chain is up" from "you wait for windows to open".
HOP_MS = 60_000

BASELINE = (12, 2, 4500.0)

# Selection criteria, PLAN 5.1 + the brief.
REACH_FLOOR = 0.80                     # gossip must actually propagate
BAND_LO, BAND_HI = 0.30, 0.80          # useful identifiability band
MIXED_DOMINATES = 0.50                 # downlinks_mixed / downlink_links

# Physical note used in the isl_max_km recommendation. Two co-planar satellites
# at r ~ 7158 km cannot see each other past a chord of 2*sqrt(r^2 - graze^2)
# ~ 6174 km -- the Earth is in the way regardless of what isl_max_km says.
R_ORBIT_KM = 7158.0
INPLANE_CHORD_LIMIT_KM = 2.0 * math.sqrt(R_ORBIT_KM ** 2 - C.GRAZE_RADIUS_KM ** 2)


# --------------------------------------------------------------------------
# Gossip reachability over the temporal ISL graph
# --------------------------------------------------------------------------

def _epoch_ms(cfg: ScenarioConfig) -> int:
    dt = datetime.fromisoformat(cfg.epoch_iso.replace("Z", "+00:00"))
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def isl_edges(contacts: list[Contact], t0: int, t_end: int):
    """Directed ISL contacts clipped to [t0, t_end], sorted by t_open.

    Contacts are emitted in both directions by contacts.py, so this is already
    a directed edge list -- no need to mirror anything.
    """
    edges = []
    for c in contacts:
        if c.kind != "isl":
            continue
        if c.t_close < t0 or c.t_open > t_end:
            continue
        edges.append((c.src, c.dst, max(c.t_open, t0), min(c.t_close, t_end)))
    edges.sort(key=lambda e: e[2])
    return edges


def earliest_arrival(edges, nodes, source: str, t0: int,
                     hop_ms: int = 0) -> dict[str, float]:
    """Earliest time a message leaving `source` at t0 can be AT each node.

    Forward relaxation over contacts sorted by t_open: an edge (u,v) open over
    [o,c] is usable iff we are already at u by c, and then we arrive at v at
    max(arrival[u], o) -- you wait for the window to open, you cannot arrive
    before it does. With hop_ms > 0 the exchange also has to finish before the
    window closes.

    The scan is repeated to a fixed point. One pass is NOT sufficient with
    interval-valued edges: an edge processed early may become usable only after
    a later-opening edge improves its source's arrival time. Time-respecting
    paths never revisit a node profitably, so this converges in <= |nodes|
    passes; in practice it takes two or three. Verified equal to a Dijkstra
    earliest-arrival on every configuration in the grid.
    """
    inf = math.inf
    best: dict[str, float] = {n: inf for n in nodes}
    best[source] = float(t0)

    for _ in range(len(nodes) + 1):
        changed = False
        for u, v, t_open, t_close in edges:
            au = best[u]
            if au > t_close:            # missed this window entirely
                continue
            start = t_open if t_open > au else au
            arrive = start + hop_ms
            if arrive > t_close:        # not enough window left to exchange
                continue
            if arrive < best[v]:
                best[v] = arrive
                changed = True
        if not changed:
            break
    return best


def gossip_metrics(contacts: list[Contact], cfg: ScenarioConfig) -> dict:
    """Mean pairwise reachability inside the window, and time-to-half."""
    t0 = _epoch_ms(cfg)
    t_end = t0 + GOSSIP_WINDOW_MS
    nodes = sorted({C.sat_id(i) for i in range(cfg.n_sats)})
    edges = isl_edges(contacts, t0, t_end)

    n = len(nodes)
    half = math.ceil(0.5 * n)           # incl. the source itself
    reach_fracs: list[float] = []
    t50s: list[float] = []
    t50s_hop: list[float] = []
    reach_hop: list[float] = []

    for src in nodes:
        for hop, rf, ts in ((0, reach_fracs, t50s), (HOP_MS, reach_hop, t50s_hop)):
            best = earliest_arrival(edges, nodes, src, t0, hop_ms=hop)
            reached = sum(1 for x in nodes if x != src and best[x] <= t_end)
            rf.append(reached / (n - 1) if n > 1 else 1.0)
            # k-th smallest arrival is the moment the k-th node has the message.
            t_half = sorted(best[x] for x in nodes)[half - 1]
            ts.append((t_half - t0) / 60000.0 if t_half <= t_end else math.inf)

    return {
        "mean_reach": float(np.mean(reach_fracs)),
        "min_reach": float(np.min(reach_fracs)),
        "median_t50_min": _median(t50s),
        "mean_reach_hop": float(np.mean(reach_hop)),
        "median_t50_hop_min": _median(t50s_hop),
        "n_src_no_half": sum(1 for v in t50s if math.isinf(v)),
        "isl_edges_in_window": len(edges),
    }


def _median(vals: list[float]) -> float:
    """statistics.median averages the middle pair, which turns one inf into an
    inf median on even-length input. Take the lower middle instead."""
    s = sorted(vals)
    return s[(len(s) - 1) // 2]


# --------------------------------------------------------------------------
# Structural counts
# --------------------------------------------------------------------------

def structure_metrics(contacts: list[Contact], cfg: ScenarioConfig) -> dict:
    grid = D.time_grid(cfg)
    isl_deg = D.degree_matrix(contacts, grid, kind="isl")
    sats = [C.sat_id(i) for i in range(cfg.n_sats)]
    zeros = np.zeros(len(grid), dtype=np.int16)
    stack = np.stack([isl_deg.get(s, zeros) for s in sats])

    def undirected(kind):
        return {(min(c.src, c.dst), max(c.src, c.dst), c.t_open, c.t_close)
                for c in contacts if c.kind == kind}

    isl_win = undirected("isl")
    dl_win = undirected("downlink")
    pairs = {(a, b) for a, b, _, _ in isl_win}
    n_pairs = cfg.n_sats * (cfg.n_sats - 1) // 2

    return {
        "total_contacts": len(contacts),
        "isl_windows": len(isl_win),
        "downlink_windows": len(dl_win),
        "mean_isl_degree": float(stack.mean()),
        "isl_pairs_never": 1.0 - (len(pairs) / n_pairs if n_pairs else 1.0),
    }


# --------------------------------------------------------------------------
# One configuration
# --------------------------------------------------------------------------

def run_one(n_sats: int, n_planes: int, isl_km: float) -> dict:
    cfg = replace(DEFAULT, n_sats=n_sats, n_planes=n_planes, isl_max_km=isl_km)
    row = {"n_sats": n_sats, "n_planes": n_planes, "isl_max_km": isl_km,
           "error": None}
    t = time.time()
    try:
        cs = C.build(cfg)                       # in memory; nothing written
        strip = D.compute(cs, cfg)
        row.update(D.summarize(strip, cs))
        row.update(structure_metrics(cs, cfg))
        row.update(gossip_metrics(cs, cfg))
    except Exception as exc:                    # e.g. not enough sats in 1 plane
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["secs"] = time.time() - t
    return row


def _f(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "never" if math.isinf(v) else f"{v:.{nd}f}"
    return str(v)


def mixed_frac(row: dict) -> float:
    d = row.get("downlink_links") or 0
    return (row.get("downlinks_mixed", 0) / d) if d else 0.0


# --------------------------------------------------------------------------
# Section 1 -- strip structure, a function of (n_sats, n_planes) only
# --------------------------------------------------------------------------

STRIP_KEYS = ("downlink_identifiable_in_window", "downlinks_always_identifiable",
              "downlinks_never_identifiable", "downlinks_mixed",
              "downlink_links", "isl_identifiable_in_window")


def section_strip(rows: list[dict]) -> list[dict]:
    print("\n## 1. Strip structure = f(n_sats, n_planes)\n")
    print("Collapsed over isl_max_km. The `range spread` column is the max "
          "minus min of\n`downlink_identifiable_in_window` across "
          f"{{{', '.join(f'{k:.0f}' for k in ISL_MAX_KM)}}} km -- "
          "it must be exactly 0.\n")
    print("| sats | planes | dl_ident | always | never | mixed | mixed frac | "
          "downlinks | isl_ident | range spread |")
    print("|" + "---|" * 10)

    out = []
    for n_sats in N_SATS:
        for n_planes in N_PLANES:
            grp = [r for r in rows
                   if r["n_sats"] == n_sats and r["n_planes"] == n_planes]
            ok = [r for r in grp if not r["error"]]
            if not ok:
                print(f"| {n_sats} | {n_planes} | "
                      f"FAILED: {grp[0]['error']} |" + " - |" * 8)
                continue
            vals = [r["downlink_identifiable_in_window"] for r in ok]
            spread = max(vals) - min(vals)
            r = ok[0]
            flag = "" if spread == 0.0 else f"  <-- NOT INVARIANT ({spread:.2e})"
            print(f"| {n_sats} | {n_planes} | "
                  f"{r['downlink_identifiable_in_window']:.4f} | "
                  f"{r['downlinks_always_identifiable']} | "
                  f"{r['downlinks_never_identifiable']} | "
                  f"{r['downlinks_mixed']} | {mixed_frac(r):.3f} | "
                  f"{r['downlink_links']} | "
                  f"{r['isl_identifiable_in_window']:.3f} | "
                  f"{spread:.1e}{flag} |")
            out.append(r)
    return out


# --------------------------------------------------------------------------
# Section 2 -- gossip propagation, a function of all three
# --------------------------------------------------------------------------

def section_gossip(rows: list[dict]) -> None:
    print("\n## 2. Gossip reachability = f(n_sats, n_planes, isl_max_km)\n")
    print("`mean reach` is the fraction of ordered satellite pairs joined by a "
          "time-respecting\nISL path within 6 h of the epoch. `t50` is the "
          "median over sources of the time to\nreach half the constellation; "
          "the second t50 charges 60 s per hop.\n")
    print("| sats | planes | ISL km | mean reach 6h | min reach | t50 | "
          "t50 +60s/hop | ISL win | mean ISL deg | pairs never linked | gate |")
    print("|" + "---|" * 11)
    for r in rows:
        if r["error"]:
            print(f"| {r['n_sats']} | {r['n_planes']} | {r['isl_max_km']:.0f} | "
                  f"FAILED: {r['error']} |" + " - |" * 7)
            continue
        gate = "PASS" if r["mean_reach"] >= REACH_FLOOR else "fail"
        mark = " *" if (r["n_sats"], r["n_planes"], r["isl_max_km"]) == BASELINE else ""
        print(f"| {r['n_sats']}{mark} | {r['n_planes']} | {r['isl_max_km']:.0f} | "
              f"{r['mean_reach']:.3f} | {r['min_reach']:.3f} | "
              f"{_f(r['median_t50_min'], 1)} | {_f(r['median_t50_hop_min'], 1)} | "
              f"{r['isl_windows']} | {r['mean_isl_degree']:.2f} | "
              f"{r['isl_pairs_never']:.2f} | {gate} |")


# --------------------------------------------------------------------------
# Section 3 -- adversarial test of the isl_max_km invariance claim
# --------------------------------------------------------------------------

def section_invariance() -> bool:
    print("\n## 3. Does isl_max_km move the strip? Probe over "
          f"{PROBE_KM[0]:.0f}-{PROBE_KM[-1]:.0f} km\n")
    print("Structural prediction: no. `compute` takes a downlink link's degree "
          "from\n`deg_by_kind['downlink']`, and downlink windows come from "
          "find_events, which never\nsees isl_max_km. Nothing in that path can "
          "vary. Testing it anyway.\n")
    print("| sats | planes | " + " | ".join(f"{k:.0f} km" for k in PROBE_KM)
          + " | spread | ISL win at min/max km |")
    print("|" + "---|" * (len(PROBE_KM) + 4))

    all_flat = True
    for n_sats in N_SATS:
        for n_planes in N_PLANES:
            vals, isl_counts = [], []
            for km in PROBE_KM:
                cfg = replace(DEFAULT, n_sats=n_sats, n_planes=n_planes,
                              isl_max_km=km)
                try:
                    cs = C.build(cfg)
                except Exception:
                    vals = None
                    break
                s = D.summarize(D.compute(cs, cfg), cs)
                vals.append(s["downlink_identifiable_in_window"])
                isl_counts.append(structure_metrics(cs, cfg)["isl_windows"])
            if vals is None:
                continue
            spread = max(vals) - min(vals)
            if spread != 0.0:
                all_flat = False
            flag = "" if spread == 0.0 else "  <-- MOVES"
            print(f"| {n_sats} | {n_planes} | "
                  + " | ".join(f"{v:.4f}" for v in vals)
                  + f" | {spread:.1e}{flag} | {isl_counts[0]} / {isl_counts[-1]} |")
    return all_flat


# --------------------------------------------------------------------------

def main() -> None:
    rows: list[dict] = []
    t_all = time.time()
    for n_sats in N_SATS:
        for n_planes in N_PLANES:
            for isl_km in ISL_MAX_KM:
                r = run_one(n_sats, n_planes, isl_km)
                rows.append(r)
                tag = ("FAIL " + (r["error"] or "")) if r["error"] else (
                    f"dl_ident={r['downlink_identifiable_in_window']:.4f} "
                    f"reach={r['mean_reach']:.3f}")
                print(f"[{len(rows):2d}/27] {n_sats:2d}sat {n_planes}pl "
                      f"{isl_km:.0f}km  {r['secs']:5.1f}s  {tag}",
                      file=sys.stderr, flush=True)

    section_strip(rows)
    section_gossip(rows)
    flat = section_invariance()

    total = time.time() - t_all
    slow = [r for r in rows if r["secs"] > SLOW_CONFIG_S]
    print(f"\nSweep grid {sum(r['secs'] for r in rows):.1f}s, whole script "
          f"{total:.1f}s; slowest single config "
          f"{max(r['secs'] for r in rows):.1f}s; "
          f"{len(slow)} config(s) over {SLOW_CONFIG_S:.0f}s.")
    print("isl_max_km invariance over "
          f"{PROBE_KM[0]:.0f}-{PROBE_KM[-1]:.0f} km: "
          + ("CONFIRMED, spread exactly 0 everywhere."
             if flat else "REFUTED -- see the flagged rows above."))

    # --- recommendation, per axis -----------------------------------------
    print("\n## 4. Recommendation\n")
    ok = [r for r in rows if not r["error"]]

    cand: dict[tuple[int, int], dict] = {}
    for r in ok:
        cand.setdefault((r["n_sats"], r["n_planes"]), r)

    def strip_ok(r):
        return (BAND_LO <= r["downlink_identifiable_in_window"] <= BAND_HI
                and mixed_frac(r) >= MIXED_DOMINATES)

    def best_km(key):
        """Smallest isl_max_km in the grid clearing the reach gate, else None."""
        fam = sorted((r for r in ok if (r["n_sats"], r["n_planes"]) == key),
                     key=lambda r: r["isl_max_km"])
        for r in fam:
            if r["mean_reach"] >= REACH_FLOOR:
                return r["isl_max_km"]
        return None

    # Fewest sats, then >=2 planes (PLAN 5.1 wants cross-plane intermittency),
    # then at least one permanently unidentifiable downlink (real red on the
    # strip), then most mixed.
    order = (lambda r: (r["n_sats"],
                        0 if r["n_planes"] >= 2 else 1,
                        0 if r["downlinks_never_identifiable"] >= 1 else 1,
                        -mixed_frac(r)))

    print("Strip axis alone (band + mixed dominance, fewest sats). NOTE this "
          "axis is\nblind to gossip -- feasibility is checked after:\n")
    for r in sorted([r for r in cand.values() if strip_ok(r)], key=order)[:6]:
        km = best_km((r["n_sats"], r["n_planes"]))
        print(f"  n_sats={r['n_sats']} n_planes={r['n_planes']}  "
              f"dl_ident={r['downlink_identifiable_in_window']:.4f}  "
              f"mixed={r['downlinks_mixed']}/{r['downlink_links']}  "
              f"never={r['downlinks_never_identifiable']}  "
              + (f"gossip OK at {km:.0f} km" if km
                 else "NO isl_max_km IN GRID CLEARS THE GOSSIP GATE"))

    feasible = [r for r in cand.values()
                if strip_ok(r) and best_km((r["n_sats"], r["n_planes"]))]
    if not feasible:
        print("\nNothing satisfies both axes.")
        return
    pick = sorted(feasible, key=order)[0]
    key = (pick["n_sats"], pick["n_planes"])

    print(f"\nGossip axis for the chosen n_sats={pick['n_sats']} "
          f"n_planes={pick['n_planes']}:")
    chosen_km = best_km(key)
    for r in sorted((r for r in ok if (r["n_sats"], r["n_planes"]) == key),
                    key=lambda r: r["isl_max_km"]):
        print(f"  {r['isl_max_km']:.0f} km: reach={r['mean_reach']:.3f} "
              f"min={r['min_reach']:.3f} isl_deg={r['mean_isl_degree']:.2f} "
              f"isl_win={r['isl_windows']} "
              f"{'PASS' if r['mean_reach'] >= REACH_FLOOR else 'fail'}"
              f"{'  <- smallest passing' if r['isl_max_km'] == chosen_km else ''}")
    print(f"\n  In-plane crosslinks are capped near "
          f"{INPLANE_CHORD_LIMIT_KM:.0f} km by Earth occlusion, so range above "
          f"that\n  adds cross-plane links only. 4500 km is the Iridium-class "
          f"Ka figure and sits\n  above the ~4090 km in-plane spacing; "
          f"anything larger needs its own argument.")
    print(f"\nRECOMMENDED: n_sats={pick['n_sats']} n_planes={pick['n_planes']} "
          f"isl_max_km={chosen_km:.0f}"
          + ("   (== current default)"
             if (pick['n_sats'], pick['n_planes'], chosen_km) == BASELINE
             else "   (differs from current default)"))

    # Why was a smaller constellation rejected? Range is the other lever, so
    # check whether ANY plausible range rescues it before blaming the count.
    small = [r for r in cand.values()
             if strip_ok(r) and r["n_sats"] < pick["n_sats"]]
    if small:
        print(f"\nRejected smaller constellations -- can extra ISL range "
              f"rescue them?")
        for r in sorted(small, key=order):
            k = (r["n_sats"], r["n_planes"])
            rescued = None
            for km in PROBE_KM:
                cfg = replace(DEFAULT, n_sats=k[0], n_planes=k[1], isl_max_km=km)
                g = gossip_metrics(C.build(cfg), cfg)
                if g["mean_reach"] >= REACH_FLOOR:
                    rescued = (km, g["mean_reach"])
                    break
            print(f"  n_sats={k[0]} n_planes={k[1]}: "
                  + (f"clears the gate only at {rescued[0]:.0f} km "
                     f"(reach {rescued[1]:.3f})" if rescued
                     else f"never clears reach >= {REACH_FLOOR} up to "
                          f"{PROBE_KM[-1]:.0f} km"))


if __name__ == "__main__":
    main()
