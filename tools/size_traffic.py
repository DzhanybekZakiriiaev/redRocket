"""Traffic sizing sweep. PLAN.md section 9: "size buffers and traffic so the
fault-free case drops ~0 and the faulted case drops >0".

Nothing under sim/ is modified. The candidate configurations are applied
through `dataclasses.replace(DEFAULT, ...)`; the proposed per-node buffer cap
-- which does not exist in the engine yet -- is emulated here by swapping
`Engine.queues` for a defaultdict of size-tracking lists that tail-drop on
overflow. That reproduces exactly what an in-engine cap would do at the two
enqueue points (`run()`'s BUNDLE_NEW branch and `_deliver()`'s ISL relay
branch) without touching the module.

Usage:
    python tools/size_traffic.py                 # full candidate sweep
    python tools/size_traffic.py --seeds 5       # + multi-seed check on the pick
    python tools/size_traffic.py --only C2       # one candidate
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import contacts as C, faults as F                      # noqa: E402
from sim.config import DEFAULT, ScenarioConfig                  # noqa: E402
from sim.contacts import _elev_scaled                           # noqa: E402
from sim.engine import Engine                                   # noqa: E402
from sim.advisor.bayes import BayesAdvisor, NullAdvisor         # noqa: E402
from sim.types import Contact                                   # noqa: E402

HOUR = 3600 * 1000
MB = 1024 * 1024
GB = 1024 * MB


# --------------------------------------------------------------------------
# rate rescaling -- avoids re-running SGP4 for every candidate
# --------------------------------------------------------------------------

def retimed(contacts: list[Contact], cfg: ScenarioConfig) -> list[Contact]:
    """Same windows, rates recomputed for cfg's link rates.

    Contact windows depend only on geometry, so the (expensive) plan is built
    once and only `rate_bps` is re-derived. Downlinks go back through
    `_elev_scaled` so the elevation taper is applied to the new base rate
    rather than being approximated by a ratio.
    """
    if cfg.rate_downlink == DEFAULT.rate_downlink and cfg.rate_isl == DEFAULT.rate_isl:
        return contacts
    out = []
    for c in contacts:
        if c.kind == "downlink":
            out.append(replace(c, rate_bps=_elev_scaled(cfg.rate_downlink, c.max_elev)))
        else:
            out.append(replace(c, rate_bps=cfg.rate_isl))
    return out


# --------------------------------------------------------------------------
# per-node buffer cap, emulated
# --------------------------------------------------------------------------

def _capped_queues(cap: int | None, stats: dict):
    """defaultdict whose lists refuse bundles past `cap` bytes.

    The engine touches its queues in exactly four ways -- append, pop(0),
    len(), and a sum over sizes -- so overriding append/pop is enough to make
    a finite buffer behave correctly at BOTH enqueue points without copying a
    line of engine code.
    """

    class Q(list):
        def __init__(self):
            super().__init__()
            self.nbytes = 0

        def append(self, b):
            if cap is not None and self.nbytes + b.size > cap:
                stats["tail_dropped"] += 1
                return
            self.nbytes += b.size
            if self.nbytes > stats["peak_bytes"]:
                stats["peak_bytes"] = self.nbytes
                stats["peak_n"] = len(self) + 1
            list.append(self, b)

        def pop(self, i=-1):
            b = list.pop(self, i)
            self.nbytes -= b.size
            return b

    return defaultdict(Q)


def run_case(contacts, faults, cfg, advisor_name="bayes", buffer_bytes=None) -> dict:
    adv = (BayesAdvisor(grid_ms=cfg.grid_s * 1000) if advisor_name == "bayes"
           else NullAdvisor())
    eng = Engine(contacts, faults, adv, cfg)
    stats = {"tail_dropped": 0, "peak_bytes": 0, "peak_n": 0}
    eng.queues = _capped_queues(buffer_bytes, stats)
    t = time.time()
    r = eng.run()
    r["tail_dropped"] = stats["tail_dropped"]
    r["peak_queue_bytes"] = stats["peak_bytes"]
    r["peak_queue_n"] = stats["peak_n"]
    r["offered"] = r["delivered"] + r["dropped"] + r["undelivered"] + r["tail_dropped"]
    r["lost"] = r["dropped"] + r["tail_dropped"]
    r["secs"] = round(time.time() - t, 2)
    return r


# --------------------------------------------------------------------------
# plan capacity, for the utilisation figure
# --------------------------------------------------------------------------

def capacity(contacts, cfg) -> dict:
    dl = [c for c in contacts if c.kind == "downlink" and not c.src.startswith("GS_")]
    per_sat = defaultdict(int)
    for c in dl:
        per_sat[c.src] += c.rate_bps * (c.t_close - c.t_open) // 8000
    vals = sorted(per_sat.values())
    gaps = defaultdict(list)
    bysat = defaultdict(list)
    for c in dl:
        bysat[c.src].append(c)
    for s, v in bysat.items():
        v.sort(key=lambda c: c.t_open)
        gaps[s] = [(v[i + 1].t_open - v[i].t_close) for i in range(len(v) - 1)]
    allgaps = [g for v in gaps.values() for g in v]
    return {
        "per_sat_bytes_min": vals[0],
        "per_sat_bytes_med": st.median(vals),
        "n_sats": len(vals),
        "max_gap_ms": max(allgaps),
        "p90_gap_ms": sorted(allgaps)[int(0.9 * len(allgaps))],
    }


def offered_per_sat(cfg) -> int:
    """Mirror of Engine._seed_traffic's bundle count."""
    n = max(1, int(cfg.arrival_rate_hz * cfg.horizon_ms / 1000 / 60))
    return n * cfg.bundle_bytes


def bundles_per_sat(cfg) -> int:
    return max(1, int(cfg.arrival_rate_hz * cfg.horizon_ms / 1000 / 60))


def rate_for_bundles(n_per_sat: int) -> float:
    """Inverse of the above: arrival_rate_hz giving n bundles/sat/day."""
    return n_per_sat / (DEFAULT.horizon_ms / 1000 / 60)


# --------------------------------------------------------------------------
# candidates
#
# Design constraints that pin the family down:
#
#  * Contact capacity is ~230 GB per satellite per day (160 Mbps X-band, five
#    operational stations, ~37 passes, elevation-tapered). That figure is
#    already realistic: a Sentinel-2-class imager generates on the order of
#    1.5-2 TB/day and a smallsat EO payload 10-100 GB/day, so ~15 GB/orbit of
#    return capacity is the right order. The problem is not the link, it is
#    that the simulator offers 72 kB/sat/day against it.
#
#  * Bundle COUNT is the runtime budget (~12 us/bundle end to end). Real
#    volume at 1 kB/bundle would need ~1.5e8 objects. So the bundle has to
#    become an aggregation unit -- one object per image granule, not per
#    CCSDS frame. 8 MB keeps ~20k bundles/sat and still gives ~750 quanta per
#    pass, so packing stays effectively continuous.
#
#  * TTL is the sharpest knob. Fault-free per-satellite gaps are median 25 min,
#    p90 80 min, max 92 min. A 4 h TTL cannot expire anything the geometry
#    produces; a 2 h TTL clears the worst fault-free gap by 28 min and expires
#    on any outage past ~2 h -- which is where pointing (0.5-6 h) and node_down
#    live. 2 h is also a real EO near-real-time latency requirement.
# --------------------------------------------------------------------------

def _c(**kw) -> ScenarioConfig:
    return replace(DEFAULT, **kw)


# THE RECOMMENDATION. Every number here is reachable from a physical statement:
#
#   bundle_bytes    8 MiB   one compressed image granule, not one CCSDS frame.
#                           A 1 kB bundle cannot be used: real EO volume at
#                           1 kB/object is ~1e8 objects/day and the engine costs
#                           ~12 us/bundle. 8 MiB still gives ~750 granules per
#                           pass, so contact packing stays effectively continuous.
#   arrival_rate    6.0     -> 8640 granules/sat/day = 67.5 GiB/sat/day
#                           = 4.6 GiB per 97-min orbit. Smallsat-EO class.
#   bundle_ttl_ms   2 h     near-real-time product latency. Clears the worst
#                           fault-free pass gap (92 min) by 28 min; expires on
#                           any outage past ~2 h, which is where pointing
#                           (0.5-6 h) and node_down live.
#   buffer_bytes    11 GiB  mass memory ~ 3.3 h of generation. Sits above the
#                           fault-free peak queue (8.06 GiB) and below the
#                           faulted peak (16.8 GiB), so it never bites unless
#                           something is wrong.
REC = dict(bundle_bytes=8 * MB, arrival_rate_hz=6.0, bundle_ttl_ms=2 * HOUR)
REC_BUFFER = 11 * GB


def candidates() -> list[tuple[str, ScenarioConfig, int | None, str]]:
    """(label, cfg, buffer_bytes_or_None, note)"""
    B = 8 * MB
    out = [
        ("A0", DEFAULT, None,
         "status quo: 1 kB bundles, 0.05/min, TTL 4 h, 160 Mbps"),
        ("A1", _c(bundle_ttl_ms=2 * HOUR), None,
         "A0 + TTL 2 h  (TTL alone, traffic untouched)"),
        ("A2", _c(arrival_rate_hz=20.0, bundle_ttl_ms=2 * HOUR), None,
         "1 kB bundles pushed to 28.8k/sat -- still only 28 MB/sat/day (U=0.0001)"),
        # --- volume family: 8 MiB granules, TTL 2 h, utilisation swept -------
        ("B1", _c(bundle_bytes=B, arrival_rate_hz=6.0,
                  bundle_ttl_ms=2 * HOUR), None, "8 MiB granules, 8.6k/sat, U 0.31"),
        ("B2", _c(bundle_bytes=B, arrival_rate_hz=8.0,
                  bundle_ttl_ms=2 * HOUR), None, "8 MiB granules, 11.5k/sat, U 0.42"),
        ("B3", _c(bundle_bytes=B, arrival_rate_hz=12.0,
                  bundle_ttl_ms=2 * HOUR), None, "8 MiB granules, 17.3k/sat, U 0.63"),
        ("B4", _c(bundle_bytes=B, arrival_rate_hz=20.0,
                  bundle_ttl_ms=2 * HOUR), None, "8 MiB granules, 28.8k/sat, U 1.05"),
        # --- the recommendation, and the buffer cap bracketed -----------------
        ("R", _c(**REC), REC_BUFFER, "*** RECOMMENDED *** B1 + 11 GiB mass memory"),
        ("R-", _c(**REC), 9 * GB, "R with a 9 GiB buffer -- too tight, bites fault-free"),
        ("R+", _c(**REC), 16 * GB, "R with a 16 GiB buffer -- too loose, never bites"),
        # --- alternative: keep 1 MiB bundles, drop the link rate ---------------
        ("E", _c(bundle_bytes=MB, arrival_rate_hz=6.0, bundle_ttl_ms=2 * HOUR,
                 rate_downlink=20_000_000), 11 * GB // 8,
         "1 MiB bundles + 20 Mbps smallsat X-band: same U, 8x finer granularity"),
    ]
    return out


# --------------------------------------------------------------------------

HDR = (f"{'id':<4}{'bnd/sat':>9}{'GiB/sat':>9}{'U':>5}{'Uwrst':>6}"
       f"{'|':>2} {'deliv':>7}{'ttl':>7}{'buf':>7}{'und':>6}{'peakQ':>7}"
       f"{'|':>2} {'deliv':>7}{'ttl':>7}{'buf':>7}{'und':>6}{'peakQ':>7}"
       f"{'|':>2}{'s':>5}")


def fmt_row(label, cfg, cap, nf, ff) -> str:
    off = offered_per_sat(cfg)
    U = off / cap["per_sat_bytes_med"]
    Uw = off / cap["per_sat_bytes_min"]
    return (f"{label:<4}{bundles_per_sat(cfg):>9,}{off/GB:>9.1f}{U:>5.2f}{Uw:>6.2f}"
            f"{'|':>2} {ff['delivered']:>7,}{ff['dropped']:>7,}"
            f"{ff['tail_dropped']:>7,}{ff['undelivered']:>6,}"
            f"{ff['peak_queue_bytes']/GB:>7.1f}"
            f"{'|':>2} {nf['delivered']:>7,}{nf['dropped']:>7,}"
            f"{nf['tail_dropped']:>7,}{nf['undelivered']:>6,}"
            f"{nf['peak_queue_bytes']/GB:>7.1f}"
            f"{'|':>2}{nf['secs']+ff['secs']:>5.1f}")


def policy_cost(contacts, faults, cfg, buffer_bytes) -> dict:
    """How much delivery capacity the advisor's own policies destroy.

    Every action the enforcer can take is a RESTRICTION -- exclude, hold,
    throttle. None of them opens an alternative, because the router is
    first-contact and has nothing to switch to. So a correct diagnosis cannot
    move a bundle it would not otherwise have moved; it can only decline
    contacts. This counts how many of the declined contacts were healthy.
    """
    from sim import observe as OBS
    from sim.types import Outcome

    eng = Engine(contacts, faults, BayesAdvisor(grid_ms=cfg.grid_s * 1000), cfg)
    stats = {"tail_dropped": 0, "peak_bytes": 0, "peak_n": 0}
    eng.queues = _capped_queues(buffer_bytes, stats)
    eng.run()

    by_id = {c.id: c for c in contacts}
    skipped = [e for e in eng.events if e["type"] == "contact_skipped"]
    healthy = wasted = 0
    for e in skipped:
        c = by_id.get(e["id"])
        if c is None:
            continue
        o = OBS.observe(c, faults, c.t_open, node=c.src, peer=c.dst,
                        queue_bytes=0, ms_since_ok=0, peer_degree=1, self_degree=1)
        if o.outcome is Outcome.OK:
            healthy += 1
            wasted += c.rate_bps * (c.t_close - c.t_open) // 8000
    return {"skipped": len(skipped), "skipped_healthy": healthy,
            "wasted_bytes": wasted,
            "wasted_bundles": wasted // cfg.bundle_bytes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run one candidate by id")
    ap.add_argument("--seeds", type=int, default=0,
                    help="also run the recommendation over N fault seeds")
    ap.add_argument("--advisor", default="bayes", choices=("bayes", "null"))
    ap.add_argument("--recommend", default="R")
    args = ap.parse_args()

    base = C.read()
    print(f"contact plan: {len(base)} contacts "
          f"({sum(1 for c in base if c.kind=='downlink')} downlink)")

    cap0 = capacity(base, DEFAULT)
    print(f"per-satellite downlink capacity: med {cap0['per_sat_bytes_med']/GB:.0f} GB/day, "
          f"min {cap0['per_sat_bytes_min']/GB:.0f} GB/day  "
          f"(fault-free pass gaps: p90 {cap0['p90_gap_ms']/60000:.0f} min, "
          f"max {cap0['max_gap_ms']/60000:.0f} min)")
    print("reference: a Sentinel-2-class optical EO satellite generates ~1.5-2 TB/day")
    print("           (~100-140 GB per 100-min orbit); a smallsat EO payload 10-100 GB/day.")
    print("           So ~230 GB/sat/day of return capacity is the right order -- the")
    print("           mismatch is entirely on the traffic side (0.07 MB/sat/day today).")
    print()
    print("           left block = NO FAULTS,  right block = standard 8-fault draw (seed 42)")
    print(HDR)
    print("-" * len(HDR))

    cases = candidates()
    if args.only:
        cases = [c for c in cases if c[0] == args.only]

    results = {}
    for label, cfg, buf, note in cases:
        cs = retimed(base, cfg)
        faults = F.generate(base, cfg, n_faults=8)
        ff = run_case(cs, [], cfg, args.advisor, buf)
        nf = run_case(cs, faults, cfg, args.advisor, buf)
        results[label] = (cfg, buf, note, ff, nf)
        print(fmt_row(label, cfg, capacity(cs, cfg), nf, ff))
    print("-" * len(HDR))
    for label, cfg, buf, note in cases:
        print(f"  {label}: {note}")

    # --- verdict per candidate -------------------------------------------
    print()
    print("PLAN 9 gate  (fault-free loss ~0  AND  faulted loss > 0):")
    for label, (cfg, buf, note, ff, nf) in results.items():
        ff_loss = ff["lost"] / max(ff["offered"], 1)
        nf_loss = nf["lost"] / max(nf["offered"], 1)
        ok = ff_loss <= 0.002 and nf["lost"] > 0
        print(f"  {label:<4} fault-free loss {100*ff_loss:6.3f}%   "
              f"faulted loss {100*nf_loss:6.3f}%   "
              f"separation {nf['lost'] - ff['lost']:>7,} bundles   "
              f"{'PASS' if ok else 'fail'}")

    # --- multi-seed stability on the recommendation ------------------------
    if args.seeds:
        label = args.recommend
        cfg, buf, note, _, _ = results[label]
        cs = retimed(base, cfg)
        print()
        print(f"recommendation {label} over {args.seeds} fault seeds "
              f"(null vs bayes, same faults):")
        print(f"  {'seed':>4} {'arm':<6}{'deliv':>9}{'ttl-drop':>10}"
              f"{'buf-drop':>10}{'undel':>8}{'loss%':>8}")
        agg = defaultdict(list)
        for s in range(args.seeds):
            scfg = replace(cfg, seed=s)
            fs = F.generate(base, scfg, n_faults=8)
            for arm in ("null", "bayes"):
                r = run_case(cs, fs, scfg, arm, buf)
                loss = 100 * r["lost"] / max(r["offered"], 1)
                agg[arm].append(loss)
                print(f"  {s:>4} {arm:<6}{r['delivered']:>9,}{r['dropped']:>10,}"
                      f"{r['tail_dropped']:>10,}{r['undelivered']:>8,}{loss:>8.2f}")
        print(f"  mean loss%  null {st.mean(agg['null']):.2f}   "
              f"bayes {st.mean(agg['bayes']):.2f}")
        print()
        print("  why bayes cannot win on this metric -- policy is restriction-only:")
        for s in range(min(args.seeds, 4)):
            scfg = replace(cfg, seed=s)
            fs = F.generate(base, scfg, n_faults=8)
            pc = policy_cost(cs, fs, scfg, buf)
            print(f"    seed {s}: {pc['skipped']:>4} contacts declined, "
                  f"{pc['skipped_healthy']:>4} of them healthy, "
                  f"{pc['wasted_bundles']:>8,} granules of capacity thrown away")


if __name__ == "__main__":
    main()
