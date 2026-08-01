"""Contact plan generation. PLAN.md section 5.1.

TLEs + ground stations -> ContactPlan. Pure geometry: no faults, no routing,
no inference. Regenerate wholesale per run; it is sub-second.

Two things here are load-bearing and easy to get wrong:

1. Satellites are drawn from N ADJACENT orbital planes, not spread evenly.
   In-plane neighbours sit ~4090 km apart -- inside the 4500 km crosslink
   range -- so they stay connected almost continuously, and that chain is the
   path evidence walks during gossip. Sampling every Nth satellite across all
   six planes gives mean ISL degree 1.41 with 68.6% of pairs never connecting,
   which leaves gossip nowhere to go.

2. Skyfield's find_events can emit culminate-before-rise (#559) and returns
   wrong set times above 45 degrees (#1000). We pair rise->set with an explicit
   state machine and discard anything unpaired rather than assuming triplets.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
from skyfield.api import load, wgs84, EarthSatellite

from .config import ScenarioConfig, DEFAULT
from .types import Contact

R_EARTH_KM = 6378.137
ATM_MARGIN_KM = 80.0          # kills refraction/absorption near the limb
GRAZE_RADIUS_KM = R_EARTH_KM + ATM_MARGIN_KM
MIN_CONTACT_MS = 30_000       # shorter than this is not a usable window


# --------------------------------------------------------------------------
# Loading and plane selection
# --------------------------------------------------------------------------

def load_satellites(cfg: ScenarioConfig, ts) -> list[EarthSatellite]:
    """Load the TLE set and select n_sats from n_planes adjacent planes."""
    path = cfg.data_path(cfg.tle_file)
    sats = load.tle_file(str(path), reload=False)
    if not sats:
        raise RuntimeError(f"no satellites parsed from {path}")

    # Iridium is a polar constellation: 6 planes spaced ~30 deg in RAAN over 180.
    # nodeo is RAAN in radians on the underlying sgp4 model.
    binned: dict[int, list[EarthSatellite]] = {}
    for s in sats:
        raan = math.degrees(s.model.nodeo) % 180.0
        binned.setdefault(int(raan // 30.0), []).append(s)

    planes = sorted(binned)
    if not planes:
        raise RuntimeError("no orbital planes identified")

    # Choose the window of n_planes adjacent bins holding the most satellites.
    k = min(cfg.n_planes, len(planes))
    best, best_count = planes[:k], -1
    for i in range(len(planes) - k + 1):
        window = planes[i:i + k]
        count = sum(len(binned[p]) for p in window)
        if count > best_count:
            best, best_count = window, count

    # Round-robin across the chosen planes so the selection stays balanced,
    # sorting within each plane by TRUE ORBITAL PHASE at the scenario epoch.
    #
    # Do NOT sort by model.mo. Mean anomaly is stated at each satellite's own
    # TLE epoch, and CelesTrak staggers those ~9 minutes apart across a
    # constellation -- so mo is nearly constant within a plane (254-269 deg
    # here) while true position spans the whole ring. Sorting by mo sorts by
    # TLE epoch, which is not orbital order at all. That silently produced
    # selections with 58.5 and 163 degree ring holes; a 58.5 degree gap is a
    # 6995 km chord whose closest approach to Earth's centre is 6245 km,
    # inside the 6458 km graze radius, so it is occluded at ANY isl_max_km.
    # That is why raising the crosslink range never rescued 8-satellite
    # configurations. See FINDINGS F-008.
    t_epoch = ts.utc(*_iso_parts(cfg.epoch_iso))
    pools = []
    for p in best:
        pool = sorted(binned[p], key=lambda s: _arg_of_latitude(s, t_epoch))
        pools.append(pool)

    chosen: list[EarthSatellite] = []
    i = 0
    while len(chosen) < cfg.n_sats and any(pools):
        pool = pools[i % len(pools)]
        if pool:
            chosen.append(pool.pop(0))
        i += 1
        if i > 10_000:
            break

    if len(chosen) < cfg.n_sats:
        raise RuntimeError(
            f"only {len(chosen)} satellites available in {k} planes, need {cfg.n_sats}"
        )
    return chosen[:cfg.n_sats]


def _arg_of_latitude(sat: EarthSatellite, t) -> float:
    """Argument of latitude at time t, in [0, 2pi) -- the satellite's true
    angular position around its orbit, measured from the ascending node.

    This is the ordering that makes 'adjacent in the list' mean 'adjacent in
    the ring', which is what the in-plane crosslink chain depends on.
    """
    g = sat.at(t)
    r = np.asarray(g.position.km, dtype=float)
    v = np.asarray(g.velocity.km_per_s, dtype=float)
    h = np.cross(r, v)                      # orbital angular momentum
    n = np.cross(np.array([0.0, 0.0, 1.0]), h)   # line of nodes
    nn = np.linalg.norm(n)
    if nn < 1e-9:                           # equatorial orbit: node undefined
        return float(math.atan2(r[1], r[0]) % (2 * math.pi))
    n /= nn
    rn = r / np.linalg.norm(r)
    u = math.acos(float(np.clip(np.dot(n, rn), -1.0, 1.0)))
    if r[2] < 0:                            # southern half of the orbit
        u = 2 * math.pi - u
    return u


def load_stations(cfg: ScenarioConfig) -> list[dict]:
    blob = json.loads(cfg.data_path(cfg.stations_file).read_text(encoding="utf-8"))
    return blob["stations"][:cfg.n_stations]


def sat_id(index: int) -> str:
    return f"SAT{index + 1:02d}"


# --------------------------------------------------------------------------
# Downlinks
# --------------------------------------------------------------------------

def _downlinks(cfg, ts, sats, stations, t0, t1) -> list[Contact]:
    out: list[Contact] = []
    for si, sat in enumerate(sats):
        sname = sat_id(si)
        for st in stations:
            place = wgs84.latlon(st["lat"], st["lon"], elevation_m=st["alt_m"])
            times, events = sat.find_events(
                place, t0, t1, altitude_degrees=cfg.elev_mask_deg
            )

            # State machine: hold the most recent rise, close on the next set.
            # Culminations update the peak elevation. Anything unpaired is
            # dropped -- do NOT assume rise/culminate/set triplets.
            pending_rise = None
            peak = 0.0
            for t, ev in zip(times, events):
                if ev == 0:                       # rise
                    pending_rise, peak = t, 0.0
                elif ev == 1:                     # culminate
                    if pending_rise is not None:
                        alt, _, _ = (sat - place).at(t).altaz()
                        peak = max(peak, float(alt.degrees))
                elif ev == 2:                     # set
                    if pending_rise is None:
                        continue
                    a = _to_ms(pending_rise)
                    b = _to_ms(t)
                    pending_rise = None
                    if b - a < MIN_CONTACT_MS:
                        continue
                    if peak <= 0.0:
                        peak = cfg.elev_mask_deg
                    rate = _elev_scaled(cfg.rate_downlink, peak)
                    out.append(_mk(sname, st["id"], a, b, rate, "downlink", peak))
                    out.append(_mk(st["id"], sname, a, b, rate, "downlink", peak))
    return out


def _elev_scaled(base: int, elev_deg: float) -> int:
    """Low passes are genuinely worth less. Also gives weather something to
    bite: attenuation scales as 1/sin(elevation), so the edges of a pass
    degrade before the middle does."""
    f = min(1.0, max(0.3, math.sin(math.radians(max(elev_deg, 1.0)))))
    return int(base * f)


# --------------------------------------------------------------------------
# Inter-satellite links
# --------------------------------------------------------------------------

def _isls(cfg, ts, sats, epoch_ms) -> list[Contact]:
    n = len(sats)
    steps = cfg.horizon_ms // (cfg.step_s * 1000) + 1
    secs = np.arange(steps, dtype=float) * cfg.step_s

    base = ts.utc(*_iso_parts(cfg.epoch_iso))
    times = ts.tt_jd(base.tt + secs / 86400.0)

    # (nsat, 3, ntime) -- one propagation call per satellite, never per epoch.
    pos = np.stack([s.at(times).position.km for s in sats])

    out: list[Contact] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = pos[i], pos[j]
            d = b - a
            L2 = (d * d).sum(0)
            L = np.sqrt(L2)
            # Closest approach of the segment to Earth's centre. The minimiser
            # of |a + u*d| is u = -(a.d)/|d|^2; NEGATE FIRST, then clamp to
            # [0,1] to keep the test on the segment rather than the infinite
            # line. Clipping before negating puts u in [-1,0], and since
            # a.d < 0 for two bodies at similar radii the clip always returns
            # 0 -- miss collapses to |a| (the satellite's own orbital radius,
            # always above the graze radius) and the occlusion test silently
            # never rejects anything. See FINDINGS F-005.
            tp = np.clip(-(a * d).sum(0) / np.maximum(L2, 1e-9), 0.0, 1.0)
            miss = np.linalg.norm(a + tp * d, axis=0)
            vis = (miss > GRAZE_RADIUS_KM) & (L < cfg.isl_max_km)

            for s_idx, e_idx in _runs(vis):
                t_open = epoch_ms + int(secs[s_idx] * 1000)
                t_close = epoch_ms + int(secs[e_idx] * 1000)
                if t_close - t_open < MIN_CONTACT_MS:
                    continue
                si, sj = sat_id(i), sat_id(j)
                out.append(_mk(si, sj, t_open, t_close, cfg.rate_isl, "isl", 0.0))
                out.append(_mk(sj, si, t_open, t_close, cfg.rate_isl, "isl", 0.0))
    return out


def _runs(mask: np.ndarray):
    """Yield (start_index, end_index) for each contiguous True run."""
    if not mask.any():
        return
    idx = np.flatnonzero(
        np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    )
    for s, e in zip(idx[0::2], idx[1::2]):
        yield int(s), int(e - 1)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def _mk(src, dst, t_open, t_close, rate, kind, elev) -> Contact:
    return Contact(
        id=f"{src}>{dst}@{t_open // 1000}",
        src=src, dst=dst, t_open=t_open, t_close=t_close,
        rate_bps=int(rate), kind=kind, max_elev=round(float(elev), 2),
    )


def _iso_parts(iso: str):
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second


def _to_ms(t) -> int:
    """Skyfield Time -> absolute Unix epoch milliseconds. See types.py."""
    return int(t.utc_datetime().timestamp() * 1000)


def _merge_overlaps(contacts: list[Contact]) -> list[Contact]:
    """Contacts on one directed pair must not overlap -- overlapping windows
    double-count available volume. Merge rather than dropping."""
    by_pair: dict[tuple[str, str], list[Contact]] = {}
    for c in contacts:
        by_pair.setdefault((c.src, c.dst), []).append(c)

    out: list[Contact] = []
    for pair, group in by_pair.items():
        group.sort(key=lambda c: c.t_open)
        cur = group[0]
        for nxt in group[1:]:
            if nxt.t_open <= cur.t_close:
                cur = _mk(cur.src, cur.dst, cur.t_open,
                          max(cur.t_close, nxt.t_close),
                          min(cur.rate_bps, nxt.rate_bps), cur.kind,
                          max(cur.max_elev, nxt.max_elev))
            else:
                out.append(cur)
                cur = nxt
        out.append(cur)
    out.sort(key=lambda c: (c.t_open, c.id))
    return out


def build(cfg: ScenarioConfig = DEFAULT) -> list[Contact]:
    ts = load.timescale(builtin=True)
    sats = load_satellites(cfg, ts)
    stations = load_stations(cfg)

    t0 = ts.utc(*_iso_parts(cfg.epoch_iso))
    epoch_ms = int(t0.utc_datetime().timestamp() * 1000)
    t1 = ts.tt_jd(t0.tt + cfg.horizon_hours / 24.0)

    contacts = _downlinks(cfg, ts, sats, stations, t0, t1)
    contacts += _isls(cfg, ts, sats, epoch_ms)
    return _merge_overlaps(contacts)


def write(contacts: list[Contact], cfg: ScenarioConfig = DEFAULT) -> str:
    path = cfg.out_path("contact_plan.jsonl")
    with path.open("w", encoding="utf-8") as f:
        for c in contacts:
            f.write(json.dumps(c.to_dict(), separators=(",", ":")) + "\n")
    return str(path)


def read(cfg: ScenarioConfig = DEFAULT) -> list[Contact]:
    path = cfg.out_path("contact_plan.jsonl")
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Contact(**json.loads(line)))
    return out


if __name__ == "__main__":
    import time
    t = time.time()
    cs = build()
    p = write(cs)
    dl = [c for c in cs if c.kind == "downlink"]
    isl = [c for c in cs if c.kind == "isl"]
    print(f"{len(cs)} contacts ({len(dl)} downlink, {len(isl)} isl) "
          f"in {time.time()-t:.2f}s -> {p}")
