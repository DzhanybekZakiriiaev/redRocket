"""ScenarioConfig -- frozen, hashable, the single source of truth for a run.

Every artifact (contact plan, fault trace, traffic, strip) is keyed by this
config's hash, so a run is reproducible from (config, seed) alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"

# RNG stream ids. Named streams so adding one later does not perturb the others.
STREAM_TRAFFIC = 1
STREAM_FAULTS = 2
STREAM_CHANNEL = 3
STREAM_ADVISOR = 4
STREAM_GOSSIP = 5


@dataclass(frozen=True)
class ScenarioConfig:
    # --- constellation -----------------------------------------------------
    # TOPOLOGY IS LOCKED. Chosen by sweep over {8,12,16} sats x {1,2,3} planes
    # x {3000,4500,6000} km, scored on strip structure AND time-respecting
    # gossip reachability. See FINDINGS F-006/F-008 and PLAN section 5.1.
    tle_file: str = "iridium.tle"
    n_sats: int = 8            # Best on every measured axis post-F-008: gossip
                               # reach 1.000, downlink identifiability 0.6247
                               # (closest to mid-band of 8/12/16), 92.7% of
                               # links mixed (highest). Also the least cluttered
                               # on a globe, which UI-SPEC asks for.
    n_planes: int = 2          # Adjacent planes, not spread across all 6.
                               # In-plane neighbours sit ~4090 km apart, inside
                               # the crosslink range, so they form a continuous
                               # chain for evidence to walk. n_planes=1 has no
                               # cross-plane links at all -- a static ring, and
                               # the intermittency this project is about
                               # disappears.
    isl_max_km: float = 4500.0 # Smallest value clearing the gossip gate, and it
                               # clears it completely. 6000 km is no better on
                               # any gossip metric -- same reach, same latency --
                               # it only adds churn and globe clutter. Also the
                               # physically defensible Iridium-class Ka figure.
                               # Note in-plane crosslinks are capped near
                               # 6174 km by Earth occlusion regardless.

    # --- ground segment ----------------------------------------------------
    stations_file: str = "ground_stations.json"
    n_stations: int = 5        # OPERATIONAL stations only. The file holds 12 so
                               # the globe stays visually dense; the frontend
                               # renders the other 7 dimmed as other networks.
                               #
                               # 12 operational stations destroyed the premise:
                               # per-satellite coverage 47%, median gap 17 min.
                               # That is not a delay-tolerant network -- there
                               # is almost always a contact open, so "you cannot
                               # ask anyone" stops being true. At 5 the numbers
                               # are honest: 21% coverage, median gap 31.7 min,
                               # p90 80 min, max 92 min. It also gives the best
                               # strip of any option -- 100% of downlinks mixed
                               # and identifiability 0.521, dead centre of band.
                               # FINDINGS F-017.
    elev_mask_deg: float = 10.0

    # --- time --------------------------------------------------------------
    epoch_iso: str = "2026-08-01T00:00:00Z"
    horizon_hours: int = 24
    step_s: int = 10           # contact-window search granularity
    grid_s: int = 60           # diagnosability + belief emission grid

    # --- link rates (bits/s) ----------------------------------------------
    rate_isl: int = 25_000_000
    rate_downlink: int = 160_000_000
    rate_beacon: int = 2_000_000

    # --- traffic -----------------------------------------------------------
    bundle_bytes: int = 1024
    bundle_ttl_ms: int = 4 * 3600 * 1000
    arrival_rate_hz: float = 0.05

    seed: int = 42
    tag: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def horizon_ms(self) -> int:
        return self.horizon_hours * 3600 * 1000

    @property
    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def out_path(self, name: str) -> Path:
        OUT.mkdir(exist_ok=True)
        return OUT / name

    def data_path(self, name: str) -> Path:
        return DATA / name


DEFAULT = ScenarioConfig()
