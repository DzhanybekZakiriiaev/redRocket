"""Fault injection. PLAN.md section 5.3.

GROUND TRUTH. The output of this module is consumed by the engine to decide
what actually happens, and by scoring to grade a diagnosis. It is never visible
to an advisor -- that would be handing over the answer key.

The premise the whole project rests on: all five causes produce the SAME
immediate observable, a contact that doesn't open. Separation comes from the
cross-sectional and temporal pattern, never from a single observation. If that
stops being true here, the project has no problem left to solve, so the
acceptance gate for this stage (PLAN.md F4) checks it directly.

Physics constants are from PLAN.md section 9.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np

from .config import ScenarioConfig, DEFAULT, STREAM_FAULTS
from .types import Contact, FaultEvent, Cause

# --- weather ---------------------------------------------------------------
# Fade duration is lognormal above ~1 min, median ~5 min, tail to ~30 min for
# deep events. Rain cells are 5-10 km and decorrelate by ~100 km, so a fault
# applies to ONE ground station and never to its neighbours.
WX_MEDIAN_S = 300.0
WX_SIGMA = 0.9
WX_MIN_S, WX_MAX_S = 60.0, 1800.0

# --- node failure ----------------------------------------------------------
# Weibull shape < 1 (infant mortality) per Castet & Saleh. Transient modes
# dominate by count; permanent failure is rarer and absorbing.
ND_TRANSIENT_P = 0.65
ND_WATCHDOG_S = (30.0, 300.0)        # seconds to minutes
ND_REBOOT_S = (3600.0, 86400.0)      # SEU-induced, ~1 day
ND_SAFEMODE_S = (14400.0, 259200.0)  # hours to days

# --- pointing --------------------------------------------------------------
# L_dB ~= 12*(theta/theta_3dB)^2; theta_3dB ~= 5.3 deg at X-band for a 0.5 m
# dish, so a 5 deg excursion is ~12 dB and the primary link is dead. The BEACON
# SURVIVES -- that asymmetry is the discriminator against node_down.
PE_THETA_3DB = 5.3
PE_MIN_S, PE_MAX_S = 1800.0, 21600.0

# --- buffer ----------------------------------------------------------------
BUF_MIN_S, BUF_MAX_S = 1800.0, 14400.0

# --- stale schedule --------------------------------------------------------
# Shift is the same sign and magnitude across every contact of the affected
# node, growing with plan age. Under a minute there is no observable at all
# against a 5-12 min window.
#
# The range deliberately STRADDLES the 300 s listening window (FINDINGS F-011).
# Drawing only 60-300 s put every shift inside the window, so the fault read
# LATE 100% of the time and was identifiable from a single observation in every
# scenario -- one of five causes contributing nothing to the problem and
# inflating any aggregate accuracy figure. Spanning 60-600 s means roughly half
# the draws exceed the window, read SILENT, and become genuinely confusable
# with the absence-producing causes.
SS_SHIFT_MIN_S, SS_SHIFT_MAX_S = 60.0, 600.0
SS_MIN_S, SS_MAX_S = 7200.0, 43200.0


def pointing_loss_db(theta_deg: float, theta_3db: float = PE_THETA_3DB) -> float:
    return 12.0 * (theta_deg / theta_3db) ** 2


def _rng(cfg: ScenarioConfig) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([cfg.seed, STREAM_FAULTS]))


def _nodes(contacts: list[Contact]) -> tuple[list[str], list[str]]:
    sats = sorted({n for c in contacts for n in (c.src, c.dst) if not n.startswith("GS_")})
    stations = sorted({n for c in contacts for n in (c.src, c.dst) if n.startswith("GS_")})
    return sats, stations


def _lognormal_s(rng, median_s: float, sigma: float, lo: float, hi: float) -> float:
    return float(np.clip(rng.lognormal(math.log(median_s), sigma), lo, hi))


def _span(cfg: ScenarioConfig) -> tuple[int, int]:
    from .diagnosability import _epoch_ms
    t0 = _epoch_ms(cfg)
    return t0, t0 + cfg.horizon_ms


# --------------------------------------------------------------------------
# Individual fault constructors. Each returns exactly one FaultEvent so the
# demo can script one and the random generator can draw many.
# --------------------------------------------------------------------------

def weather(rng, station: str, t_start: int) -> FaultEvent:
    dur = _lognormal_s(rng, WX_MEDIAN_S, WX_SIGMA, WX_MIN_S, WX_MAX_S)
    # Severity is fade depth normalised. Ka is ~8.6x X-band at the same rain
    # rate; we carry a single scalar and let the engine apply 1/sin(elevation),
    # which is what clips the low-elevation edges of a pass first.
    severity = float(np.clip(rng.beta(2.0, 2.0) * 1.2, 0.15, 1.0))
    return FaultEvent(t_start, t_start + int(dur * 1000), station,
                      Cause.WEATHER, round(severity, 3))


def node_down(rng, sat: str, t_start: int, horizon_end: int) -> FaultEvent:
    if rng.random() < ND_TRANSIENT_P:
        mode = rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
        lo, hi = (ND_WATCHDOG_S, ND_REBOOT_S, ND_SAFEMODE_S)[int(mode)]
        end = t_start + int(rng.uniform(lo, hi) * 1000)
    else:
        end = horizon_end  # permanent within the scenario: absorbing
    return FaultEvent(t_start, min(end, horizon_end), sat, Cause.NODE_DOWN, 1.0)


def pointing(rng, sat: str, t_start: int, horizon_end: int) -> FaultEvent:
    theta = float(rng.uniform(2.0, 7.0))
    loss = pointing_loss_db(theta)
    severity = float(np.clip(loss / 12.0, 0.0, 1.0))  # 12 dB == link dead
    end = t_start + int(rng.uniform(PE_MIN_S, PE_MAX_S) * 1000)
    return FaultEvent(t_start, min(end, horizon_end), sat,
                      Cause.POINTING, round(severity, 3))


def buffer(rng, sat: str, t_start: int, horizon_end: int) -> FaultEvent:
    severity = float(rng.uniform(0.3, 0.85))  # partial by construction, never total
    end = t_start + int(rng.uniform(BUF_MIN_S, BUF_MAX_S) * 1000)
    return FaultEvent(t_start, min(end, horizon_end), sat,
                      Cause.BUFFER, round(severity, 3))


def stale_sched(rng, sat: str, t_start: int, horizon_end: int) -> FaultEvent:
    shift = int(rng.uniform(SS_SHIFT_MIN_S, SS_SHIFT_MAX_S) * 1000)
    if rng.random() < 0.5:
        shift = -shift
    end = t_start + int(rng.uniform(SS_MIN_S, SS_MAX_S) * 1000)
    return FaultEvent(t_start, min(end, horizon_end), sat,
                      Cause.STALE_SCHED, 1.0, shift_ms=shift)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def generate(contacts: list[Contact], cfg: ScenarioConfig = DEFAULT,
             n_faults: int = 8,
             kinds: tuple[Cause, ...] | None = None) -> list[FaultEvent]:
    """Draw n_faults deterministically from (cfg.seed, STREAM_FAULTS)."""
    rng = _rng(cfg)
    sats, stations = _nodes(contacts)
    t0, t1 = _span(cfg)
    # Leave a settling margin at each end so no fault is truncated by the
    # horizon -- a clipped fault has a misleading duration signature.
    lo, hi = t0 + cfg.horizon_ms // 12, t1 - cfg.horizon_ms // 6

    pool = list(kinds) if kinds else [
        Cause.WEATHER, Cause.NODE_DOWN, Cause.POINTING,
        Cause.BUFFER, Cause.STALE_SCHED,
    ]

    out: list[FaultEvent] = []
    # At most ONE active fault per target at any instant. Two faults overlapping
    # on one node makes the ground-truth label ambiguous -- if SAT11 is both
    # stale-scheduled and dead, there is no single correct answer to score a
    # diagnosis against, and any confusion matrix built from it is meaningless.
    # Compound-fault diagnosis is a real problem and explicitly out of scope.
    occupied: dict[str, list[tuple[int, int]]] = {}

    def free(target: str, a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in occupied.get(target, ()))

    attempts = 0
    while len(out) < n_faults and attempts < n_faults * 60:
        attempts += 1
        kind = pool[int(rng.integers(len(pool)))]
        t_start = int(rng.integers(lo, hi))

        if kind is Cause.WEATHER:
            ev = weather(rng, stations[int(rng.integers(len(stations)))], t_start)
        else:
            sat = sats[int(rng.integers(len(sats)))]
            fn = {Cause.NODE_DOWN: node_down, Cause.POINTING: pointing,
                  Cause.BUFFER: buffer, Cause.STALE_SCHED: stale_sched}[kind]
            ev = fn(rng, sat, t_start, t1)

        if not free(ev.target, ev.t_start, ev.t_end):
            continue
        occupied.setdefault(ev.target, []).append((ev.t_start, ev.t_end))
        out.append(ev)

    if len(out) < n_faults:
        # Long absorbing faults (permanent node_down, stale_sched) saturate a
        # target's timeline, so the pool can genuinely run out. Say so rather
        # than silently returning fewer than asked for.
        print(f"[faults] warning: placed {len(out)}/{n_faults} without overlap "
              f"after {attempts} attempts")

    out.sort(key=lambda f: (f.t_start, f.target, f.kind.value))
    return out


def scripted_demo(contacts: list[Contact], cfg: ScenarioConfig = DEFAULT
                  ) -> list[FaultEvent]:
    """The demo scenario. PLAN.md section 6.6.

    Weather first, not node death. If a satellite dies, every peer observes the
    silence directly and diagnoses independently -- red spreads across the
    constellation and looks like distributed inference while being nothing of
    the kind. Weather is the case where a satellite genuinely cannot tell
    'station is faded' from 'my own antenna is mispointed' until it hears that
    a DIFFERENT satellite also failed at the SAME station.

    Chooses the station carrying the most contacts so the ambiguity has the
    best chance of resolving inside the scenario window.
    """
    rng = _rng(cfg)
    sats, stations = _nodes(contacts)
    t0, t1 = _span(cfg)

    busiest = max(stations, key=lambda s: sum(
        1 for c in contacts if s in (c.src, c.dst)))
    first = min((c.t_open for c in contacts
                 if busiest in (c.src, c.dst) and c.t_open > t0 + 3600_000),
                default=t0 + 3600_000)

    wx = FaultEvent(first - 60_000, first + 900_000, busiest, Cause.WEATHER, 0.85)
    victim = max(sats, key=lambda s: sum(
        1 for c in contacts if c.src == s and c.kind == "downlink"))
    nd = node_down(rng, victim, first + 5 * 3600_000, t1)
    return sorted([wx, nd], key=lambda f: f.t_start)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def write(faults: list[FaultEvent], cfg: ScenarioConfig = DEFAULT,
          name: str = "fault_trace.jsonl") -> str:
    path = cfg.out_path(name)
    with path.open("w", encoding="utf-8") as f:
        for e in faults:
            f.write(json.dumps(e.to_dict(), separators=(",", ":")) + "\n")
    return str(path)


def read(cfg: ScenarioConfig = DEFAULT,
         name: str = "fault_trace.jsonl") -> list[FaultEvent]:
    out = []
    with cfg.out_path(name).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                d["kind"] = Cause(d["kind"])
                out.append(FaultEvent(**d))
    return out


if __name__ == "__main__":
    from . import contacts as C

    cs = C.read()
    fs = generate(cs)
    p = write(fs)
    print(f"{len(fs)} faults -> {p}")
    for e in fs:
        mins = (e.t_end - e.t_start) / 60000
        extra = f" shift={e.shift_ms/1000:+.0f}s" if e.shift_ms else ""
        print(f"  {e.kind.value:12s} {e.target:12s} "
              f"{mins:7.1f} min  sev={e.severity:.2f}{extra}")

    d = write(scripted_demo(cs), name="fault_trace_demo.jsonl")
    print(f"demo scenario -> {d}")
