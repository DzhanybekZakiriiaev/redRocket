"""Interface contracts. PLAN.md section 4.

These six types are the only things that cross module boundaries. Freeze them
here; every other module imports from this file and nothing else defines a
cross-module shape.

All times are **absolute Unix epoch milliseconds, UTC** -- not offsets from the
scenario epoch. This is deliberate: the frontend can pass them straight to
`new Date(t)` with no separate epoch to ship, load, or get wrong. Wall-clock
labels never drift out of sync with the data.

Never float. Accumulated float error in time arithmetic is a determinism bug
that surfaces hours later as a non-reproducible run.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------
# Causes. NOMINAL is a member: without it the layer diagnoses a fault every tick.
# --------------------------------------------------------------------------

class Cause(str, Enum):
    NOMINAL = "nominal"
    WEATHER = "weather"          # attenuation at a ground station
    NODE_DOWN = "node_down"      # spacecraft failure, total
    POINTING = "pointing"        # antenna pointing error, primary channel only
    BUFFER = "buffer"            # onboard storage saturation
    STALE_SCHED = "stale_sched"  # contact opens at the wrong time


CAUSES: tuple[Cause, ...] = tuple(Cause)
N_CAUSES = len(CAUSES)
CAUSE_INDEX: dict[Cause, int] = {c: i for i, c in enumerate(CAUSES)}


class Action(str, Enum):
    NONE = "none"
    WAIT = "wait"
    REROUTE = "reroute"
    THROTTLE = "throttle"
    BLACKLIST = "blacklist"


class Outcome(str, Enum):
    OK = "ok"
    SILENT = "silent"        # nothing heard in the scheduled window
    DEGRADED = "degraded"    # opened, but below expected rate
    LATE = "late"            # carrier detected outside the scheduled window


class Channel(str, Enum):
    PRIMARY = "primary"   # high-gain payload link
    BEACON = "beacon"     # low-gain TT&C -- survives pointing error


# --------------------------------------------------------------------------
# 4.1 Contact -- output of contacts.py
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Contact:
    id: str
    src: str
    dst: str
    t_open: int
    t_close: int
    rate_bps: int
    kind: str          # "isl" | "downlink"
    max_elev: float    # degrees; 0.0 for ISLs

    @property
    def duration_ms(self) -> int:
        return self.t_close - self.t_open

    @property
    def volume_bits(self) -> int:
        return self.rate_bps * self.duration_ms // 1000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# 4.2 FaultEvent -- output of faults.py. GROUND TRUTH.
# Consumed by engine.py and by scoring. NEVER by an advisor.
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FaultEvent:
    t_start: int
    t_end: int
    target: str          # node id or ground-station id
    kind: Cause
    severity: float      # 0..1, fraction of capacity lost
    shift_ms: int = 0    # nonzero only for STALE_SCHED

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


# --------------------------------------------------------------------------
# 4.3 Observation -- everything the diagnosis layer is ever allowed to know.
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Observation:
    t: int
    node: str            # the observing node
    peer: str
    contact_id: str
    outcome: Outcome
    measured_rate: int
    expected_rate: int
    elevation_deg: float
    channel: Channel
    queue_bytes: int
    ms_since_ok: int
    shift_observed: int  # ms offset if carrier detected outside the window
    peer_degree: int     # concurrent contacts the peer has right now
    self_degree: int     # concurrent contacts we have right now

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        d["channel"] = self.channel.value
        return d


# --------------------------------------------------------------------------
# 4.4 Policy -- what an advisor returns.
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Policy:
    belief: dict[str, float]   # cause value -> probability, sums to 1
    action: Action
    target: str
    until: int
    confidence: float
    rationale: str = ""

    def argmax_cause(self) -> Cause:
        return Cause(max(self.belief, key=self.belief.__getitem__))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d


# --------------------------------------------------------------------------
# 4.5 Evidence -- the only thing that crosses a link between nodes.
#
# Raw observations, not posteriors. Forwarding beliefs double-counts: A's
# opinion reaches C directly and via B, C treats it as two independent
# confirmations, and the network converges confidently wrong. Shipping
# observations keyed by (origin, seq) sidesteps the problem instead of
# correcting for it.
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Evidence:
    origin: str    # who observed it -- NOT who forwarded it
    seq: int       # per-origin counter; (origin, seq) is the dedup key
    t: int
    peer: str
    outcome: Outcome
    degree: int

    WIRE_BYTES = 24  # packed size; count these in any bandwidth figure

    @property
    def key(self) -> tuple[str, int]:
        return (self.origin, self.seq)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


# --------------------------------------------------------------------------
# 4.7 Strip -- output of diagnosability.py, computed from the contact plan
# alone. No simulator, no faults, no LLM.
# --------------------------------------------------------------------------

@dataclass
class Strip:
    t_grid: list[int]
    per_link: dict[str, list[bool]] = field(default_factory=dict)
    per_pair: dict[str, list[bool]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_grid": self.t_grid,
            "per_link": {k: _pack(v) for k, v in self.per_link.items()},
            "per_pair": {k: _pack(v) for k, v in self.per_pair.items()},
        }


def _pack(bits: list[bool]) -> str:
    """Run-length encode a bool list as e.g. '12T,3F,40T'. The strip is mostly
    long runs, so this is ~50x smaller than a JSON bool array and the frontend
    unpacks it in three lines."""
    if not bits:
        return ""
    out, run, cur = [], 0, bits[0]
    for b in bits:
        if b == cur:
            run += 1
        else:
            out.append(f"{run}{'T' if cur else 'F'}")
            run, cur = 1, b
    out.append(f"{run}{'T' if cur else 'F'}")
    return ",".join(out)


def unpack(s: str) -> list[bool]:
    """Inverse of _pack. Kept here so the format has exactly one definition."""
    if not s:
        return []
    out: list[bool] = []
    for tok in s.split(","):
        out.extend([tok[-1] == "T"] * int(tok[:-1]))
    return out
