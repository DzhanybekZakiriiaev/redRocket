"""Discrete-event simulator. PLAN.md sections 5.6 and 8.

Hand-rolled heapq event loop rather than SimPy: the value SimPy adds is
resource contention, which this model does not have (a contact is
rate x duration, consumed in closed form), and in exchange it costs control
over tie-break ordering and makes state hard to snapshot.

Integer milliseconds throughout. Never float -- accumulated float error in time
arithmetic is a determinism bug that surfaces hours later as a run that will
not reproduce.

Routing is first-contact: send on the next available contact toward the sink.
SABR is specified in the plan but deliberately off the build path; the project's
claim is about diagnosis quality, not routing quality, and a correct two-pass
SABR implementation is a multi-hour item that would buy nothing here.
"""

from __future__ import annotations

import heapq
import json
from collections import defaultdict
from dataclasses import dataclass, field

from .config import ScenarioConfig, DEFAULT, STREAM_TRAFFIC
from .types import (
    Contact, FaultEvent, Evidence, Observation, Outcome, Channel, Cause, Policy,
)
from .policy import PolicyEnforcer
from . import observe as OBS

CONTACT_OPEN, DECIDE, BUNDLE_NEW = 0, 1, 2
GOSSIP_BUDGET = 24        # Evidence tuples per contact
SINK = "GROUND"           # bundles are delivered on reaching any ground station


@dataclass(order=True)
class _Ev:
    t: int
    prio: int
    seq: int
    kind: int = field(compare=False)
    data: object = field(compare=False, default=None)


@dataclass
class Bundle:
    id: int
    src: str
    created: int
    size: int
    hops: int = 0
    at: str = ""


class Engine:
    def __init__(self, contacts: list[Contact], faults: list[FaultEvent],
                 advisor, cfg: ScenarioConfig = DEFAULT):
        self.cfg = cfg
        self.contacts = sorted(contacts, key=lambda c: (c.t_open, c.id))
        self.faults = faults
        self.advisor = advisor
        self.enforcer = PolicyEnforcer()

        self.sats = sorted({n for c in contacts for n in (c.src, c.dst)
                            if not n.startswith("GS_")})
        self.q: list[Bundle] = []
        self.queues: dict[str, list[Bundle]] = defaultdict(list)
        self.seq = 0
        self.ev_seq = 0
        self.heap: list[_Ev] = []

        # Per-node evidence log. Sequence numbers are per-origin so that
        # (origin, seq) is a stable dedup key no matter how many hops a tuple
        # takes -- forwarding must never make one observation look like two.
        self.ev_counter: dict[str, int] = defaultdict(int)
        self.held: dict[str, list[Evidence]] = defaultdict(list)

        self.last_ok: dict[tuple[str, str], int] = {}
        self.events: list[dict] = []
        self.beliefs: list[dict] = []
        self.delivered = 0
        self.dropped = 0
        self.gossip_bytes = 0

    # -- helpers -----------------------------------------------------------

    def _push(self, t: int, prio: int, kind: int, data=None) -> None:
        self.ev_seq += 1
        heapq.heappush(self.heap, _Ev(t, prio, self.ev_seq, kind, data))

    def _log(self, t: int, type_: str, **kw) -> None:
        self.events.append({"t": t, "type": type_, **kw})

    def _truth(self, node: str, peer: str, t: int) -> str:
        """Ground truth cause for a LINK. NEVER exposed to an advisor.

        Must consider both endpoints. Checking only the observing node made
        WEATHER invisible to scoring entirely: weather targets a ground
        station, so a satellite's own truth was never 'weather' and the cause
        the whole demo turns on never appeared in a confusion matrix.
        FINDINGS F-019.
        """
        for f in self.faults:
            if f.t_start <= t <= f.t_end and f.target in (node, peer):
                return f.kind.value
        return Cause.NOMINAL.value

    # -- setup -------------------------------------------------------------

    def _seed_traffic(self) -> None:
        import numpy as np
        rng = np.random.default_rng(
            np.random.SeedSequence([self.cfg.seed, STREAM_TRAFFIC]))
        t0 = self.contacts[0].t_open
        horizon = self.cfg.horizon_ms
        n = max(1, int(self.cfg.arrival_rate_hz * horizon / 1000 / 60))
        for _ in range(n * len(self.sats)):
            src = self.sats[int(rng.integers(len(self.sats)))]
            t = t0 + int(rng.integers(0, max(1, horizon - self.cfg.bundle_ttl_ms)))
            self._push(t, BUNDLE_NEW, BUNDLE_NEW, src)

    # -- handlers ----------------------------------------------------------

    def _on_contact(self, c: Contact) -> None:
        t = c.t_open
        cons = self.enforcer.constraints(c.src)

        if cons.is_excluded(c.dst, t):
            self._log(t, "contact_skipped", id=c.id, reason="excluded")
            return

        # Degrees at this instant, scoped by link kind -- the same quantity the
        # diagnosability strip is built from (FINDINGS F-001).
        deg_self = sum(1 for d in self.contacts
                       if d.kind == c.kind and d.src == c.src
                       and d.t_open <= t <= d.t_close)
        deg_peer = sum(1 for d in self.contacts
                       if d.kind == c.kind and d.src == c.dst
                       and d.t_open <= t <= d.t_close)

        key = (c.src, c.dst)
        since = t - self.last_ok.get(key, c.t_open)
        qbytes = sum(b.size for b in self.queues[c.src])

        # BEACON FIRST. The beacon is what separates a mispointed antenna
        # (beacon fine) from a dead spacecraft (beacon gone), and the filter
        # reads it as context when it processes the primary. Feeding the
        # primary first meant every diagnosis used the PREVIOUS contact's
        # beacon reading -- often hours stale, or absent entirely on a first
        # contact. Measured effect: pointing was misdiagnosed as node_down 63
        # times out of 87. FINDINGS F-018.
        b = OBS.observe(c, self.faults, t, node=c.src, peer=c.dst,
                        queue_bytes=qbytes, ms_since_ok=since,
                        peer_degree=deg_peer, self_degree=deg_self,
                        channel=Channel.BEACON)
        self.advisor.observe(b)

        o = OBS.observe(c, self.faults, t, node=c.src, peer=c.dst,
                        queue_bytes=qbytes, ms_since_ok=since,
                        peer_degree=deg_peer, self_degree=deg_self)
        self.advisor.observe(o)

        self.ev_counter[c.src] += 1
        mine = Evidence(c.src, self.ev_counter[c.src], t, c.dst, o.outcome,
                        deg_peer)
        self.held[c.src].append(mine)

        if o.outcome is Outcome.OK:
            self.last_ok[key] = t
            self._deliver(c, t)
            self._gossip(c, t)
        else:
            self._log(t, "contact_fail", id=c.id, src=c.src, dst=c.dst,
                      mode=o.outcome.value)

    def _gossip(self, c: Contact, t: int) -> None:
        """Exchange raw observations, not posteriors.

        Forwarding beliefs double-counts: A's opinion reaches C directly and via
        B, C treats it as two independent confirmations, and the network
        converges confidently wrong. Shipping observations keyed by
        (origin, seq) sidesteps the problem rather than correcting for it.
        """
        if c.dst.startswith("GS_"):
            return
        out = self.held[c.src][-GOSSIP_BUDGET:]
        if not out:
            return
        self.advisor.receive(out)
        seen = {e.key for e in self.held[c.dst]}
        self.held[c.dst].extend(e for e in out if e.key not in seen)
        self.gossip_bytes += len(out) * Evidence.WIRE_BYTES
        self._log(t, "gossip", **{"from": c.src, "to": c.dst, "n": len(out)})

    def _deliver(self, c: Contact, t: int) -> None:
        cons = self.enforcer.constraints(c.src)
        if cons.is_holding(c.dst, t):
            return
        cap = int(c.rate_bps * (c.t_close - c.t_open) / 8000)
        if cons.is_throttled(c.dst, t):
            cap = int(cap * PolicyEnforcer.THROTTLE_FACTOR)

        moved = 0
        queue = self.queues[c.src]
        while queue and moved < cap:
            bnd = queue[0]
            if t - bnd.created > self.cfg.bundle_ttl_ms:
                queue.pop(0)
                self.dropped += 1
                continue
            queue.pop(0)
            moved += bnd.size
            bnd.hops += 1
            if c.dst.startswith("GS_"):
                self.delivered += 1
                self._log(t, "deliver", bid=bnd.id, node=c.dst,
                          latency_ms=t - bnd.created, hops=bnd.hops)
            else:
                bnd.at = c.dst
                self.queues[c.dst].append(bnd)

    def _on_decide(self, t: int) -> None:
        for node in self.sats:
            p: Policy = self.advisor.decide(node, t)
            self.enforcer.apply(node, p, t)
            top = max(p.belief, key=p.belief.__getitem__)
            self.beliefs.append({
                "t": t, "node": node,
                "belief": {k: round(v, 4) for k, v in p.belief.items()},
                "action": p.action.value, "target": p.target,
                "argmax": top, "truth": self._truth(node, p.target, t),
            })

    # -- run ---------------------------------------------------------------

    def run(self) -> dict:
        for c in self.contacts:
            self._push(c.t_open, CONTACT_OPEN, CONTACT_OPEN, c)
        self._seed_traffic()

        t0 = self.contacts[0].t_open
        for k in range(0, self.cfg.horizon_ms, self.cfg.grid_s * 1000 * 5):
            self._push(t0 + k, DECIDE, DECIDE)

        while self.heap:
            ev = heapq.heappop(self.heap)
            if ev.kind == CONTACT_OPEN:
                self._on_contact(ev.data)
            elif ev.kind == DECIDE:
                self._on_decide(ev.t)
            elif ev.kind == BUNDLE_NEW:
                self.seq += 1
                self.queues[ev.data].append(
                    Bundle(self.seq, ev.data, ev.t, self.cfg.bundle_bytes,
                           at=ev.data))

        undelivered = sum(len(q) for q in self.queues.values())
        return {
            "advisor": getattr(self.advisor, "name", "?"),
            "delivered": self.delivered,
            "dropped": self.dropped,
            "undelivered": undelivered,
            "gossip_bytes": self.gossip_bytes,
            "policies_applied": self.enforcer.applied,
            "belief_samples": len(self.beliefs),
        }

    def write_trace(self, name: str = "trace.json") -> str:
        path = self.cfg.out_path(name)
        path.write_text(json.dumps(
            {"events": self.events, "beliefs": self.beliefs},
            separators=(",", ":")), encoding="utf-8")
        return str(path)
