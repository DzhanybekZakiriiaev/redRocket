"""Discrete Bayes filter over fault causes. PLAN.md section 5.4.

This is the accuracy anchor. It is decision-theoretically optimal given its
belief and the shared cost matrix, it costs nothing to run, and it is what the
LLM has to justify itself against. Expect it to be hard to beat on the five
modelled causes -- that is the honest result, and knowing it early is worth
more than hoping otherwise.

Two ideas carry the model.

**Per-cause dynamics.** The transition kernel is action-conditioned and each
cause decays differently: weather self-heals on a geometric dwell, a dead node
is absorbing and no action escapes it, buffer saturation clears under throttle
and persists under wait. Encoding that here is why time-to-diagnosis differs by
cause, and why a memoryless classifier underperforms.

**Cross-link context is the whole discriminator.** A single observation cannot
separate weather-at-the-station from pointing-error-on-this-satellite -- the
plan says so and the F4 gate measured it. What separates them is the shape of
the failure across links:

    my other links also failing   -> the fault is mine (pointing, node_down)
    other nodes failing at this peer -> the fault is the peer's (weather)

A node sees the first directly. The second only arrives by gossip. That gap is
exactly the demo: local-only stays ambiguous, and evidence from a peer that
failed at the same station collapses it.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..types import (
    Observation, Evidence, Policy, Cause, Outcome, Channel,
    CAUSES, CAUSE_INDEX, N_CAUSES,
)
from .base import bayes_action, nominal_belief

# --------------------------------------------------------------------------
# Emission model
#
# The outcome rows are calibrated against the measured F4 distributions rather
# than invented -- see PROGRESS.md. Keeping them tied to measurement is what
# stops this becoming a model of our own assumptions.
# --------------------------------------------------------------------------

# WEATHER and POINTING deliberately SHARE a row. PLAN section 5.2 states that
# at degree 1 these two predict identical observation sets, so the posterior
# ratio must equal the prior ratio for all t and all observations. Distinct
# rows violate that directly: each observation multiplies the ratio by a
# constant != 1 and it compounds without bound -- measured at 12530x after
# twenty identical observations on an unidentifiable link, which is the
# accuracy anchor being confidently wrong exactly where the headline result
# says nothing is knowable.
#
# Physically this is also the honest row: attenuation from rain and attenuation
# from a mispointed dish look the same down one link. Their entire difference
# lives in the co-failure features, which is why it only resolves with a
# concurrent contact or with gossip. FINDINGS F-016.
_ATTENUATION = {Outcome.OK: 0.03, Outcome.DEGRADED: 0.45,
                Outcome.SILENT: 0.51, Outcome.LATE: 0.01}

P_OUTCOME: dict[Cause, dict[Outcome, float]] = {
    Cause.NOMINAL:     {Outcome.OK: 0.96, Outcome.DEGRADED: 0.02,
                        Outcome.SILENT: 0.015, Outcome.LATE: 0.005},
    Cause.WEATHER:     dict(_ATTENUATION),
    Cause.POINTING:    dict(_ATTENUATION),
    Cause.NODE_DOWN:   {Outcome.OK: 0.01, Outcome.DEGRADED: 0.02,
                        Outcome.SILENT: 0.96, Outcome.LATE: 0.01},
    Cause.BUFFER:      {Outcome.OK: 0.05, Outcome.DEGRADED: 0.90,
                        Outcome.SILENT: 0.04, Outcome.LATE: 0.01},
    Cause.STALE_SCHED: {Outcome.OK: 0.02, Outcome.DEGRADED: 0.02,
                        Outcome.SILENT: 0.50, Outcome.LATE: 0.46},
}

# P(this link has been silent a long time | cause). Weather is bounded -- a
# ~5 min median dwell, 30 min at the extreme -- while a dead node is absorbing.
# This is what legitimately separates the two over TIME, per the separability
# table's "concurrency, plus elapsed-duration prior". Without it the filter had
# to separate them by multiplying repeated observations of one ongoing
# condition as though they were independent samples, which they are not.
P_LONG_SILENCE: dict[Cause, float] = {
    Cause.NOMINAL: 0.02, Cause.WEATHER: 0.10, Cause.NODE_DOWN: 0.90,
    Cause.POINTING: 0.60, Cause.BUFFER: 0.25, Cause.STALE_SCHED: 0.70,
}
LONG_SILENCE_MS = 45 * 60 * 1000

# P(my other links are also failing | cause). High for faults that live on my
# side of the link, low for faults that live on the peer's side.
P_SELF_COFAIL: dict[Cause, float] = {
    Cause.NOMINAL: 0.05, Cause.WEATHER: 0.10, Cause.NODE_DOWN: 0.95,
    Cause.POINTING: 0.85, Cause.BUFFER: 0.70, Cause.STALE_SCHED: 0.90,
}

# P(other nodes are also failing with this peer | cause). THIS is the weather
# discriminator, and it is only observable through gossip.
#
# Read every row from the observing node's seat: I am a satellite, the peer is
# a ground station, and every cause except WEATHER lives on MY side of the
# link. If my own spacecraft is down, other satellites at that station are
# entirely unaffected -- so NODE_DOWN belongs down here with pointing and
# buffer, not up with weather.
#
# It was originally 0.90, on a muddled reading where "the node that is down"
# could be either endpoint. That made peer-cofailure unable to separate weather
# from node_down at all: gossip reporting two other satellites silent at the
# same station still concluded my spacecraft had died. The single most
# important discriminator in the model was inert. See FINDINGS F-012.
P_PEER_COFAIL: dict[Cause, float] = {
    Cause.NOMINAL: 0.05, Cause.WEATHER: 0.90, Cause.NODE_DOWN: 0.05,
    Cause.POINTING: 0.08, Cause.BUFFER: 0.10, Cause.STALE_SCHED: 0.10,
}

# P(the low-gain beacon is still nominal | cause). A mispointed high-gain dish
# leaves the beacon untouched; a dead spacecraft takes everything with it.
P_BEACON_OK: dict[Cause, float] = {
    Cause.NOMINAL: 0.98, Cause.WEATHER: 0.55, Cause.NODE_DOWN: 0.02,
    Cause.POINTING: 0.95, Cause.BUFFER: 0.95, Cause.STALE_SCHED: 0.05,
}

# P(my outbound queue is deep | cause). Buffer saturation is the only cause
# that is *defined* by a full queue; every other cause can occur with an empty
# one. Without this the filter has no way to rule buffer out, and because
# DEGRADED is buffer's most likely outcome (0.90, the highest in the table) it
# soaked up ~80% of the posterior on any sustained degradation regardless of
# whether the queue held a single byte. See FINDINGS F-013.
P_QUEUE_DEEP: dict[Cause, float] = {
    Cause.NOMINAL: 0.10, Cause.WEATHER: 0.30, Cause.NODE_DOWN: 0.25,
    Cause.POINTING: 0.30, Cause.BUFFER: 0.95, Cause.STALE_SCHED: 0.25,
}

# Bytes above which the outbound queue counts as deep. Faults that stall a link
# do back traffic up, so this is not a clean buffer-only signal -- hence the
# non-trivial probabilities above for the other causes.
QUEUE_DEEP_BYTES = 256 * 1024

# Per-step self-transition probability on a 60 s grid.
P_STAY: dict[Cause, float] = {
    Cause.NOMINAL: 0.995,
    Cause.WEATHER: 0.85,       # self-healing, geometric dwell ~5 min median
    Cause.NODE_DOWN: 0.999,    # absorbing; no action escapes it
    Cause.POINTING: 0.97,      # semi-absorbing
    Cause.BUFFER: 0.93,
    Cause.STALE_SCHED: 0.995,  # absorbing until a schedule refresh
}

COFAIL_THRESHOLD = 0.34   # "several of my links" rather than "one flaked"
EVIDENCE_HALFLIFE_MS = 45 * 60 * 1000


def _transition() -> np.ndarray:
    """T[i, j] = P(next = j | current = i). Leaving a fault state returns to
    NOMINAL; there is no direct fault-to-fault transition, which matches the
    one-active-fault-per-target invariant the generator enforces."""
    T = np.zeros((N_CAUSES, N_CAUSES))
    nom = CAUSE_INDEX[Cause.NOMINAL]
    for c in CAUSES:
        i = CAUSE_INDEX[c]
        stay = P_STAY[c]
        T[i, i] = stay
        if c is Cause.NOMINAL:
            leak = (1.0 - stay) / (N_CAUSES - 1)
            for d in CAUSES:
                if d is not c:
                    T[i, CAUSE_INDEX[d]] = leak
        else:
            T[i, nom] = 1.0 - stay
    return T


T_MATRIX = _transition()


def _bern(p: float, observed: bool) -> float:
    return p if observed else (1.0 - p)


class BayesAdvisor:
    """One filter per (node, peer) link."""

    name = "bayes"

    def __init__(self, grid_ms: int = 60_000):
        self.grid_ms = grid_ms
        self._belief: dict[tuple[str, str], np.ndarray] = {}
        self._last_t: dict[tuple[str, str], int] = {}
        # Most recent local outcome per link, for the self-cofailure feature.
        self._recent: dict[str, dict[str, tuple[int, Outcome]]] = defaultdict(dict)
        # Gossiped evidence, keyed by peer, for the peer-cofailure feature.
        self._peer_reports: dict[str, list[Evidence]] = defaultdict(list)
        self._seen: set[tuple[str, int]] = set()
        self._beacon: dict[tuple[str, str], tuple[int, bool]] = {}

    # -- inputs ------------------------------------------------------------

    def observe(self, obs: Observation) -> None:
        key = (obs.node, obs.peer)
        if obs.channel is Channel.BEACON:
            # Beacon health is a feature of the primary link's diagnosis, not a
            # link to be diagnosed in its own right.
            self._beacon[key] = (obs.t, obs.outcome is Outcome.OK)
            return

        self._recent[obs.node][obs.peer] = (obs.t, obs.outcome)
        b = self._belief.get(key)
        if b is None:
            b = np.array([nominal_belief()[c.value] for c in CAUSES])

        # Predict forward over however many grid steps have elapsed, then
        # correct. Skipping the predict on long gaps would let a stale belief
        # stay confident through hours of silence.
        #
        # CARRY THE REMAINDER. Advancing _last_t to obs.t discards the
        # sub-grid part of every interval, and with cfg.step_s = 10 the
        # quotient is then 0 on every single update -- the predict step never
        # runs at all and every per-cause dynamic in T_MATRIX is inert. The
        # kernel's leak is what bounds the belief's log-odds range, so without
        # it a handful of bad contacts drives NOMINAL down nine orders of
        # magnitude and no amount of subsequent good news recovers it.
        # Measured: 60000 ms spacing -> nominal 0.9988; 59999 ms -> 0.5137.
        # One millisecond per observation decided the diagnosis. FINDINGS F-015.
        last = self._last_t.get(key, obs.t)
        elapsed = max(0, obs.t - last)
        steps = min(elapsed // self.grid_ms, 240)
        for _ in range(steps):
            b = T_MATRIX.T @ b
        consumed = steps * self.grid_ms

        b = b * self._likelihood(obs)
        s = b.sum()
        b = b / s if s > 0 else np.array([nominal_belief()[c.value] for c in CAUSES])

        self._belief[key] = b
        # Advance by whole grid steps only, so the unconsumed remainder carries
        # into the next interval. Also monotone: a late-arriving observation
        # must not rewind this link's clock.
        self._last_t[key] = max(self._last_t.get(key, last), last + consumed)

    def receive(self, evidence: list[Evidence]) -> None:
        for e in evidence:
            if e.key in self._seen:
                continue  # (origin, seq) dedup -- repeats must not inflate confidence
            self._seen.add(e.key)
            self._peer_reports[e.peer].append(e)

    # -- features ----------------------------------------------------------

    def _self_cofail(self, obs: Observation) -> bool | None:
        """Are my OTHER links failing right now too?

        Returns None when I have no other recent link to compare against. This
        is the identifiability condition (PLAN 5.2) surfacing inside the
        inference, and getting it wrong is subtle: returning False here would
        mean 'my other links are fine', which the likelihood reads as positive
        evidence AGAINST a fault on my side -- from a node that has no other
        links at all.

        That is precisely the degree-1 case the strip paints red. With no
        concurrent contact there is no discriminating evidence in either
        direction, and the filter must stay ambiguous rather than manufacture
        confidence from absence.
        """
        recent = self._recent.get(obs.node, {})
        window = 15 * 60 * 1000   # identifiability needs SIMULTANEITY; a pass
                                  # that closed two hours ago is not concurrent
        others = [(t, o) for p, (t, o) in recent.items()
                  if p != obs.peer and 0 <= (obs.t - t) <= window]
        if not others:
            return None
        bad = sum(1 for _, o in others if o is not Outcome.OK)
        return (bad / len(others)) >= COFAIL_THRESHOLD

    def _peer_cofail(self, obs: Observation) -> bool | None:
        """Are OTHER nodes failing with this same peer? None if nobody has told
        us -- absence of gossip is not evidence of absence, and treating it as
        such is how a local-only filter talks itself into false confidence."""
        window = 4 * 3600 * 1000
        reports = [e for e in self._peer_reports.get(obs.peer, ())
                   if e.origin != obs.node and 0 <= (obs.t - e.t) <= window]
        if not reports:
            return None
        bad = sum(1 for e in reports if e.outcome is not Outcome.OK)
        return (bad / len(reports)) >= COFAIL_THRESHOLD

    def _likelihood(self, obs: Observation) -> np.ndarray:
        self_cf = self._self_cofail(obs)
        peer_cf = self._peer_cofail(obs)

        beacon = self._beacon.get((obs.node, obs.peer))
        beacon_ok = None
        if beacon and 0 <= (obs.t - beacon[0]) <= 15 * 60 * 1000:
            beacon_ok = beacon[1]

        lik = np.ones(N_CAUSES)
        for c in CAUSES:
            i = CAUSE_INDEX[c]
            p = P_OUTCOME[c].get(obs.outcome, 0.01)
            if self_cf is not None:
                p *= _bern(P_SELF_COFAIL[c], self_cf)
            if peer_cf is not None:
                p *= _bern(P_PEER_COFAIL[c], peer_cf)
            if beacon_ok is not None:
                p *= _bern(P_BEACON_OK[c], beacon_ok)
            p *= _bern(P_QUEUE_DEEP[c], obs.queue_bytes >= QUEUE_DEEP_BYTES)
            # ONE-SIDED. A long silence is informative; a short one is not.
            # Applying the complement (1 - p) when silence is short made
            # "this only just started" strong evidence for weather over
            # pointing -- 0.90 vs 0.40, a 2.25x ratio on the very first
            # observation of a fault, before duration could possibly have
            # said anything. That reintroduced exactly the degree-1
            # discrimination that sharing the outcome row removed.
            if obs.ms_since_ok >= LONG_SILENCE_MS:
                p *= P_LONG_SILENCE[c]
            lik[i] = max(p, 1e-9)
        return lik

    # -- output ------------------------------------------------------------

    def belief_for(self, node: str, peer: str) -> dict[str, float]:
        b = self._belief.get((node, peer))
        if b is None:
            return nominal_belief()
        return {c.value: float(b[CAUSE_INDEX[c]]) for c in CAUSES}

    def decide(self, node: str, t: int) -> Policy:
        """Report on the node's worst link -- the one furthest from NOMINAL.

        A node with several links holds several beliefs; the policy has to name
        one peer, so it names the one most in need of a decision.
        """
        links = [(p, b) for (n, p), b in self._belief.items() if n == node]
        if not links:
            return Policy(nominal_belief(), *bayes_action(nominal_belief())[:1],
                          target="", until=t, confidence=1.0)

        nom = CAUSE_INDEX[Cause.NOMINAL]
        peer, b = min(links, key=lambda kv: kv[1][nom])
        belief = {c.value: float(b[CAUSE_INDEX[c]]) for c in CAUSES}
        action, _ = bayes_action(belief)
        return Policy(
            belief=belief, action=action, target=peer,
            until=t + 30 * 60 * 1000,
            confidence=float(b.max()),
            rationale="",   # the filter has no natural language; that is the point
        )


class NullAdvisor:
    """Runs the identical path and concludes nothing. The baseline arm."""

    name = "null"

    def observe(self, obs: Observation) -> None:
        pass

    def receive(self, evidence: list[Evidence]) -> None:
        pass

    def decide(self, node: str, t: int) -> Policy:
        from ..types import Action
        return Policy(nominal_belief(), Action.NONE, "", t, 1.0, "")
