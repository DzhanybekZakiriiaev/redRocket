"""Fault -> observable mapping, and Observation construction. PLAN.md 5.3 / 4.3.

This module is where the project's central premise lives: **all five causes
produce the same immediate observable.** A contact that doesn't open looks
identical whether the ground station is rained out, the spacecraft is dead, its
antenna is mispointed, its buffer is full, or the schedule is stale.

Separation has to come from the pattern across time and across peers -- never
from a single observation. So this module deliberately throws information away:
it maps a rich `FaultEvent` down to a narrow `Outcome`, and it is the ONLY place
that touches ground truth on the way to an `Observation`.

Everything downstream sees only `Observation`. If a field here leaked the cause
-- a `fault_kind` left in for logging, an `expected_rate` derived from post-fault
reality -- the whole evaluation would be measuring nothing.
"""

from __future__ import annotations

import math

from .types import (
    Contact, FaultEvent, Observation, Cause, Outcome, Channel,
)

# Below this fraction of expected rate a contact reads as silent rather than
# degraded. Real receivers have a lock threshold; without one, every fault
# would be distinguishable by its exact rate, which is precisely the leak we
# are trying to avoid.
LOCK_THRESHOLD = 0.15


def active_faults(faults: list[FaultEvent], node: str, t: int) -> list[FaultEvent]:
    return [f for f in faults if f.target == node and f.t_start <= t <= f.t_end]


def effective_rate(contact: Contact, faults: list[FaultEvent], t: int,
                   channel: Channel = Channel.PRIMARY) -> tuple[int, int]:
    """(effective_rate_bps, schedule_shift_ms) for this contact at time t.

    Applies every fault active on either endpoint. Returns the shift separately
    because a stale schedule moves the window rather than attenuating it.
    """
    rate = float(contact.rate_bps)
    shift = 0

    for node in (contact.src, contact.dst):
        for f in active_faults(faults, node, t):
            if f.kind is Cause.NODE_DOWN:
                # Total and binary. Kills every channel including the beacon --
                # that is what separates it from a pointing error.
                return 0, shift

            if f.kind is Cause.WEATHER:
                # Attenuation scales as 1/sin(elevation), so the low-elevation
                # edges of a pass degrade before the middle does. ISLs are above
                # the atmosphere and are untouched.
                if contact.kind == "downlink":
                    el = max(contact.max_elev, 1.0)
                    depth = f.severity / max(math.sin(math.radians(el)), 0.15)
                    rate *= max(0.0, 1.0 - min(depth, 1.0))

            elif f.kind is Cause.POINTING:
                # High-gain link only. THE BEACON SURVIVES. This asymmetry is
                # the single strongest discriminator against node_down, and it
                # is why Observation carries a channel field at all.
                if channel is Channel.PRIMARY:
                    rate *= max(0.0, 1.0 - f.severity)

            elif f.kind is Cause.BUFFER:
                # Partial by construction, never total -- a saturated buffer
                # still forwards, just less.
                rate *= max(0.0, 1.0 - f.severity)

            elif f.kind is Cause.STALE_SCHED:
                # The contact is not lost. It happens somewhere else in time.
                shift += f.shift_ms

    return int(rate), shift


def classify(measured: int, expected: int, shift_ms: int,
             listening_window_ms: int) -> Outcome:
    """Collapse to one of four observable states.

    This is the deliberate information loss. Note that a stale schedule reads
    as SILENT unless the receiver is listening far enough outside the window to
    catch the shifted carrier -- which is exactly why out-of-window listening is
    a hard requirement in PLAN.md section 6.
    """
    if shift_ms:
        if abs(shift_ms) <= listening_window_ms:
            return Outcome.LATE
        return Outcome.SILENT
    if expected <= 0:
        return Outcome.SILENT
    frac = measured / expected
    if frac >= 0.85:
        return Outcome.OK
    if frac < LOCK_THRESHOLD:
        return Outcome.SILENT
    return Outcome.DEGRADED


def observe(contact: Contact, faults: list[FaultEvent], t: int,
            *, node: str, peer: str, queue_bytes: int, ms_since_ok: int,
            peer_degree: int, self_degree: int,
            channel: Channel = Channel.PRIMARY,
            listening_window_ms: int = 300_000) -> Observation:
    """Build the one object the diagnosis layer is allowed to see."""
    measured, shift = effective_rate(contact, faults, t, channel)
    outcome = classify(measured, contact.rate_bps, shift, listening_window_ms)
    return Observation(
        t=t, node=node, peer=peer, contact_id=contact.id,
        outcome=outcome,
        measured_rate=measured if outcome is not Outcome.SILENT else 0,
        expected_rate=contact.rate_bps,
        elevation_deg=contact.max_elev,
        channel=channel,
        queue_bytes=queue_bytes,
        ms_since_ok=ms_since_ok,
        shift_observed=shift if outcome is Outcome.LATE else 0,
        peer_degree=peer_degree,
        self_degree=self_degree,
    )


# --------------------------------------------------------------------------
# PLAN.md requirement F4 -- the acceptance gate for this stage.
# --------------------------------------------------------------------------

def f4_gate(contacts: list[Contact], verbose: bool = True) -> dict:
    """All five causes must produce the SAME observable at first occurrence.

    If they don't, one of them is trivially identifiable from a single
    observation, there is no diagnosis problem for that cause, and the premise
    is broken. Better to fail here than at S8.

    Each cause is injected at maximum severity onto an identical contact and
    the resulting PRIMARY-channel outcome is compared.
    """
    from .faults import pointing_loss_db  # noqa: F401  (documents the model)

    probe = next(c for c in contacts if c.kind == "downlink")
    t = probe.t_open + 1000
    span = (probe.t_open - 60_000, probe.t_close + 60_000)

    cases: dict[Cause, FaultEvent] = {
        Cause.WEATHER: FaultEvent(*span, probe.dst, Cause.WEATHER, 1.0),
        Cause.NODE_DOWN: FaultEvent(*span, probe.src, Cause.NODE_DOWN, 1.0),
        Cause.POINTING: FaultEvent(*span, probe.src, Cause.POINTING, 1.0),
        Cause.BUFFER: FaultEvent(*span, probe.src, Cause.BUFFER, 1.0),
        # Shift beyond the listening window, so the contact is simply absent.
        Cause.STALE_SCHED: FaultEvent(*span, probe.src, Cause.STALE_SCHED, 1.0,
                                      shift_ms=600_000),
    }

    results = {}
    for cause, ev in cases.items():
        o = observe(probe, [ev], t, node=probe.src, peer=probe.dst,
                    queue_bytes=0, ms_since_ok=0, peer_degree=1, self_degree=1)
        results[cause] = o.outcome

    outcomes = set(results.values())
    passed = len(outcomes) == 1

    if verbose:
        print(f"F4 gate -- identical observable at first occurrence: "
              f"{'PASS' if passed else 'FAIL'}")
        for cause, out in results.items():
            print(f"  {cause.value:12s} -> {out.value}")
        if not passed:
            print(f"  !! {len(outcomes)} distinct outcomes: "
                  f"{sorted(o.value for o in outcomes)}")

    return {"pass": passed,
            "outcomes": {c.value: o.value for c, o in results.items()}}


def f4_gate_distributional(contacts: list[Contact], cfg=None,
                           n_draws: int = 400, verbose: bool = True) -> dict:
    """The gate that actually matters.

    `f4_gate` injects every cause at severity 1.0, where everything is silent
    trivially -- it is necessary but far too easy. The real question is whether
    the causes stay confusable at the severities the generator actually draws.

    Perfect overlap is NOT the target here. Some separation is legitimate and
    expected: a saturated buffer really does degrade rather than vanish, and
    that is a genuine physical signal, not a leak. What would break the premise
    is a cause whose outcome distribution is UNIQUE -- one that can be named
    from a single observation with no pattern, no history, and no peers.
    """
    from collections import Counter
    from .config import DEFAULT
    from . import faults as F

    cfg = cfg or DEFAULT
    rng = F._rng(cfg)
    probes = [c for c in contacts if c.kind == "downlink"][:n_draws]

    dist: dict[Cause, Counter] = {}
    for cause in (Cause.WEATHER, Cause.NODE_DOWN, Cause.POINTING,
                  Cause.BUFFER, Cause.STALE_SCHED):
        counter: Counter = Counter()
        for probe in probes:
            span = (probe.t_open - 60_000, probe.t_close + 60_000)
            t = probe.t_open + 1000
            target = probe.dst if cause is Cause.WEATHER else probe.src
            if cause is Cause.WEATHER:
                ev = F.weather(rng, target, span[0])
            elif cause is Cause.NODE_DOWN:
                ev = F.node_down(rng, target, span[0], span[1])
            elif cause is Cause.POINTING:
                ev = F.pointing(rng, target, span[0], span[1])
            elif cause is Cause.BUFFER:
                ev = F.buffer(rng, target, span[0], span[1])
            else:
                ev = F.stale_sched(rng, target, span[0], span[1])
            ev = FaultEvent(span[0], span[1], target, ev.kind,
                            ev.severity, ev.shift_ms)
            o = observe(probe, [ev], t, node=probe.src, peer=probe.dst,
                        queue_bytes=0, ms_since_ok=0,
                        peer_degree=1, self_degree=1)
            counter[o.outcome.value] += 1
        dist[cause] = counter

    # Scope of the gate (see FINDINGS F-010):
    #
    # `ok` is excluded. It means no fault was observed at all -- a low-severity
    # pointing error that leaves the link above threshold is not a diagnostic
    # signature, it is a non-event.
    #
    # STALE_SCHED is excluded from the confusability requirement, because the
    # plan's own separability table (section 6) states that its signature IS a
    # time shift rather than an absence. Requiring it to be indistinguishable
    # would contradict the physics we deliberately modelled. It remains
    # conditionally identifiable: it reads LATE only when the shift falls inside
    # the listening window, and SILENT -- fully confusable -- when it does not.
    # That conditionality is the interesting part and is checked separately.
    ABSENCE_CAUSES = (Cause.WEATHER, Cause.NODE_DOWN, Cause.POINTING, Cause.BUFFER)

    unique: dict[str, list[str]] = {}
    for cause in ABSENCE_CAUSES:
        counter = dist[cause]
        others = set().union(*(set(dist[c]) for c in ABSENCE_CAUSES if c is not cause))
        solo = sorted(set(counter) - others - {Outcome.OK.value})
        if solo:
            unique[cause.value] = solo

    passed = not unique

    if verbose:
        n = len(probes)
        print(f"F4 distributional gate over {n} real contacts, "
              f"generator-drawn severities: {'PASS' if passed else 'FAIL'}")
        for cause, counter in dist.items():
            share = "  ".join(f"{k}={v/n:.2f}" for k, v in sorted(counter.items()))
            print(f"  {cause.value:12s} {share}")
        if unique:
            for cause, solo in unique.items():
                print(f"  !! {cause} is trivially identifiable: "
                      f"only it ever yields {solo}")
        ss = dist[Cause.STALE_SCHED]
        late = ss.get(Outcome.LATE.value, 0) / max(sum(ss.values()), 1)
        print(f"  stale_sched is conditionally identifiable by design: "
              f"LATE {late:.0%} (caught by out-of-window listening), "
              f"SILENT {1-late:.0%} (shift exceeds the window -> fully confusable)")

    return {"pass": passed,
            "dist": {c.value: dict(d) for c, d in dist.items()},
            "unique": unique}


if __name__ == "__main__":
    from . import contacts as C
    r = f4_gate(C.read())
    raise SystemExit(0 if r["pass"] else 1)
