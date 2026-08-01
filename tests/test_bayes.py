"""Bayes-filter test suite: sim/advisor/bayes.py, sim/advisor/base.py.

Contracts under test come from PLAN.md sections 5.2 and 5.4, from the physics
in sim/observe.py and PLAN.md 5.3, and from the separability table. Nothing in
this file derives an expectation from the emission tables in bayes.py -- doing
that would only assert that the code equals itself, and three bugs have already
shipped through exactly that gap. Every expected posterior below is argued from
what the fault physically does to a link, then checked.

What this file does NOT test, deliberately
------------------------------------------
That beliefs sum to 1. They do (see `test_belief_stays_on_simplex_...`), and it
is worth one cheap check, but every bug found so far produced a perfectly
normalised, confident, plausible-looking posterior. Normalisation is not
evidence of correctness here.

Marking convention
------------------
`@defect(...)` marks a test that encodes behaviour the implementation does not
currently have. By default it is `xfail(strict=True)`, so the suite stays green
AND the moment someone fixes the implementation the test XPASSes and fails,
forcing the marker off. Every `defect` reason carries a `file:line`.

    python -m pytest tests/test_bayes.py -q         # green; defects are xfail
    python -m pytest tests/test_bayes.py -q -rxX    # list every defect + reason

To make the defects fail loudly instead, set BAYES_STRICT=1:

    $env:BAYES_STRICT=1; python -m pytest tests/test_bayes.py -q   # PowerShell
    BAYES_STRICT=1 python -m pytest tests/test_bayes.py -q         # bash
"""

from __future__ import annotations

import itertools
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import diagnosability as D                            # noqa: E402
from sim.advisor import bayes as B                             # noqa: E402
from sim.advisor.base import (                                 # noqa: E402
    ACTIONS,
    COST,
    Advisor,
    bayes_action,
    nominal_belief,
    uniform_belief,
)
from sim.advisor.bayes import BayesAdvisor, NullAdvisor        # noqa: E402
from sim.config import ScenarioConfig                          # noqa: E402
from sim.types import (                                        # noqa: E402
    CAUSE_INDEX,
    CAUSES,
    Action,
    Cause,
    Channel,
    Contact,
    Evidence,
    Observation,
    Outcome,
)

MIN_MS = 60_000
HOUR_MS = 3_600_000
T0 = 1_700_000_000_000   # arbitrary absolute epoch ms; the filter is epoch-free

STRICT = os.environ.get("BAYES_STRICT", "").lower() not in ("", "0", "false", "no")


def defect(ref: str, reason: str):
    """Mark a test that asserts correct behaviour the implementation lacks.

    A falsy condition makes the mark inert, so under BAYES_STRICT the test runs
    for real and the defect shows up as an ordinary failure. Usable both as a
    decorator and inside `pytest.param(marks=...)`.
    """
    return pytest.mark.xfail(not STRICT, strict=True, reason=f"{ref} -- {reason}")


# ==========================================================================
# helpers
# ==========================================================================

def _obs(t: int, *, node: str = "SAT_A", peer: str = "GS_1",
         outcome: Outcome = Outcome.SILENT, channel: Channel = Channel.PRIMARY,
         queue_bytes: int = 0, elevation_deg: float = 30.0,
         measured_rate: int = 0, expected_rate: int = 160_000_000,
         ms_since_ok: int = 0, shift_observed: int = 0,
         peer_degree: int = 1, self_degree: int = 1,
         contact_id: str = "c") -> Observation:
    """One Observation with every field explicit. Defaults are the neutral
    degree-1 downlink: no concurrency, empty queue, mid elevation."""
    return Observation(
        t=t, node=node, peer=peer, contact_id=contact_id, outcome=outcome,
        measured_rate=measured_rate, expected_rate=expected_rate,
        elevation_deg=elevation_deg, channel=channel, queue_bytes=queue_bytes,
        ms_since_ok=ms_since_ok, shift_observed=shift_observed,
        peer_degree=peer_degree, self_degree=self_degree,
    )


def _ev(origin: str, seq: int, t: int, peer: str = "GS_1",
        outcome: Outcome = Outcome.SILENT, degree: int = 1) -> Evidence:
    return Evidence(origin=origin, seq=seq, t=t, peer=peer,
                    outcome=outcome, degree=degree)


DEEP_QUEUE = B.QUEUE_DEEP_BYTES * 4
EMPTY_QUEUE = 0


def _top(belief: dict[str, float]) -> tuple[str, float]:
    k = max(belief, key=belief.__getitem__)
    return k, belief[k]


def _vec(belief: dict[str, float]) -> np.ndarray:
    return np.array([belief[c.value] for c in CAUSES])


def _fresh() -> BayesAdvisor:
    return BayesAdvisor(grid_ms=60_000)


# ==========================================================================
# 1. SEMANTIC CORRECTNESS
#
# One scenario per cause. Each observation sequence is built from what the
# fault physically does (PLAN 5.3 + sim/observe.py), never from bayes.py's
# tables. The assertion is that a correct filter resolves to that cause.
# ==========================================================================

def test_weather_resolves_when_gossip_confirms_the_station_is_the_common_factor():
    """Weather lives on the GROUND STATION. Physical signature (PLAN 5.3):
    it hits every satellite at that station, and only at that station.

    So: my link to GS_1 fails, my link to GS_2 is fine (rules out anything on
    my side), and gossip says other satellites are also failing at GS_1 (the
    peer is the common factor). That is the definition of weather, and it is
    the one signature only gossip can supply.
    """
    a = _fresh()
    a.observe(_obs(T0, peer="GS_2", outcome=Outcome.OK))            # I am fine elsewhere
    a.receive([_ev("SAT_B", 1, T0, "GS_1", Outcome.SILENT),
               _ev("SAT_C", 1, T0, "GS_1", Outcome.SILENT)])        # peers fail at GS_1
    for k in range(3):
        a.observe(_obs(T0 + k * MIN_MS, peer="GS_1", outcome=Outcome.SILENT))

    cause, p = _top(a.belief_for("SAT_A", "GS_1"))
    assert cause == Cause.WEATHER.value, f"got {cause} at {p:.3f}"
    assert p > 0.9, f"weather only reached {p:.3f}"


def test_gossip_is_what_collapses_the_weather_ambiguity():
    """The module docstring's central claim: local-only stays ambiguous, and
    evidence from a peer that failed at the same station collapses it.

    The comparison has to be made at degree 1, with no other link of my own.
    A healthy concurrent link is already the identifiability condition
    (diagnosability.py:22) and resolves weather without any gossip -- that is
    the previous test, and it is why this one must not include one. Identical
    local observations both times; only the gossip differs.
    """
    def run(with_gossip: bool) -> dict[str, float]:
        a = _fresh()
        if with_gossip:
            a.receive([_ev("SAT_B", 1, T0, "GS_1", Outcome.SILENT),
                       _ev("SAT_C", 1, T0, "GS_1", Outcome.SILENT)])
        for k in range(3):
            a.observe(_obs(T0 + k * MIN_MS, peer="GS_1", outcome=Outcome.SILENT))
        return a.belief_for("SAT_A", "GS_1")

    local_only = run(False)
    with_gossip = run(True)
    assert with_gossip[Cause.WEATHER.value] > 0.9
    assert _top(local_only)[0] != Cause.WEATHER.value, (
        "local-only already named weather; the gossip demo proves nothing")
    assert local_only[Cause.WEATHER.value] < 0.2


def test_node_down_resolves_when_every_link_including_the_beacon_is_silent():
    """A dead spacecraft takes everything with it -- PLAN 5.3: 'Hits every link
    of that node, including its beacon'. The beacon is the discriminator: it is
    the one thing a mispointed high-gain dish leaves alone, so a silent beacon
    plus silent everything-else can only be the node.

    Gossip says other satellites at GS_1 are fine, which rules out weather.
    """
    a = _fresh()
    a.receive([_ev("SAT_B", 1, T0, "GS_1", Outcome.OK),
               _ev("SAT_C", 1, T0, "GS_1", Outcome.OK)])
    for k in range(3):
        t = T0 + k * MIN_MS
        a.observe(_obs(t, peer="GS_2", outcome=Outcome.SILENT))
        a.observe(_obs(t, peer="GS_3", outcome=Outcome.SILENT))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.SILENT, channel=Channel.BEACON))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.SILENT))

    cause, p = _top(a.belief_for("SAT_A", "GS_1"))
    assert cause == Cause.NODE_DOWN.value, f"got {cause} at {p:.3f}"
    assert p > 0.9, f"node_down only reached {p:.3f}"


def test_pointing_resolves_when_primary_degrades_everywhere_and_the_beacon_holds():
    """Pointing error, partial regime. PLAN 5.3: 'Hits all peers of that
    satellite, primary channel only -- beacon stays nominal', and L_dB grows as
    12(theta/theta_3dB)^2, so a moderate offset degrades rather than kills.

    Signature: primary degraded at EVERY station (fault is on my side), beacon
    nominal (rules out node_down), queue empty (rules out buffer).
    """
    a = _fresh()
    a.receive([_ev("SAT_B", 1, T0, "GS_1", Outcome.OK),
               _ev("SAT_C", 1, T0, "GS_1", Outcome.OK)])
    for k in range(3):
        t = T0 + k * MIN_MS
        a.observe(_obs(t, peer="GS_2", outcome=Outcome.DEGRADED, queue_bytes=EMPTY_QUEUE))
        a.observe(_obs(t, peer="GS_3", outcome=Outcome.DEGRADED, queue_bytes=EMPTY_QUEUE))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.OK, channel=Channel.BEACON))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.DEGRADED, queue_bytes=EMPTY_QUEUE))

    cause, p = _top(a.belief_for("SAT_A", "GS_1"))
    assert cause == Cause.POINTING.value, f"got {cause} at {p:.3f}"
    assert p > 0.9, f"pointing only reached {p:.3f}"


def test_pointing_resolves_when_the_primary_is_dead_and_the_beacon_is_nominal():
    """Pointing error, total regime -- theta well past theta_3dB, primary link
    gone. This is the SHARPEST pointing signature in PLAN 5.3, not the weakest:
    'primary channel only -- beacon stays nominal'. A satellite whose high-gain
    link is dead at every station while its low-gain TT&C beacon answers
    normally is mispointed. Nothing else in the model does that:

      node_down    kills the beacon too      (sim/observe.py:49-52)
      weather      is per-station, not all   (and gossip here says peers are OK)
      buffer       still forwards, degraded  (sim/observe.py:70-73)
      stale_sched  shifts EVERY contact of the node, beacon included
                   (sim/observe.py:75-77 -- no channel guard)

    The filter says stale_sched.
    """
    a = _fresh()
    a.receive([_ev("SAT_B", 1, T0, "GS_1", Outcome.OK),
               _ev("SAT_C", 1, T0, "GS_1", Outcome.OK)])
    for k in range(3):
        t = T0 + k * MIN_MS
        a.observe(_obs(t, peer="GS_2", outcome=Outcome.SILENT))
        a.observe(_obs(t, peer="GS_3", outcome=Outcome.SILENT))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.OK, channel=Channel.BEACON))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.SILENT))

    cause, p = _top(a.belief_for("SAT_A", "GS_1"))
    assert cause == Cause.POINTING.value, f"got {cause} at {p:.3f}"


def test_buffer_resolves_on_sustained_degradation_with_a_deep_queue():
    """Buffer saturation is the only cause DEFINED by a full queue (PLAN 5.3:
    'Fractional loss keyed to queue occupancy... Physical layer clean'). It is
    partial by construction and never total.

    Signature: DEGRADED, never SILENT; queue deep; beacon nominal (physical
    layer clean); affects my links generally, not one station's.
    """
    a = _fresh()
    for k in range(3):
        t = T0 + k * MIN_MS
        a.observe(_obs(t, peer="GS_2", outcome=Outcome.DEGRADED, queue_bytes=DEEP_QUEUE))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.OK, channel=Channel.BEACON))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.DEGRADED, queue_bytes=DEEP_QUEUE))

    cause, p = _top(a.belief_for("SAT_A", "GS_1"))
    assert cause == Cause.BUFFER.value, f"got {cause} at {p:.3f}"
    assert p > 0.9, f"buffer only reached {p:.3f}"


def test_buffer_is_ruled_out_when_the_queue_is_empty():
    """Regression guard for FINDINGS F-013. Identical degradation, empty queue
    instead of deep: buffer must not be the answer. This is the bug that made
    DEGRADED mean 'buffer' regardless of whether the queue held a single byte."""
    a = _fresh()
    for k in range(3):
        t = T0 + k * MIN_MS
        a.observe(_obs(t, peer="GS_2", outcome=Outcome.DEGRADED, queue_bytes=EMPTY_QUEUE))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.OK, channel=Channel.BEACON))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.DEGRADED, queue_bytes=EMPTY_QUEUE))

    b = a.belief_for("SAT_A", "GS_1")
    assert _top(b)[0] != Cause.BUFFER.value
    assert b[Cause.BUFFER.value] < 0.1, f"buffer still at {b[Cause.BUFFER.value]:.3f}"


def test_stale_sched_resolves_on_a_late_carrier():
    """PLAN 5.3: 'Contact is shifted, not absent. Same shift across all that
    node's contacts.' LATE means out-of-window listening caught the carrier
    outside its window -- a contact that HAPPENED, just not when planned. No
    other modelled cause moves a contact in time; the other four attenuate or
    remove it."""
    a = _fresh()
    for k in range(3):
        t = T0 + k * MIN_MS
        a.observe(_obs(t, peer="GS_2", outcome=Outcome.LATE, shift_observed=120_000))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.LATE, shift_observed=120_000))

    cause, p = _top(a.belief_for("SAT_A", "GS_1"))
    assert cause == Cause.STALE_SCHED.value, f"got {cause} at {p:.3f}"
    assert p > 0.9, f"stale_sched only reached {p:.3f}"


def test_nominal_is_the_answer_when_nothing_is_wrong():
    """NOMINAL is a member of Cause for exactly this reason (types.py:24)."""
    a = _fresh()
    for k in range(5):
        t = T0 + k * MIN_MS
        a.observe(_obs(t, peer="GS_2", outcome=Outcome.OK))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.OK, channel=Channel.BEACON))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.OK))
    cause, p = _top(a.belief_for("SAT_A", "GS_1"))
    assert cause == Cause.NOMINAL.value and p > 0.9


# ==========================================================================
# 2. AMBIGUITY MUST SURVIVE WHERE THEORY DEMANDS IT
#
# PLAN 5.2 / diagnosability.py:14-20: when deg(s,t) == 1 and deg(g,t) == 1,
# weather@g and pointing@s predict the identical observation set and
#
#     p(weather | o) / p(pointing | o)  ==  p(weather) / p(pointing)
#
# for all t and all observations. Both priors are 0.02 (base.py:34), so the
# ratio must stay at 1.0. Nothing can move it. A filter that drifts is
# manufacturing information it does not have.
# ==========================================================================

AMBIGUITY_BOUND = 3.0   # generous: theory says the ratio is pinned at exactly 1.0


@pytest.mark.parametrize("outcome", [
    pytest.param(Outcome.SILENT, marks=defect(
        "sim/advisor/bayes.py:50-63",
        "P_OUTCOME rows differ across weather/pointing/node_down, so each "
        "observation multiplies the posterior ratio by a constant != 1. Over a "
        "single 16-minute pass the weather/pointing ratio reaches ~17x with no "
        "concurrency and no gossip -- information the geometry cannot supply.")),
    pytest.param(Outcome.DEGRADED, marks=defect(
        "sim/advisor/bayes.py:P_STAY",
        "Drifts to 3.28 against a bound of 3.0 after 10 observations. NOT a "
        "likelihood leak: the weather and pointing outcome rows are now "
        "identical and the instantaneous ratio is exactly 1.0. The drift is the "
        "transition kernel acting on genuinely different dwell times (weather "
        "P_STAY 0.85, pointing 0.97) -- a fault still running after ten "
        "observations really is more likely pointing than rain. PLAN 5.2 "
        "overstates the invariant by asserting it over the posterior rather "
        "than the likelihood. Documented deviation, see PLAN C-009.")),
])
def test_degree_one_with_no_gossip_keeps_weather_and_pointing_ambiguous(outcome):
    """A node with exactly one open contact, no gossip, no beacon. This is the
    unidentifiable case the strip paints red, and the case bayes.py:210-223
    explicitly promises to handle ('the filter must stay ambiguous rather than
    manufacture confidence from absence').

    Even a 3x bound is far looser than theory allows; a 10x bound also fails,
    by the 5th observation.
    """
    a = _fresh()
    for k in range(10):
        a.observe(_obs(T0 + k * MIN_MS, peer="GS_1", outcome=outcome,
                       peer_degree=1, self_degree=1))
    b = a.belief_for("SAT_A", "GS_1")
    ratio = b[Cause.POINTING.value] / b[Cause.WEATHER.value]
    assert 1 / AMBIGUITY_BOUND <= ratio <= AMBIGUITY_BOUND, (
        f"pointing/weather drifted to {ratio:.3g} after 10 observations at "
        f"degree 1; prior ratio is 1.0 and PLAN 5.2 says it must stay there")


@defect("sim/advisor/bayes.py:246-267",
        "the likelihood ignores peer_degree/self_degree, so it never learns "
        "that this moment is unidentifiable and reaches 0.92 on node_down "
        "after 5 SILENT observations at degree 1.")
def test_degree_one_never_becomes_confident_about_any_concurrency_pair_member():
    """Weaker and even harder to argue with than the ratio bound: at degree 1
    the three causes joined by the 'concurrency' rule in
    diagnosability.PAIR_RULES (weather, pointing, node_down) are mutually
    inseparable, so no single one of them may take the posterior."""
    for outcome in (Outcome.SILENT, Outcome.DEGRADED):
        a = _fresh()
        for k in range(10):
            a.observe(_obs(T0 + k * MIN_MS, outcome=outcome,
                           peer_degree=1, self_degree=1))
            b = a.belief_for("SAT_A", "GS_1")
            for c in (Cause.WEATHER, Cause.POINTING, Cause.NODE_DOWN):
                assert b[c.value] <= 0.9, (
                    f"{outcome.value}: {c.value} hit {b[c.value]:.3f} after "
                    f"{k + 1} observations at degree 1")


# ==========================================================================
# 3. DATA-INCEST GUARD
# ==========================================================================

def _peer_reports_setup(a: BayesAdvisor) -> None:
    """Two peers reporting the station is FINE. With one adverse report added
    the bad fraction is 1/3 = 0.333, a hair under COFAIL_THRESHOLD (0.34).
    That is deliberate: it puts the test right on the cliff, so a dedup that
    silently stopped working would flip the feature and be caught."""
    a.receive([_ev("SAT_B", 1, T0, "GS_1", Outcome.OK),
               _ev("SAT_C", 1, T0, "GS_1", Outcome.OK)])


def test_the_dedup_test_below_is_actually_sharp():
    """Control. Proves the x50 assertion is not vacuous: the SAME payload with
    DISTINCT seqs does flip peer-cofailure from False to True, so a broken
    dedup would change the answer and the next test would fail."""
    a = _fresh()
    _peer_reports_setup(a)
    a.receive([_ev("SAT_D", 7, T0, "GS_1", Outcome.SILENT)])
    probe = _obs(T0, peer="GS_1", outcome=Outcome.SILENT)
    assert a._peer_cofail(probe) is False

    a2 = _fresh()
    _peer_reports_setup(a2)
    a2.receive([_ev("SAT_D", 7 + i, T0, "GS_1", Outcome.SILENT) for i in range(50)])
    assert a2._peer_cofail(probe) is True, (
        "50 distinct-seq reports did not move the feature; the dedup test "
        "below would be vacuous")


@pytest.mark.parametrize("delivery", ["one_batch", "one_at_a_time", "interleaved"])
def test_repeating_the_same_evidence_leaves_the_posterior_bit_identical(delivery):
    """(origin, seq) is the dedup key (types.py:180). Fifty deliveries of one
    observation are one observation. Bit-identical, not approximately equal --
    any drift here is confidence conjured from a retransmission."""
    dup = _ev("SAT_D", 7, T0, "GS_1", Outcome.SILENT)

    once = _fresh()
    _peer_reports_setup(once)
    once.receive([dup])
    once.observe(_obs(T0, peer="GS_1", outcome=Outcome.SILENT))
    expected = once.belief_for("SAT_A", "GS_1")

    many = _fresh()
    _peer_reports_setup(many)
    if delivery == "one_batch":
        many.receive([dup] * 50)
    elif delivery == "one_at_a_time":
        for _ in range(50):
            many.receive([dup])
    else:
        for i in range(50):
            many.receive([dup])
            if i % 7 == 0:
                many.receive([_ev("SAT_B", 1, T0, "GS_1", Outcome.OK)])  # also a dup
    many.observe(_obs(T0, peer="GS_1", outcome=Outcome.SILENT))

    assert many.belief_for("SAT_A", "GS_1") == expected
    assert len(many._peer_reports["GS_1"]) == 3


def test_distinct_seq_from_the_same_origin_must_count_as_new_evidence():
    """The other direction. Same origin, different seq is a different
    observation and must move the posterior -- a dedup keyed on origin alone
    would silence a peer after its first report."""
    a = _fresh()
    _peer_reports_setup(a)
    a.receive([_ev("SAT_D", 7, T0, "GS_1", Outcome.SILENT)])
    a.observe(_obs(T0, peer="GS_1", outcome=Outcome.SILENT))
    before = a.belief_for("SAT_A", "GS_1")

    a.receive([_ev("SAT_D", 8, T0, "GS_1", Outcome.SILENT)])
    a.observe(_obs(T0 + MIN_MS, peer="GS_1", outcome=Outcome.SILENT))
    after = a.belief_for("SAT_A", "GS_1")

    assert len(a._peer_reports["GS_1"]) == 4
    assert after[Cause.WEATHER.value] > before[Cause.WEATHER.value], (
        "a second report from the same origin did not raise weather")


def test_dedup_is_global_not_per_peer():
    """(origin, seq) is unique per origin, full stop. The same key arriving
    labelled with a different peer is still a replay of one observation."""
    a = _fresh()
    a.receive([_ev("SAT_B", 1, T0, "GS_1", Outcome.SILENT)])
    a.receive([_ev("SAT_B", 1, T0, "GS_2", Outcome.SILENT)])
    assert len(a._peer_reports["GS_2"]) == 0


def test_a_nodes_own_gossip_does_not_confirm_itself():
    """Evidence whose origin is me is my own observation coming back around the
    network. Counting it as peer confirmation is textbook incest."""
    a = _fresh()
    a.receive([_ev("SAT_A", 1, T0, "GS_1", Outcome.SILENT),
               _ev("SAT_A", 2, T0, "GS_1", Outcome.SILENT),
               _ev("SAT_A", 3, T0, "GS_1", Outcome.SILENT)])
    assert a._peer_cofail(_obs(T0, peer="GS_1", outcome=Outcome.SILENT)) is None


# ==========================================================================
# 4. ADVERSARIAL PROBES
# ==========================================================================

def _degree_one_plan(cfg: ScenarioConfig):
    """SAT_A <-> GS_1 alone from minute 10 to 25. Neither endpoint has any
    other contact during that window, so deg(SAT_A) == deg(GS_1) == 1 and
    diagnosability.compute must mark the whole window unidentifiable."""
    epoch = D._epoch_ms(cfg)

    def c(cid, src, dst, t0m, t1m):
        return Contact(id=cid, src=src, dst=dst,
                       t_open=epoch + int(t0m * MIN_MS),
                       t_close=epoch + int(t1m * MIN_MS),
                       rate_bps=160_000_000, kind="downlink", max_elev=30.0)

    return epoch, [
        c("a1", "SAT_A", "GS_1", 10, 25), c("a2", "GS_1", "SAT_A", 10, 25),
        c("b1", "SAT_B", "GS_2", 40, 50), c("b2", "GS_2", "SAT_B", 40, 50),
    ]


def test_the_unidentifiable_window_really_is_unidentifiable():
    """Precondition for the probe below: assert against diagnosability.py, the
    module the whole project rests on, that this window is red."""
    cfg = ScenarioConfig(horizon_hours=1, grid_s=60)
    epoch, plan = _degree_one_plan(cfg)
    strip = D.compute(plan, cfg)
    idx = [i for i, t in enumerate(strip.t_grid)
           if epoch + 10 * MIN_MS <= t <= epoch + 25 * MIN_MS]

    assert len(idx) >= 10
    assert not any(strip.per_link["GS_1|SAT_A"][i] for i in idx)
    for c1, c2 in itertools.combinations(
            [Cause.WEATHER, Cause.POINTING, Cause.NODE_DOWN], 2):
        sep = D.pair_separable(strip, ("SAT_A", "GS_1"), c1, c2)
        assert not any(sep[i] for i in idx), f"{c1.value}/{c2.value} claims separable"


@pytest.mark.parametrize("outcome", [
    pytest.param(Outcome.SILENT, marks=defect(
        "sim/advisor/bayes.py:246-267",
        "reaches P(node_down)=0.92 after 5 observations at grid points "
        "diagnosability.compute marks unidentifiable. THE headline defect: the "
        "filter is confidently wrong exactly where the project's own theory "
        "says no estimator can know anything.")),
    pytest.param(Outcome.DEGRADED, marks=defect(
        "sim/advisor/bayes.py:246-267",
        "same, P(pointing)=0.94 after 5 observations.")),
])
def test_filter_is_never_confident_where_diagnosability_says_it_cannot_be(outcome):
    """The adversarial probe the brief asked for, and it needs no adversarial
    construction at all: a plain repeated observation on an ordinary degree-1
    downlink does it.

    Observations are placed on the exact grid points that
    diagnosability.compute marks unidentifiable (asserted in the test above),
    and the causes checked are exactly the three joined by the 'concurrency'
    rule in PAIR_RULES -- the ones whose separability that flag governs.
    """
    cfg = ScenarioConfig(horizon_hours=1, grid_s=60)
    epoch, plan = _degree_one_plan(cfg)
    strip = D.compute(plan, cfg)
    grid = strip.t_grid
    window = [(i, t) for i, t in enumerate(grid)
              if epoch + 10 * MIN_MS <= t <= epoch + 25 * MIN_MS]

    a = BayesAdvisor(grid_ms=cfg.grid_s * 1000)
    for n, (i, t) in enumerate(window, 1):
        a.observe(_obs(t, outcome=outcome, peer_degree=1, self_degree=1))
        b = a.belief_for("SAT_A", "GS_1")
        for c1, c2 in itertools.combinations(
                [Cause.WEATHER, Cause.POINTING, Cause.NODE_DOWN], 2):
            if D.pair_separable(strip, ("SAT_A", "GS_1"), c1, c2)[i]:
                continue
            for c in (c1, c2):
                assert b[c.value] <= 0.9, (
                    f"P({c.value})={b[c.value]:.4f} after {n} {outcome.value} "
                    f"observations, at a grid point where {c1.value}/{c2.value} "
                    f"is marked INSEPARABLE by diagnosability.compute")


def test_self_cofailure_requires_concurrency_not_a_two_hour_lookback():
    """`deg(n, t)` in diagnosability.py counts contacts open AT TIME t. Contacts
    are ~10 minutes long, so a 2-hour lookback admits roughly a dozen passes
    that share no instant with the one being diagnosed. The observation carries
    `self_degree` -- the exact quantity -- and the likelihood ignores it."""
    a = _fresh()
    a.observe(_obs(T0 - 119 * MIN_MS, peer="GS_9", outcome=Outcome.SILENT,
                   self_degree=1))
    probe = _obs(T0, peer="GS_1", outcome=Outcome.SILENT, self_degree=1)
    assert a._self_cofail(probe) is None, (
        "a pass that closed 119 minutes ago is not a concurrent link")


@defect("sim/advisor/bayes.py:226-232",
        "same root cause, observed end to end: the belief crosses 0.9 on "
        "node_down purely on the strength of a non-concurrent stale entry.")
def test_a_stale_non_concurrent_link_cannot_manufacture_confidence():
    a = _fresh()
    a.observe(_obs(T0 - 100 * MIN_MS, peer="GS_9", outcome=Outcome.SILENT))
    for k in range(6):
        a.observe(_obs(T0 + k * MIN_MS, peer="GS_1", outcome=Outcome.SILENT,
                       self_degree=1, peer_degree=1))
    cause, p = _top(a.belief_for("SAT_A", "GS_1"))
    assert p <= 0.9, f"{cause} reached {p:.4f} at self_degree=1"


# -- unused Observation fields ---------------------------------------------
#
# Each case varies ONE field between two values that a correct likelihood must
# distinguish, and asserts the likelihood moves. `queue_bytes` is the control:
# it was exactly this bug (FINDINGS F-013) and it is now wired in.

_FIELD_RATIONALE = {
    "ms_since_ok":
        "Elapsed silence. PLAN 5.2's table: 'node_down transient vs permanent "
        "-- NEVER at a single t; only elapsed silence separates them'. It is "
        "also the cheapest weather discriminator there is: weather self-heals "
        "on a geometric dwell with a ~5 min median (P_STAY 0.85), so ten hours "
        "of unbroken silence is ~140 dwell times and rules weather out on its "
        "own. SHOULD BE USED.",
    "elevation_deg":
        "Rain fade scales as 1/sin(elevation) -- PLAN 5.3, implemented in "
        "sim/observe.py:59-61. A contact degraded at 5 deg is ordinary "
        "weather; the same degradation at 85 deg, straight up through minimum "
        "atmosphere, is far more likely to be the spacecraft. This "
        "discriminates weather from pointing WITHIN a single link, so it works "
        "at degree 1 where concurrency cannot. Given the degree-1 drift "
        "documented above, this is the highest-value missing feature in the "
        "model. SHOULD BE USED.",
    "measured_rate":
        "With expected_rate, this is the fade depth -- the difference between "
        "'barely off nominal' and 'one notch above lock loss', both of which "
        "collapse to DEGRADED (sim/observe.py:96-102). Depth separates buffer "
        "(fractional, tracks offered load) from pointing "
        "(12(theta/theta_3dB)^2, steep) and grades weather severity. The "
        "filter throws the magnitude away and keeps only the 4-way bucket. "
        "SHOULD BE USED.",
    "peer_degree":
        "deg(g, t) -- literally the left half of the identifiability condition "
        "in diagnosability.py:22. With peer_degree > 1 the peer has concurrent "
        "contacts and weather is separable; at 1 it is not. Reading it is how "
        "the filter would know when it is entitled to be confident. SHOULD BE "
        "USED -- this is the root of the headline defect above.",
    "self_degree":
        "deg(s, t) -- the right half of the same condition, and the honest "
        "version of what _self_cofail approximates with a 2-hour lookback. "
        "SHOULD BE USED.",
    "shift_observed":
        "The measured carrier offset. LATE already carries most of the "
        "stale_sched signal so this is the weakest of the six, but a shift in "
        "PLAN 5.3's stated 1-5 min band, consistent across the node's "
        "contacts, is direct confirmation, and an out-of-band value is "
        "evidence against. Marginal -- SHOULD BE USED, lowest priority.",
    "queue_bytes":
        "CONTROL -- this one is wired in (bayes.py:265) and must stay wired "
        "in. Regression guard for FINDINGS F-013.",
}

_FIELD_CASES = [
    pytest.param("ms_since_ok", 0, 10 * HOUR_MS,
                 marks=defect("sim/advisor/bayes.py:246-267",
                              "ms_since_ok never read; the one observable "
                              "PLAN 5.2 names as the ONLY separator for "
                              "transient vs permanent node_down.")),
    pytest.param("elevation_deg", 5.0, 85.0,
                 marks=defect("sim/advisor/bayes.py:246-267",
                              "elevation_deg never read; weather's defining "
                              "geometry-dependence is invisible to the "
                              "filter.")),
    pytest.param("measured_rate", 990_000, 10,
                 marks=defect("sim/advisor/bayes.py:246-267",
                              "measured_rate/expected_rate never read; all "
                              "degradations look alike regardless of depth.")),
    pytest.param("peer_degree", 1, 5,
                 marks=defect("sim/advisor/bayes.py:246-267",
                              "peer_degree never read; the filter cannot tell "
                              "an identifiable moment from an unidentifiable "
                              "one.")),
    pytest.param("self_degree", 1, 5,
                 marks=defect("sim/advisor/bayes.py:246-267",
                              "self_degree never read; _self_cofail uses a "
                              "2-hour lookback as a proxy for it, unsoundly.")),
    pytest.param("shift_observed", 0, 180_000,
                 marks=defect("sim/advisor/bayes.py:246-267",
                              "shift_observed never read.")),
    pytest.param("queue_bytes", EMPTY_QUEUE, DEEP_QUEUE),
]


@pytest.mark.parametrize("field,lo,hi", _FIELD_CASES)
def test_likelihood_uses_the_observation_fields_it_is_given(field, lo, hi):
    """Every field in Observation is there because something needs it
    (types.py:112-127). A field the likelihood never reads is either dead
    weight in the contract or a discriminator left on the floor. Each case
    varies ONE field between two values a correct likelihood must distinguish;
    the reasoning for each is in _FIELD_RATIONALE above."""
    import dataclasses
    a = _fresh()
    base = _obs(T0, outcome=Outcome.DEGRADED, measured_rate=500_000,
                expected_rate=1_000_000)
    lik_lo = a._likelihood(dataclasses.replace(base, **{field: lo}))
    lik_hi = a._likelihood(dataclasses.replace(base, **{field: hi}))
    assert not np.allclose(lik_lo, lik_hi), (
        f"{field}: {lo!r} and {hi!r} give an identical likelihood vector. "
        f"{_FIELD_RATIONALE[field]}")


def test_every_observation_field_has_a_documented_verdict():
    """Guard against a field being added to the contract and quietly ignored:
    the audit above must cover every field that could carry fault information."""
    audited = {p.values[0] for p in _FIELD_CASES} | {
        "t", "node", "peer", "contact_id", "outcome", "channel", "expected_rate"}
    assert set(Observation.__dataclass_fields__) == audited


# -- long silence and the predict cap --------------------------------------

def test_a_long_gap_relaxes_the_belief_toward_nominal():
    """Sanity: silence must cost confidence. bayes.py:186-192."""
    a = _fresh()
    for k in range(6):
        t = T0 + k * MIN_MS
        a.observe(_obs(t, peer="GS_2", outcome=Outcome.SILENT))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.SILENT, channel=Channel.BEACON))
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.SILENT))
    hot = a.belief_for("SAT_A", "GS_1")[Cause.NODE_DOWN.value]

    a.observe(_obs(T0 + 5 * MIN_MS + 3 * HOUR_MS, peer="GS_1", outcome=Outcome.OK))
    cooled = a.belief_for("SAT_A", "GS_1")[Cause.NODE_DOWN.value]
    assert cooled < hot


def test_predict_saturates_at_the_240_step_cap():
    """Characterization of bayes.py:190. 240 steps on a 60 s grid is 4 hours;
    past that, elapsed time stops entering the filter at all. Locked in so the
    cap cannot be changed silently -- and so the next test's claim is precise."""
    def after(gap_ms: int) -> dict[str, float]:
        a = _fresh()
        t = T0
        for _ in range(6):
            a.observe(_obs(t, peer="GS_2", outcome=Outcome.SILENT))
            a.observe(_obs(t, peer="GS_1", outcome=Outcome.SILENT, channel=Channel.BEACON))
            a.observe(_obs(t, peer="GS_1", outcome=Outcome.SILENT))
            t += MIN_MS
        a.observe(_obs(t - MIN_MS + gap_ms, peer="GS_1", outcome=Outcome.OK))
        return a.belief_for("SAT_A", "GS_1")

    assert after(3 * HOUR_MS) != after(4 * HOUR_MS)          # below the cap: time matters
    at_cap = after(4 * HOUR_MS)
    assert after(8 * HOUR_MS) == at_cap                      # at/above: bit-identical
    assert after(24 * HOUR_MS) == at_cap
    assert after(30 * 24 * HOUR_MS) == at_cap
    assert after(365 * 24 * HOUR_MS) == at_cap


@defect("sim/advisor/bayes.py:190",
        "min(..., 240) means four hours of silence and a year of silence "
        "produce a bit-identical belief. Defensible for the absorbing states, "
        "but it also caps how far a stale belief can ever relax, and it is the "
        "second time the module quantises elapsed time lossily.")
def test_a_year_of_silence_relaxes_further_than_four_hours():
    def after(gap_ms: int) -> dict[str, float]:
        a = _fresh()
        t = T0
        for _ in range(6):
            a.observe(_obs(t, peer="GS_2", outcome=Outcome.SILENT))
            a.observe(_obs(t, peer="GS_1", outcome=Outcome.SILENT, channel=Channel.BEACON))
            a.observe(_obs(t, peer="GS_1", outcome=Outcome.SILENT))
            t += MIN_MS
        a.observe(_obs(t - MIN_MS + gap_ms, peer="GS_1", outcome=Outcome.OK))
        return a.belief_for("SAT_A", "GS_1")

    assert (after(365 * 24 * HOUR_MS)[Cause.NODE_DOWN.value]
            < after(4 * HOUR_MS)[Cause.NODE_DOWN.value])


# -- sub-grid time quantisation --------------------------------------------

def _weather_then_recovery(spacing_ms: int) -> dict[str, float]:
    """Four contacts fail while gossip says the station is rained out, then
    forty consecutive perfect contacts. Any filter must end at NOMINAL."""
    a = _fresh()
    a.receive([_ev("SAT_B", 1, T0, "GS_1", Outcome.SILENT),
               _ev("SAT_C", 1, T0, "GS_1", Outcome.SILENT)])
    t = T0
    for _ in range(4):
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.SILENT))
        t += spacing_ms
    for _ in range(40):
        a.observe(_obs(t, peer="GS_1", outcome=Outcome.OK))
        t += spacing_ms
    return a.belief_for("SAT_A", "GS_1")


@pytest.mark.parametrize("spacing_ms", [
    120_000,
    60_000,
    59_999,
    30_000,
    10_000,
])
def test_recovery_does_not_depend_on_observation_spacing(spacing_ms):
    """Forty perfect contacts in a row mean the link is healthy. How often the
    engine chose to sample is not a fact about the link."""
    b = _weather_then_recovery(spacing_ms)
    assert b[Cause.NOMINAL.value] > 0.9, (
        f"spacing {spacing_ms} ms: nominal only {b[Cause.NOMINAL.value]:.4f}, "
        f"weather still {b[Cause.WEATHER.value]:.4f} after 40 OK contacts")


def test_one_millisecond_of_timestamp_jitter_does_not_change_the_diagnosis():
    """Identical observations, identical order, spans equal to within 60 ms."""
    on_grid = _weather_then_recovery(60_000)
    off_grid = _weather_then_recovery(59_999)
    delta = float(np.abs(_vec(on_grid) - _vec(off_grid)).max())
    assert delta < 0.05, f"L-inf = {delta:.4f}"


def test_out_of_order_observations_do_not_rewind_the_links_clock():
    a = _fresh()
    a.observe(_obs(T0 + 100 * MIN_MS, peer="GS_1", outcome=Outcome.SILENT))
    a.observe(_obs(T0, peer="GS_1", outcome=Outcome.SILENT))
    assert a._last_t[("SAT_A", "GS_1")] >= T0 + 100 * MIN_MS


# -- gossip age discounting ------------------------------------------------

@defect("sim/advisor/bayes.py:125,238-244",
        "EVIDENCE_HALFLIFE_MS is declared and NEVER REFERENCED anywhere in the "
        "module -- grep it. PLAN 5.5 requires a cause-dependent age discount "
        "lambda^dt and calls the asymmetry 'the interesting part of doing "
        "inference over a delay-tolerant link'. It is not implemented: "
        "_peer_cofail applies a flat 4-hour window instead.")
def test_stale_gossip_is_discounted_by_age():
    """Weather self-heals on a geometric dwell with a ~5 minute median
    (P_STAY 0.85 on a 60 s grid). Gossip that a station was rained out 239
    minutes ago is roughly 55 half-lives stale and is worth almost nothing --
    yet it produces a bit-identical posterior to gossip from one minute ago.

    A node failure is absorbing, so old evidence about it stays nearly as good
    as fresh. That asymmetry is precisely what PLAN 5.5 asks for.
    """
    def weather_after(age_ms: int) -> float:
        a = _fresh()
        a.receive([_ev("SAT_B", 1, T0 - age_ms, "GS_1", Outcome.SILENT),
                   _ev("SAT_C", 1, T0 - age_ms, "GS_1", Outcome.SILENT)])
        a.observe(_obs(T0, peer="GS_1", outcome=Outcome.SILENT))
        return a.belief_for("SAT_A", "GS_1")[Cause.WEATHER.value]

    fresh = weather_after(1 * MIN_MS)
    stale = weather_after(239 * MIN_MS)
    assert stale < fresh * 0.5, (
        f"239-minute-old weather gossip is worth {stale:.6f} against "
        f"{fresh:.6f} for one-minute-old -- no discount at all")


@defect("sim/advisor/bayes.py:238-240",
        "the 4-hour window is a step function: gossip 239 minutes old drives "
        "weather to 0.75, gossip 241 minutes old is discarded entirely and the "
        "posterior jumps to node_down 0.30. Two minutes of age flips the "
        "diagnosis. A decay would not have a cliff.")
def test_the_gossip_window_is_not_a_cliff():
    def belief_after(age_ms: int) -> dict[str, float]:
        a = _fresh()
        a.receive([_ev("SAT_B", 1, T0 - age_ms, "GS_1", Outcome.SILENT),
                   _ev("SAT_C", 1, T0 - age_ms, "GS_1", Outcome.SILENT)])
        a.observe(_obs(T0, peer="GS_1", outcome=Outcome.SILENT))
        return a.belief_for("SAT_A", "GS_1")

    just_inside = belief_after(239 * MIN_MS)
    just_outside = belief_after(241 * MIN_MS)
    delta = float(np.abs(_vec(just_inside) - _vec(just_outside)).max())
    assert delta < 0.2, f"two minutes of gossip age moved the belief by {delta:.3f}"


def test_features_are_causal():
    """A beacon reading taken 90 minutes after the contact under diagnosis must
    not change what the filter believed at the time of that contact."""
    a = _fresh()
    a.observe(_obs(T0 + 90 * MIN_MS, peer="GS_1", outcome=Outcome.OK,
                   channel=Channel.BEACON))
    a.observe(_obs(T0, peer="GS_1", outcome=Outcome.SILENT))

    causal = _fresh()
    causal.observe(_obs(T0, peer="GS_1", outcome=Outcome.SILENT))
    assert a.belief_for("SAT_A", "GS_1") == causal.belief_for("SAT_A", "GS_1")


# ==========================================================================
# 5. CHEAP INVARIANTS
# ==========================================================================

def test_t_matrix_rows_sum_to_one():
    assert B.T_MATRIX.shape == (len(CAUSES), len(CAUSES))
    np.testing.assert_allclose(B.T_MATRIX.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(B.T_MATRIX >= 0.0)


def test_t_matrix_has_no_direct_fault_to_fault_transition():
    """bayes.py:129-131: leaving a fault returns to NOMINAL, matching the
    one-active-fault-per-target invariant the generator enforces."""
    nom = CAUSE_INDEX[Cause.NOMINAL]
    for i, ci in enumerate(CAUSES):
        for j, cj in enumerate(CAUSES):
            if i == j or i == nom or j == nom:
                continue
            assert B.T_MATRIX[i, j] == 0.0, f"{ci.value} -> {cj.value}"


def test_nominal_leaks_to_every_fault_equally():
    nom = CAUSE_INDEX[Cause.NOMINAL]
    leaks = {c.value: B.T_MATRIX[nom, CAUSE_INDEX[c]]
             for c in CAUSES if c is not Cause.NOMINAL}
    assert len(set(leaks.values())) == 1, leaks
    assert all(v > 0 for v in leaks.values()), "a fault the prior can never enter"


@pytest.mark.parametrize("cause", [Cause.NODE_DOWN, Cause.STALE_SCHED])
def test_node_down_and_stale_sched_are_near_absorbing(cause):
    """PLAN 5.4: node_down is 'absorbing -- no action escapes it'; stale_sched
    is 'absorbing until a schedule refresh'."""
    i = CAUSE_INDEX[cause]
    assert B.T_MATRIX[i, i] >= 0.99, f"{cause.value} self-transition too leaky"

    v = np.zeros(len(CAUSES))
    v[i] = 1.0
    for _ in range(60):
        v = B.T_MATRIX.T @ v
    assert int(v.argmax()) == i, (
        f"{cause.value} lost the plurality after one hour of pure prediction")
    assert v[i] > 0.5


@pytest.mark.parametrize("cause", [Cause.WEATHER, Cause.BUFFER])
def test_self_healing_causes_are_not_absorbing(cause):
    """The counterpart: weather self-heals on a geometric dwell and buffer
    clears. If these were absorbing too, the per-cause dynamics PLAN 5.4 calls
    the point of the model would carry no information."""
    i = CAUSE_INDEX[cause]
    v = np.zeros(len(CAUSES))
    v[i] = 1.0
    for _ in range(60):
        v = B.T_MATRIX.T @ v
    assert int(v.argmax()) == CAUSE_INDEX[Cause.NOMINAL]
    assert v[i] < 0.1


def test_belief_stays_on_simplex_over_long_random_streams():
    """Cheap and nearly worthless on its own -- every bug found in this filter
    produced a perfectly normalised posterior. Kept only to catch NaN and
    underflow-to-zero-mass."""
    rng = np.random.default_rng(20260801)
    outcomes = list(Outcome)
    channels = list(Channel)
    for _ in range(8):
        a = _fresh()
        t = T0
        for _ in range(300):
            t += int(rng.integers(0, 20 * MIN_MS))
            a.observe(_obs(t,
                           peer=f"GS_{rng.integers(1, 4)}",
                           outcome=outcomes[rng.integers(len(outcomes))],
                           channel=channels[rng.integers(len(channels))],
                           queue_bytes=int(rng.integers(0, 4 * B.QUEUE_DEEP_BYTES))))
            if rng.random() < 0.3:
                a.receive([_ev(f"SAT_{rng.integers(1, 5)}",
                               int(rng.integers(0, 10 ** 6)),
                               t - int(rng.integers(0, 6 * HOUR_MS)),
                               f"GS_{rng.integers(1, 4)}",
                               outcomes[rng.integers(len(outcomes))])])
        for key, vec in a._belief.items():
            assert np.all(np.isfinite(vec)), key
            assert np.all(vec >= 0.0), key
            assert vec.sum() == pytest.approx(1.0, abs=1e-12), key
            assert vec.max() > 0.0, key


def test_belief_for_an_unseen_link_is_the_prior():
    a = _fresh()
    assert a.belief_for("SAT_A", "GS_1") == nominal_belief()


@pytest.mark.parametrize("fn", [nominal_belief, uniform_belief])
def test_prior_helpers_lie_on_the_simplex(fn):
    b = fn()
    assert set(b) == {c.value for c in CAUSES}
    assert sum(b.values()) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in b.values())


def test_nominal_prior_favours_nominal():
    """base.py:30-33: a filter starting at uniform spends its first several
    updates just relearning that most links are fine."""
    b = nominal_belief()
    assert b[Cause.NOMINAL.value] > 0.5
    assert b[Cause.NOMINAL.value] > max(
        v for k, v in b.items() if k != Cause.NOMINAL.value)


# -- decision rule ---------------------------------------------------------

def _reference_action(belief: dict[str, float]) -> tuple[Action, float]:
    """Independent brute force over COST. First minimum in ACTIONS order wins,
    matching base.py:93-98's strict `<`."""
    best, best_cost = None, float("inf")
    for a in ACTIONS:
        c = 0.0
        for cause in CAUSES:
            c += belief.get(cause.value, 0.0) * COST[cause][a]
        if c < best_cost:
            best, best_cost = a, c
    return best, best_cost


def test_bayes_action_is_the_true_argmin_over_cost():
    rng = np.random.default_rng(4242)
    beliefs = []
    for c in CAUSES:                                    # vertices
        beliefs.append({d.value: (1.0 if d is c else 0.0) for d in CAUSES})
    for c1, c2 in itertools.combinations(CAUSES, 2):    # edges
        for w in (0.25, 0.5, 0.75):
            beliefs.append({d.value: (w if d is c1 else (1 - w) if d is c2 else 0.0)
                            for d in CAUSES})
    beliefs.append(nominal_belief())
    beliefs.append(uniform_belief())
    for conc in (0.05, 0.3, 1.0, 5.0):                  # interior
        for v in rng.dirichlet(np.ones(len(CAUSES)) * conc, size=1500):
            beliefs.append({c.value: float(v[CAUSE_INDEX[c]]) for c in CAUSES})

    for belief in beliefs:
        got_action, got_cost = bayes_action(belief)
        ref_action, ref_cost = _reference_action(belief)
        assert got_cost == pytest.approx(ref_cost, rel=1e-12, abs=1e-15)
        costs = sorted(sum(belief.get(c.value, 0.0) * COST[c][a] for c in CAUSES)
                       for a in ACTIONS)
        if costs[1] - costs[0] > 1e-12:                 # skip measure-zero ties
            assert got_action == ref_action, belief


def test_bayes_action_tolerates_a_partial_belief_dict():
    """base.py:95 uses .get(..., 0.0); an advisor may hand back a sparse dict."""
    action, cost = bayes_action({Cause.NODE_DOWN.value: 1.0})
    assert action is Action.BLACKLIST and cost == pytest.approx(0.0)


def test_every_action_is_selectable_somewhere_on_the_simplex():
    """An action no belief ever selects is dead weight in COST and a modelling
    bug: either the cost row is dominated or the action is not distinct."""
    rng = np.random.default_rng(99)
    seen: set[Action] = set()
    for c in CAUSES:
        seen.add(bayes_action({d.value: (1.0 if d is c else 0.0)
                               for d in CAUSES})[0])
    for conc in (0.05, 0.15, 0.5, 1.0, 3.0):
        for v in rng.dirichlet(np.ones(len(CAUSES)) * conc, size=4000):
            seen.add(bayes_action({c.value: float(v[CAUSE_INDEX[c]])
                                   for c in CAUSES})[0])
    missing = set(ACTIONS) - seen
    assert not missing, f"never selectable for any belief: {[a.value for a in missing]}"


def test_each_causes_own_remedy_is_optimal_when_that_cause_is_certain():
    """COST's diagonal is what makes an action 'correct' (base.py:58-64).
    stale_sched is the exception by design -- there is no schedule-refresh
    action, so WAIT is the best available."""
    expected = {
        Cause.NOMINAL: Action.NONE,
        Cause.WEATHER: Action.WAIT,
        Cause.NODE_DOWN: Action.BLACKLIST,
        Cause.POINTING: Action.REROUTE,
        Cause.BUFFER: Action.THROTTLE,
        Cause.STALE_SCHED: Action.WAIT,
    }
    for cause, action in expected.items():
        belief = {c.value: (1.0 if c is cause else 0.0) for c in CAUSES}
        assert bayes_action(belief)[0] is action, cause


def test_cost_matrix_is_complete():
    assert set(COST) == set(CAUSES)
    for cause, row in COST.items():
        assert set(row) == set(ACTIONS), cause
        assert all(v >= 0.0 for v in row.values()), cause


# -- policy surface --------------------------------------------------------

def test_decide_with_no_observations_returns_the_prior_and_does_nothing():
    p = _fresh().decide("SAT_A", T0)
    assert p.belief == nominal_belief()
    assert p.action is Action.NONE
    assert p.target == ""
    assert p.until == T0


def test_decide_reports_the_link_furthest_from_nominal():
    """bayes.py:277-282: a node holds one belief per link but a Policy names
    one peer, so it names the one most in need of a decision."""
    a = _fresh()
    for k in range(4):
        t = T0 + k * MIN_MS
        a.observe(_obs(t, peer="GS_HEALTHY", outcome=Outcome.OK))
        a.observe(_obs(t, peer="GS_BROKEN", outcome=Outcome.SILENT))
    p = a.decide("SAT_A", T0 + 4 * MIN_MS)
    assert p.target == "GS_BROKEN"
    assert p.belief == a.belief_for("SAT_A", "GS_BROKEN")


def test_decide_is_self_consistent():
    a = _fresh()
    for k in range(4):
        a.observe(_obs(T0 + k * MIN_MS, peer="GS_1", outcome=Outcome.DEGRADED,
                       queue_bytes=DEEP_QUEUE))
    p = a.decide("SAT_A", T0)
    assert sum(p.belief.values()) == pytest.approx(1.0)
    assert p.confidence == pytest.approx(max(p.belief.values()))
    assert p.action is bayes_action(p.belief)[0]
    assert p.rationale == ""              # the filter has no natural language
    assert p.until > T0


def test_decide_ignores_other_nodes_links():
    a = _fresh()
    a.observe(_obs(T0, node="SAT_A", peer="GS_1", outcome=Outcome.OK))
    a.observe(_obs(T0, node="SAT_B", peer="GS_9", outcome=Outcome.SILENT))
    assert a.decide("SAT_A", T0).target == "GS_1"
    assert a.decide("SAT_B", T0).target == "GS_9"


def test_beacon_observations_do_not_create_a_link_to_diagnose():
    """bayes.py:175-179: beacon health is a feature of the primary link's
    diagnosis, not a link in its own right."""
    a = _fresh()
    a.observe(_obs(T0, peer="GS_1", outcome=Outcome.OK, channel=Channel.BEACON))
    assert a._belief == {}
    assert a.decide("SAT_A", T0).target == ""


def test_advisors_satisfy_the_protocol():
    """base.py:39-55. The comparison between arms is only meaningful if they
    are genuinely interchangeable."""
    for advisor in (BayesAdvisor(), NullAdvisor()):
        assert isinstance(advisor, Advisor)
        assert isinstance(advisor.name, str) and advisor.name


def test_null_advisor_accepts_the_identical_input_stream_and_concludes_nothing():
    """base.py:8-11: if the null case skipped observation construction the arms
    would differ in more than the advisor."""
    n = NullAdvisor()
    n.observe(_obs(T0, peer="GS_1", outcome=Outcome.SILENT))
    n.receive([_ev("SAT_B", 1, T0, "GS_1", Outcome.SILENT)])
    p = n.decide("SAT_A", T0)
    assert p.belief == nominal_belief()
    assert p.action is Action.NONE


def test_advisor_never_sees_ground_truth():
    """base.py:13-15: an advisor never sees FaultTrace, the engine, or anything
    else constituting ground truth. Observation is the only channel, and it
    must not carry a cause label."""
    fields = set(Observation.__dataclass_fields__)
    forbidden = {"cause", "kind", "fault", "fault_kind", "severity", "truth",
                 "t_start", "t_end", "target"}
    assert not (fields & forbidden), fields & forbidden

    src = pathlib.Path(B.__file__).read_text(encoding="utf-8")
    for banned in ("FaultEvent", "FaultTrace", "from ..faults", "import faults"):
        assert banned not in src, f"bayes.py references {banned}"


def test_filters_for_different_links_are_independent():
    """bayes.py:156: one filter per (node, peer) link."""
    a = _fresh()
    for k in range(5):
        a.observe(_obs(T0 + k * MIN_MS, peer="GS_1", outcome=Outcome.SILENT))
    assert a.belief_for("SAT_A", "GS_2") == nominal_belief()
    assert a.belief_for("SAT_Z", "GS_1") == nominal_belief()


def test_belief_for_returns_a_copy_not_the_live_vector():
    a = _fresh()
    a.observe(_obs(T0, peer="GS_1", outcome=Outcome.SILENT))
    b = a.belief_for("SAT_A", "GS_1")
    b[Cause.NODE_DOWN.value] = 99.0
    assert a.belief_for("SAT_A", "GS_1")[Cause.NODE_DOWN.value] != 99.0


def test_observing_is_deterministic():
    """Two advisors fed the identical stream must agree bit for bit; the whole
    evaluation is replay-based."""
    def run() -> dict[str, float]:
        a = _fresh()
        a.receive([_ev("SAT_B", 1, T0, "GS_1", Outcome.SILENT)])
        for k in range(20):
            a.observe(_obs(T0 + k * MIN_MS, peer="GS_1",
                           outcome=Outcome.SILENT if k % 3 else Outcome.OK,
                           queue_bytes=k * 40_000))
        return a.belief_for("SAT_A", "GS_1")

    assert run() == run()
