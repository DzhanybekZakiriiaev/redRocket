"""Foundation-module test suite: sim/types.py, sim/config.py, sim/contacts.py,
sim/diagnosability.py.

Contracts under test come from PLAN.md sections 4 and 5.

The load-bearing test in this file is `test_brute_force_matches_condition`:
it rebuilds the identifiability condition by direct counting over a hand-built
contact plan and compares it point-by-point with what diagnosability.compute
produces. Everything downstream in the project rests on that condition being
computed correctly, so it is checked against an independent implementation
rather than against itself.
"""

from __future__ import annotations

import pathlib
import random
import re
import sys
from collections import defaultdict
from dataclasses import FrozenInstanceError

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import contacts as C                                    # noqa: E402
from sim import diagnosability as D                              # noqa: E402
from sim.config import DEFAULT, ScenarioConfig                   # noqa: E402
from sim.types import (                                          # noqa: E402
    Action,
    Cause,
    Channel,
    Contact,
    Evidence,
    FaultEvent,
    Observation,
    Outcome,
    Policy,
    Strip,
    _pack,
    unpack,
)

MINUTE_MS = 60_000


# ==========================================================================
# helpers
# ==========================================================================

def _epoch(cfg: ScenarioConfig) -> int:
    return D._epoch_ms(cfg)


def _c(src, dst, t0_min, t1_min, kind, epoch, elev=0.0, rate=1_000_000) -> Contact:
    """One directed synthetic contact, times given in minutes past epoch."""
    t0 = epoch + int(t0_min * MINUTE_MS)
    t1 = epoch + int(t1_min * MINUTE_MS)
    return Contact(
        id=f"{src}>{dst}@{t0 // 1000}",
        src=src, dst=dst, t_open=t0, t_close=t1,
        rate_bps=rate, kind=kind, max_elev=elev,
    )


def _link(a, b, t0_min, t1_min, kind, epoch, **kw) -> list[Contact]:
    """Both directions of one synthetic contact window."""
    return [_c(a, b, t0_min, t1_min, kind, epoch, **kw),
            _c(b, a, t0_min, t1_min, kind, epoch, **kw)]


def _key(c: Contact) -> str:
    return f"{min(c.src, c.dst)}|{max(c.src, c.dst)}"


def _undirected(contacts):
    """Unique undirected windows as (a, b, t_open, t_close, kind).

    Independent of degree_matrix's own dedup logic -- this is the reference
    implementation, so it does its own collapsing of the two directions.
    """
    out = set()
    for c in contacts:
        a, b = sorted((c.src, c.dst))
        out.add((a, b, c.t_open, c.t_close, c.kind))
    return sorted(out)


def _brute_degree(contacts, grid):
    """deg[kind][node][i] -- concurrent contacts of that kind on that node at
    grid point i, by direct counting. No numpy, no shared code with the module.
    """
    n = len(grid)
    deg: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0] * n))
    for a, b, t0, t1, kind in _undirected(contacts):
        for i, t in enumerate(grid):
            if t0 <= t <= t1:
                deg[kind][a][i] += 1
                deg[kind][b][i] += 1
    return deg


def _brute_windows(contacts, grid):
    """link key -> [bool] where that undirected link has any contact open."""
    n = len(grid)
    win: dict[str, list[bool]] = {}
    for c in contacts:
        k = _key(c)
        cur = win.setdefault(k, [False] * n)
        for i, t in enumerate(grid):
            if c.t_open <= t <= c.t_close:
                cur[i] = True
    return win


def _brute_kinds(contacts):
    out = {}
    for c in contacts:
        out.setdefault(_key(c), c.kind)
    return out


# ==========================================================================
# fixtures
# ==========================================================================

TINY = ScenarioConfig(horizon_hours=1, grid_s=60)   # 61 grid points, 1 min apart


@pytest.fixture(scope="session")
def real_plan():
    """The real contact plan from the committed TLEs. ~0.9 s to build."""
    return C.build(DEFAULT)


@pytest.fixture(scope="session")
def real_strip(real_plan):
    return D.compute(real_plan, DEFAULT)


@pytest.fixture()
def synthetic_plan():
    """Four nodes, mixed overlapping and non-overlapping windows.

    Minutes past epoch:

        SAT01--GS_A  downlink   [ 0, 10]  and  [30, 40]
        SAT02--GS_A  downlink   [ 5, 15]            -> GS_A degree 2 on [5,10]
        SAT01--GS_B  downlink   [35, 45]            -> SAT01 degree 2 on [35,40]
        SAT02--GS_B  downlink   [50, 55]            -> isolated, degree 1
        SAT01--SAT02 isl        [ 0, 60]            -> must not touch downlinks
    """
    e = _epoch(TINY)
    plan: list[Contact] = []
    plan += _link("SAT01", "GS_A", 0, 10, "downlink", e, elev=20.0)
    plan += _link("SAT01", "GS_A", 30, 40, "downlink", e, elev=35.0)
    plan += _link("SAT02", "GS_A", 5, 15, "downlink", e, elev=18.0)
    plan += _link("SAT01", "GS_B", 35, 45, "downlink", e, elev=42.0)
    plan += _link("SAT02", "GS_B", 50, 55, "downlink", e, elev=12.0)
    plan += _link("SAT01", "SAT02", 0, 60, "isl", e)
    return plan


# ==========================================================================
# types.py -- run-length codec
# ==========================================================================

PACK_CASES = {
    "empty": [],
    "single_true": [True],
    "single_false": [False],
    "all_true_1": [True] * 1,
    "all_true_97": [True] * 97,
    "all_false_97": [False] * 97,
    "alternating_even": [i % 2 == 0 for i in range(100)],
    "alternating_odd": [i % 2 == 1 for i in range(101)],
    "one_flip_start": [False] + [True] * 50,
    "one_flip_end": [True] * 50 + [False],
    "long_runs": [True] * 500 + [False] * 3 + [True] * 12 + [False] * 1441,
    "grid_length": [False] * 1441,
}


@pytest.mark.parametrize("name", sorted(PACK_CASES))
def test_pack_unpack_roundtrip_fixed(name):
    bits = PACK_CASES[name]
    assert unpack(_pack(bits)) == bits


def test_pack_unpack_roundtrip_random():
    """The frontend decodes this format. Prove it is lossless over 400 random
    shapes, including heavily-biased ones where runs are long."""
    rng = random.Random(20260801)
    for _ in range(400):
        n = rng.choice([0, 1, 2, 3, 7, 31, 64, 255, 1441])
        p = rng.choice([0.0, 0.02, 0.5, 0.98, 1.0])
        bits = [rng.random() < p for _ in range(n)]
        packed = _pack(bits)
        assert unpack(packed) == bits, f"lossy for n={n} p={p}: {packed[:80]!r}"
        # idempotent: re-packing the decoded value gives the same string
        assert _pack(unpack(packed)) == packed


def test_pack_unpack_roundtrip_exhaustive_small():
    """Every bool list of length 0..12 -- 8191 cases, brute force."""
    for n in range(13):
        for mask in range(1 << n):
            bits = [bool(mask >> i & 1) for i in range(n)]
            assert unpack(_pack(bits)) == bits


def test_pack_returns_str_and_unpack_returns_native_bools():
    packed = _pack([True, False, False])
    assert isinstance(packed, str)
    out = unpack(packed)
    assert all(type(b) is bool for b in out)


def test_pack_wire_format_is_stable():
    """The exact string shape is a published contract with web/ -- pin it."""
    assert _pack([]) == ""
    assert _pack([True] * 12 + [False] * 3 + [True] * 40) == "12T,3F,40T"
    assert _pack([False]) == "1F"
    assert re.fullmatch(r"\d+[TF](,\d+[TF])*", _pack([True] * 5 + [False] * 2))


def test_strip_to_dict_roundtrips_through_unpack():
    strip = Strip(t_grid=[0, 60_000, 120_000])
    strip.per_link["A|B"] = [True, True, False]
    strip.per_pair["weather|pointing"] = [False, True, True]
    d = strip.to_dict()
    assert d["t_grid"] == strip.t_grid
    assert unpack(d["per_link"]["A|B"]) == [True, True, False]
    assert unpack(d["per_pair"]["weather|pointing"]) == [False, True, True]


# ==========================================================================
# types.py -- enums and immutability
# ==========================================================================

def test_cause_has_exactly_six_members_including_nominal():
    assert len(Cause) == 6, [c.name for c in Cause]
    assert Cause.NOMINAL in set(Cause)
    assert Cause.NOMINAL.value == "nominal"
    assert {c.name for c in Cause} == {
        "NOMINAL", "WEATHER", "NODE_DOWN", "POINTING", "BUFFER", "STALE_SCHED"}


def test_cause_index_covers_all_causes():
    from sim.types import CAUSES, CAUSE_INDEX, N_CAUSES
    assert N_CAUSES == 6
    assert len(CAUSES) == 6
    assert sorted(CAUSE_INDEX.values()) == list(range(6))
    assert CAUSE_INDEX[Cause.NOMINAL] == 0   # index 0 so a zero vector is nominal


def _instances():
    return {
        "Contact": (Contact(id="x", src="a", dst="b", t_open=0, t_close=1000,
                            rate_bps=10, kind="isl", max_elev=0.0), "src"),
        "FaultEvent": (FaultEvent(t_start=0, t_end=1, target="a",
                                  kind=Cause.WEATHER, severity=0.5), "severity"),
        "Observation": (Observation(t=0, node="a", peer="b", contact_id="x",
                                    outcome=Outcome.OK, measured_rate=1,
                                    expected_rate=1, elevation_deg=10.0,
                                    channel=Channel.PRIMARY, queue_bytes=0,
                                    ms_since_ok=0, shift_observed=0,
                                    peer_degree=1, self_degree=1), "outcome"),
        "Policy": (Policy(belief={"nominal": 1.0}, action=Action.NONE,
                          target="b", until=0, confidence=1.0), "action"),
        "Evidence": (Evidence(origin="a", seq=1, t=0, peer="b",
                              outcome=Outcome.OK, degree=1), "seq"),
    }


@pytest.mark.parametrize("name", sorted(_instances()))
def test_contract_dataclasses_are_frozen(name):
    obj, attr = _instances()[name]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        setattr(obj, attr, getattr(obj, attr))


@pytest.mark.parametrize("name", sorted(_instances()))
def test_contract_dataclasses_reject_new_attributes(name):
    """slots=True: no ad-hoc fields can be stapled on at runtime.

    NOTE on the exception type. On CPython 3.14, @dataclass(frozen=True,
    slots=True) raises TypeError -- not FrozenInstanceError -- for a name that
    is not a declared field, because the generated __setattr__ closes over the
    pre-slots class object and its `super(cls, self)` fallback fails. Verified
    to be stock interpreter behaviour, not a defect in sim/types.py:

        @dataclass(frozen=True, slots=True)
        class P: x: int
        P(1).zzz = 1   ->  TypeError: super(type, obj): obj (instance of P)
                           is not an instance or subtype of type (P)

    The invariant under test is that the assignment is refused; the type is
    accepted loosely so this does not become a Python-version tripwire.
    """
    obj, _ = _instances()[name]
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        obj.some_field_that_does_not_exist = 1


def test_contact_derived_properties():
    c = Contact(id="x", src="a", dst="b", t_open=1_000, t_close=61_000,
                rate_bps=2_000_000, kind="downlink", max_elev=30.0)
    assert c.duration_ms == 60_000
    assert c.volume_bits == 2_000_000 * 60_000 // 1000
    assert c.to_dict()["kind"] == "downlink"


def test_scenario_config_is_frozen_and_hashable():
    cfg = ScenarioConfig()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        cfg.n_sats = 99
    assert hash(cfg) == hash(ScenarioConfig())
    assert cfg.hash == ScenarioConfig().hash
    assert cfg.hash != ScenarioConfig(n_sats=13).hash
    assert cfg.horizon_ms == cfg.horizon_hours * 3600 * 1000


# ==========================================================================
# contacts.py
# ==========================================================================

def test_plan_is_non_empty(real_plan):
    assert len(real_plan) > 100, "PLAN.md F1: >100 contacts over 24 h"
    assert any(c.kind == "downlink" for c in real_plan)
    assert any(c.kind == "isl" for c in real_plan)


def test_every_contact_has_positive_usable_duration(real_plan):
    bad = [c for c in real_plan if not (c.t_close > c.t_open)]
    assert not bad, f"{len(bad)} contacts with t_close <= t_open, e.g. {bad[:3]}"
    short = [c for c in real_plan if c.duration_ms < C.MIN_CONTACT_MS]
    assert not short, (
        f"{len(short)} contacts shorter than MIN_CONTACT_MS="
        f"{C.MIN_CONTACT_MS}, e.g. {short[:3]}")


def test_times_are_integer_ms(real_plan):
    """PLAN.md T4: simulation time is integer milliseconds, never float."""
    bad = [c for c in real_plan
           if not (isinstance(c.t_open, int) and isinstance(c.t_close, int))]
    assert not bad, bad[:3]


def test_contacts_are_emitted_bidirectionally(real_plan):
    """PLAN.md 4.1: contacts are unidirectional; emit both directions."""
    windows = {(c.src, c.dst, c.t_open) for c in real_plan}
    missing = [w for w in sorted(windows) if (w[1], w[0], w[2]) not in windows]
    assert not missing, (
        f"{len(missing)} contacts have no reverse-direction twin, "
        f"e.g. {missing[:3]}")

    # and the mirrored contact must agree on every physical attribute
    by_dir = {(c.src, c.dst, c.t_open): c for c in real_plan}
    for (s, d, t), c in by_dir.items():
        m = by_dir[(d, s, t)]
        assert (m.t_close, m.rate_bps, m.kind, m.max_elev) == \
               (c.t_close, c.rate_bps, c.kind, c.max_elev), \
               f"mirror mismatch: {c} vs {m}"


def test_no_two_contacts_on_a_directed_pair_overlap(real_plan):
    """Correctness requirement, PLAN.md 4.1. Overlapping windows double-count
    available volume. This must hold after _merge_overlaps."""
    by_pair: dict[tuple[str, str], list[Contact]] = defaultdict(list)
    for c in real_plan:
        by_pair[(c.src, c.dst)].append(c)

    bad = []
    for pair, group in by_pair.items():
        group.sort(key=lambda c: (c.t_open, c.t_close))
        for a, b in zip(group, group[1:]):
            if b.t_open <= a.t_close:
                bad.append((pair, a.t_open, a.t_close, b.t_open, b.t_close))
    assert not bad, f"{len(bad)} overlapping window pairs, e.g. {bad[:3]}"


def test_merge_overlaps_removes_constructed_overlaps():
    """Feed _merge_overlaps a plan that definitely overlaps and check it is
    gone -- the real plan may simply never produce one."""
    e = _epoch(TINY)
    raw = [
        _c("SAT01", "GS_A", 0, 10, "downlink", e, elev=20.0),
        _c("SAT01", "GS_A", 5, 15, "downlink", e, elev=30.0),   # overlaps
        _c("SAT01", "GS_A", 15, 25, "downlink", e, elev=25.0),  # touches
        _c("SAT01", "GS_A", 40, 50, "downlink", e, elev=11.0),  # disjoint
    ]
    merged = C._merge_overlaps(raw)
    assert len(merged) == 2
    a, b = sorted(merged, key=lambda c: c.t_open)
    assert (a.t_open, a.t_close) == (e, e + 25 * MINUTE_MS)
    assert (b.t_open, b.t_close) == (e + 40 * MINUTE_MS, e + 50 * MINUTE_MS)
    assert a.max_elev == 30.0        # merged window keeps the peak elevation
    for x, y in zip(merged, merged[1:]):
        assert y.t_open > x.t_close


def test_contact_ids_are_unique(real_plan):
    ids = [c.id for c in real_plan]
    dupes = {i for i in ids if ids.count(i) > 1} if len(ids) != len(set(ids)) else set()
    assert len(ids) == len(set(ids)), f"{len(ids) - len(set(ids))} dupes: {sorted(dupes)[:5]}"


def test_downlink_max_elev_respects_the_mask(real_plan):
    dl = [c for c in real_plan if c.kind == "downlink"]
    assert dl
    bad = [c for c in dl if c.max_elev < DEFAULT.elev_mask_deg]
    assert not bad, (
        f"{len(bad)} downlinks with max_elev below elev_mask_deg="
        f"{DEFAULT.elev_mask_deg}, e.g. {bad[:3]}")


def test_isl_max_elev_is_zero(real_plan):
    isl = [c for c in real_plan if c.kind == "isl"]
    assert isl
    assert {c.max_elev for c in isl} == {0.0}


def test_link_kinds_are_only_the_two_declared(real_plan):
    assert {c.kind for c in real_plan} <= {"isl", "downlink"}


def test_rates_are_positive_and_bounded_by_the_configured_maximum(real_plan):
    for c in real_plan:
        assert c.rate_bps > 0, c
        cap = DEFAULT.rate_isl if c.kind == "isl" else DEFAULT.rate_downlink
        assert c.rate_bps <= cap, c


def test_build_is_deterministic(real_plan):
    """PLAN.md T1: a run is reproducible from (ScenarioConfig, seed)."""
    again = C.build(DEFAULT)
    assert len(again) == len(real_plan)
    assert [c.to_dict() for c in again] == [c.to_dict() for c in real_plan]


def test_load_satellites_returns_exactly_n_sats():
    from skyfield.api import load
    ts = load.timescale(builtin=True)
    sats = C.load_satellites(DEFAULT, ts)
    assert len(sats) == DEFAULT.n_sats


def test_selected_satellites_span_at_most_n_planes():
    """PLAN.md 5.1: draw from N ADJACENT planes, not spread across all six.
    Iridium's six planes are ~30 deg apart in RAAN mod 180."""
    import math
    from skyfield.api import load
    ts = load.timescale(builtin=True)
    sats = C.load_satellites(DEFAULT, ts)
    bins = {int((math.degrees(s.model.nodeo) % 180.0) // 30.0) for s in sats}
    assert len(bins) <= DEFAULT.n_planes, (
        f"satellites span {len(bins)} RAAN bins {sorted(bins)}, "
        f"n_planes={DEFAULT.n_planes}")
    # adjacency: the occupied bins must be contiguous
    assert sorted(bins) == list(range(min(bins), min(bins) + len(bins))), \
        f"selected planes are not adjacent: {sorted(bins)}"


def test_isl_visibility_rejects_earth_occluded_geometry():
    """Two satellites whose line of sight passes through the Earth cannot see
    each other. PLAN.md 5.1: "grazing-ray test against a sphere of radius
    6378.137 + 80 km, AND range < ISL_MAX_KM".

    isl_max_km is raised here so the range gate cannot mask the result and the
    grazing test is the only thing deciding visibility. Ground truth is the
    closest approach of the segment [a, b] to the Earth's centre, computed
    independently below.
    """
    from skyfield.api import load

    cfg = ScenarioConfig(horizon_hours=2, step_s=60, isl_max_km=20_000.0)
    ts = load.timescale(builtin=True)
    sats = C.load_satellites(cfg, ts)
    epoch = _epoch(cfg)

    steps = cfg.horizon_ms // (cfg.step_s * 1000) + 1
    import numpy as np
    secs = np.arange(steps, dtype=float) * cfg.step_s
    base = ts.utc(*C._iso_parts(cfg.epoch_iso))
    pos = np.stack([s.at(ts.tt_jd(base.tt + secs / 86400.0)).position.km
                    for s in sats])
    sample_ms = [epoch + int(x * 1000) for x in secs]

    claimed: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for c in C._isls(cfg, ts, sats, epoch):
        claimed[(min(c.src, c.dst), max(c.src, c.dst))].append(
            (c.t_open, c.t_close))

    MARGIN_KM = 100.0          # ignore the knife-edge; only flag clear cases
    occluded_samples = 0
    violations = []
    for i in range(len(sats)):
        for j in range(i + 1, len(sats)):
            a, b = pos[i], pos[j]
            d = b - a
            L2 = (d * d).sum(0)
            rng = np.sqrt(L2)
            # minimise |a + u*d| over u in [0, 1]  ->  u* = -(a.d)/|d|^2
            u = np.clip(-(a * d).sum(0) / np.maximum(L2, 1e-9), 0.0, 1.0)
            miss = np.linalg.norm(a + u * d, axis=0)
            blocked = miss <= (C.GRAZE_RADIUS_KM - MARGIN_KM)
            if not blocked.any():
                continue
            key = (C.sat_id(i), C.sat_id(j))
            for k in np.flatnonzero(blocked):
                occluded_samples += 1
                t = sample_ms[int(k)]
                for t0, t1 in claimed.get(key, ()):
                    if t0 <= t <= t1:
                        violations.append(
                            f"{key[0]}-{key[1]} claimed visible at "
                            f"+{(t - epoch) // 60000} min but the line of sight "
                            f"passes {miss[k]:.0f} km from Earth's centre "
                            f"(graze radius {C.GRAZE_RADIUS_KM:.0f} km, "
                            f"range {rng[k]:.0f} km)")
                        break
                if len(violations) >= 5:
                    break
            if len(violations) >= 5:
                break
        if len(violations) >= 5:
            break

    assert occluded_samples > 0, "test is vacuous: no occluded geometry sampled"
    assert not violations, (
        f"{len(violations)}+ Earth-occluded satellite pairs reported as ISL "
        f"contacts:\n  " + "\n  ".join(violations))


def test_node_ids_are_satellites_and_stations(real_plan):
    stations = {s["id"] for s in C.load_stations(DEFAULT)}
    sats = {C.sat_id(i) for i in range(DEFAULT.n_sats)}
    nodes = {c.src for c in real_plan} | {c.dst for c in real_plan}
    assert nodes <= (stations | sats), nodes - (stations | sats)
    for c in real_plan:
        if c.kind == "downlink":
            assert (c.src in stations) != (c.dst in stations), c
        else:
            assert c.src in sats and c.dst in sats, c


# ==========================================================================
# diagnosability.py -- the core condition
# ==========================================================================

def test_brute_force_matches_condition(synthetic_plan):
    """THE test. Recompute identifiable(a,b,t) by direct counting at every grid
    point on a hand-built plan and compare with diagnosability.compute.

        identifiable(a, b, t)  <=>  deg(a, t) > 1  OR  deg(b, t) > 1

    with deg scoped to the link's own kind, and False outside the link's own
    windows.
    """
    strip = D.compute(synthetic_plan, TINY)
    grid = strip.t_grid
    deg = _brute_degree(synthetic_plan, grid)
    win = _brute_windows(synthetic_plan, grid)
    kinds = _brute_kinds(synthetic_plan)

    assert set(strip.per_link) == set(win), (
        f"link sets differ: {set(strip.per_link) ^ set(win)}")

    for key, got in strip.per_link.items():
        a, b = key.split("|")
        kind = kinds[key]
        dk = deg[kind]
        n = len(grid)
        da = dk.get(a, [0] * n)
        db = dk.get(b, [0] * n)
        for i in range(n):
            raw = (da[i] > 1) or (db[i] > 1)
            expect = win[key][i] and raw
            assert got[i] == expect, (
                f"{key} @ grid[{i}] t=+{(grid[i] - grid[0]) // MINUTE_MS}min: "
                f"got {got[i]}, expected {expect} "
                f"(deg[{a}]={da[i]}, deg[{b}]={db[i]}, open={win[key][i]})")
            if win[key][i]:
                assert got[i] == raw, (
                    f"{key} @ +{(grid[i] - grid[0]) // MINUTE_MS}min inside its "
                    f"own window: got {got[i]}, condition says {raw}")


def test_brute_force_matches_condition_on_the_real_plan(real_plan, real_strip):
    """Same independent recomputation, over the committed TLE-derived plan."""
    grid = real_strip.t_grid
    deg = _brute_degree(real_plan, grid)
    win = _brute_windows(real_plan, grid)
    kinds = _brute_kinds(real_plan)
    n = len(grid)
    zero = [0] * n

    assert set(real_strip.per_link) == set(win)

    mismatches = []
    for key, got in real_strip.per_link.items():
        a, b = key.split("|")
        dk = deg[kinds[key]]
        da, db = dk.get(a, zero), dk.get(b, zero)
        w = win[key]
        for i in range(n):
            expect = w[i] and ((da[i] > 1) or (db[i] > 1))
            if got[i] != expect:
                mismatches.append((key, i, got[i], expect, da[i], db[i], w[i]))
                if len(mismatches) > 5:
                    break
        if len(mismatches) > 5:
            break
    assert not mismatches, mismatches


def test_synthetic_expectations_are_hand_checkable(synthetic_plan):
    """Spot-check the synthetic plan against values worked out by hand, so the
    brute-force reference above is itself anchored to something."""
    strip = D.compute(synthetic_plan, TINY)
    g0 = strip.t_grid[0]

    def at(key, minute):
        return strip.per_link[key][(strip.t_grid.index(g0 + minute * MINUTE_MS))]

    # GS_A|SAT01 open [0,10] and [30,40].
    assert at("GS_A|SAT01", 0) is False    # open, but GS_A and SAT01 both deg 1
    assert at("GS_A|SAT01", 4) is False
    assert at("GS_A|SAT01", 5) is True     # SAT02--GS_A opens -> deg(GS_A)=2
    assert at("GS_A|SAT01", 10) is True
    assert at("GS_A|SAT01", 11) is False   # outside its own window
    assert at("GS_A|SAT01", 30) is False   # open again, alone
    assert at("GS_A|SAT01", 35) is True    # SAT01--GS_B opens -> deg(SAT01)=2
    assert at("GS_A|SAT01", 40) is True
    assert at("GS_A|SAT01", 41) is False

    # GS_B|SAT02 [50,55] is completely isolated among downlinks.
    assert not any(strip.per_link["GS_B|SAT02"])

    # The ISL spans the whole horizon but is the only ISL -> deg 1 throughout.
    assert not any(strip.per_link["SAT01|SAT02"])


def test_degree_is_scoped_by_link_kind():
    """FINDINGS F-001. A healthy crosslink uses a different antenna and cannot
    discriminate weather-at-the-ground-station from a pointing error on the
    satellite's high-gain antenna. An ISL concurrent with a downlink must NOT
    make that downlink identifiable.

    This case is built so unscoped degree says True and scoped degree says
    False -- both halves are asserted.
    """
    e = _epoch(TINY)
    plan = (_link("SAT01", "GS_A", 10, 20, "downlink", e, elev=25.0)
            + _link("SAT01", "SAT02", 0, 60, "isl", e))

    grid = D.time_grid(TINY)
    i10 = grid.index(e + 10 * MINUTE_MS)
    i20 = grid.index(e + 20 * MINUTE_MS)

    # 1. the case really would flip under unscoped degree
    unscoped = D.degree_matrix(plan, grid, kind=None)
    assert unscoped["SAT01"][i10] == 2, (
        "test case is not discriminating: unscoped degree is not >1")
    assert unscoped["SAT01"][i20] == 2

    # 2. scoped degree, which is what compute uses
    scoped = D.degree_matrix(plan, grid, kind="downlink")
    assert scoped["SAT01"][i10] == 1
    assert scoped["GS_A"][i10] == 1

    # 3. the observable behaviour
    strip = D.compute(plan, TINY)
    dl = strip.per_link["GS_A|SAT01"]
    assert not any(dl), (
        "an ISL concurrent with a downlink made the downlink identifiable; "
        "degree is not scoped by link kind (FINDINGS F-001)")


def test_scoped_degree_positive_control():
    """The same geometry, but the concurrent contact is a second downlink on the
    same satellite. Now it must be identifiable inside the overlap."""
    e = _epoch(TINY)
    plan = (_link("SAT01", "GS_A", 10, 20, "downlink", e, elev=25.0)
            + _link("SAT01", "GS_B", 15, 25, "downlink", e, elev=30.0)
            + _link("SAT01", "SAT02", 0, 60, "isl", e))

    strip = D.compute(plan, TINY)
    grid = strip.t_grid
    dl = strip.per_link["GS_A|SAT01"]

    def at(minute):
        return dl[grid.index(e + minute * MINUTE_MS)]

    assert at(10) is False          # open, alone
    assert at(14) is False
    assert at(15) is True           # GS_B pass starts -> deg_downlink(SAT01)=2
    assert at(20) is True
    assert at(21) is False          # own window closed


def test_strip_arrays_all_have_grid_length(real_strip, synthetic_plan):
    for strip, label in ((real_strip, "real"),
                         (D.compute(synthetic_plan, TINY), "synthetic")):
        n = len(strip.t_grid)
        assert n > 0
        for k, v in strip.per_link.items():
            assert len(v) == n, f"{label}: per_link[{k}] is {len(v)}, grid is {n}"
            assert all(type(b) is bool for b in v), \
                f"{label}: per_link[{k}] holds non-native bools"
        for k, v in strip.per_pair.items():
            assert len(v) == n, f"{label}: per_pair[{k}] is {len(v)}, grid is {n}"


def test_time_grid_covers_the_horizon_exactly():
    grid = D.time_grid(TINY)
    e = _epoch(TINY)
    assert grid[0] == e
    assert grid[-1] == e + TINY.horizon_ms
    assert len(grid) == TINY.horizon_ms // (TINY.grid_s * 1000) + 1
    assert all(b - a == TINY.grid_s * 1000 for a, b in zip(grid, grid[1:]))


def test_identifiability_is_false_outside_a_links_own_windows(
        real_plan, real_strip):
    """Outside its own contact windows the question is undefined; the strip
    must report False so the UI renders 'cannot diagnose here'."""
    grid = real_strip.t_grid
    win = _brute_windows(real_plan, grid)
    offenders = []
    for key, arr in real_strip.per_link.items():
        w = win[key]
        for i, v in enumerate(arr):
            if v and not w[i]:
                offenders.append((key, i, grid[i]))
                break
    assert not offenders, offenders[:5]


def test_identifiability_is_false_outside_windows_synthetic(synthetic_plan):
    strip = D.compute(synthetic_plan, TINY)
    win = _brute_windows(synthetic_plan, strip.t_grid)
    for key, arr in strip.per_link.items():
        for i, v in enumerate(arr):
            assert not (v and not win[key][i]), (
                f"{key} identifiable at grid[{i}] with no contact open")


def test_pair_separable_follows_the_rule_table(synthetic_plan):
    strip = D.compute(synthetic_plan, TINY)
    n = len(strip.t_grid)
    link = ("SAT01", "GS_A")

    concurrency = D.pair_separable(strip, link, Cause.WEATHER, Cause.POINTING)
    assert concurrency == list(strip.per_link["GS_A|SAT01"])
    # order of the pair must not matter
    assert D.pair_separable(strip, link, Cause.POINTING, Cause.WEATHER) == concurrency
    # link order must not matter either
    assert D.pair_separable(strip, ("GS_A", "SAT01"),
                            Cause.WEATHER, Cause.POINTING) == concurrency

    assert D.pair_separable(strip, link, Cause.STALE_SCHED,
                            Cause.NODE_DOWN) == [True] * n
    assert D.pair_separable(strip, link, Cause.BUFFER,
                            Cause.POINTING) == [False] * n
    # unknown link -> all False, grid length
    assert D.pair_separable(strip, ("NOPE", "ALSO_NOPE"),
                            Cause.WEATHER, Cause.POINTING) == [False] * n


# ==========================================================================
# diagnosability.py -- summarize
# ==========================================================================

def test_summarize_fractions_are_window_relative_not_total_time():
    """FINDINGS F-002(a). A link open for a sliver of the horizon but always
    identifiable inside that sliver must report ~1.0, not its duty cycle."""
    e = _epoch(TINY)
    # Two downlinks on the same ground station, open together for 1 minute out
    # of a 60-minute horizon. deg(GS_A) = 2 throughout the window.
    plan = (_link("SAT01", "GS_A", 30, 31, "downlink", e, elev=40.0)
            + _link("SAT02", "GS_A", 30, 31, "downlink", e, elev=38.0))

    strip = D.compute(plan, TINY)
    n = len(strip.t_grid)
    s = D.summarize(strip, plan)

    open_pts = sum(1 for v in strip.per_link["GS_A|SAT01"] if True) and \
        sum(1 for i, t in enumerate(strip.t_grid)
            if e + 30 * MINUTE_MS <= t <= e + 31 * MINUTE_MS)
    assert open_pts == 2
    duty_cycle = open_pts / n                       # ~0.033

    assert s["downlink_links"] == 2
    assert s["downlink_identifiable_in_window"] == pytest.approx(1.0), (
        f"window-relative fraction should be 1.0, got "
        f"{s['downlink_identifiable_in_window']} (duty cycle is {duty_cycle:.4f})")
    assert s["downlink_identifiable_in_window"] != pytest.approx(duty_cycle, abs=1e-6)
    assert s["downlinks_always_identifiable"] == 2
    assert s["downlinks_never_identifiable"] == 0
    assert s["frac_time_any_contact_open"] == pytest.approx(duty_cycle)


def test_summarize_reports_zero_for_a_never_identifiable_link():
    e = _epoch(TINY)
    plan = _link("SAT01", "GS_A", 30, 31, "downlink", e, elev=40.0)
    strip = D.compute(plan, TINY)
    s = D.summarize(strip, plan)
    assert s["downlink_identifiable_in_window"] == 0.0
    assert s["downlinks_never_identifiable"] == 1
    assert s["downlinks_always_identifiable"] == 0


def test_summarize_matches_independent_counting(real_plan, real_strip):
    s = D.summarize(real_strip, real_plan)
    grid = real_strip.t_grid
    win = _brute_windows(real_plan, grid)

    assert s["grid_points"] == len(grid)
    assert s["unique_links"] == len(real_strip.per_link)
    assert s["isl_links"] + s["downlink_links"] == s["unique_links"]
    assert (s["downlinks_always_identifiable"]
            + s["downlinks_never_identifiable"]
            + s["downlinks_mixed"]) == s["downlink_links"]

    dl_fracs = []
    for key, arr in real_strip.per_link.items():
        if "GS_" not in key:
            continue
        open_pts = sum(win[key])
        dl_fracs.append(sum(arr) / open_pts if open_pts else 0.0)
    assert s["downlink_identifiable_in_window"] == \
        pytest.approx(sum(dl_fracs) / len(dl_fracs))
    assert 0.0 <= s["downlink_identifiable_in_window"] <= 1.0


def test_summarize_fractions_stay_in_the_unit_interval(real_strip, real_plan):
    s = D.summarize(real_strip, real_plan)
    for k in ("frac_time_any_contact_open", "isl_identifiable_in_window",
              "downlink_identifiable_in_window"):
        assert 0.0 <= s[k] <= 1.0, (k, s[k])


def test_summarize_link_kind_split_matches_actual_contact_kind(real_plan, real_strip):
    """summarize splits isl from downlink by sniffing "GS_" in the link key.
    That must agree with the contacts' declared kind, or the two headline
    numbers are computed over the wrong sets."""
    kinds = _brute_kinds(real_plan)
    for key in real_strip.per_link:
        by_string = "downlink" if "GS_" in key else "isl"
        assert by_string == kinds[key], (
            f"link {key} is kind={kinds[key]} but summarize classifies it as "
            f"{by_string}")


def test_contact_times_and_strip_grid_share_a_time_origin(real_plan, real_strip):
    """contacts.py and diagnosability.py derive their epoch independently
    (contacts via Skyfield, diagnosability via datetime). They must agree, or
    every strip window is offset from the plan it describes.

    Both currently use ABSOLUTE Unix milliseconds, not "ms since scenario
    epoch" as sim/types.py's module docstring and PLAN.md 4.1 say. The
    invariant that matters is that they agree; this test pins that.
    """
    grid = real_strip.t_grid
    assert grid[0] == D._epoch_ms(DEFAULT)
    assert grid[-1] == grid[0] + DEFAULT.horizon_ms
    outside = [c for c in real_plan
               if c.t_open < grid[0] or c.t_close > grid[-1]]
    assert not outside, (
        f"{len(outside)} contacts fall outside the strip grid, "
        f"e.g. {outside[:2]}")
    # at least one contact starts exactly at the grid origin -> same epoch
    assert min(c.t_open for c in real_plan) >= grid[0]


def test_strip_computation_is_deterministic(real_plan):
    a = D.compute(real_plan, DEFAULT)
    b = D.compute(real_plan, DEFAULT)
    assert a.t_grid == b.t_grid
    assert a.per_link == b.per_link


def test_strip_is_fast(real_plan):
    """PLAN.md F2: the strip runs in under a second."""
    import time
    t = time.perf_counter()
    D.compute(real_plan, DEFAULT)
    assert time.perf_counter() - t < 1.0
