"""Diagnosability. PLAN.md section 5.2.

Computed from the contact plan alone -- no simulator, no faults, no inference.
This module is the reason the project has a result even if everything
downstream fails, and it is also the instrument used to tune the topology.

The argument
------------
Two causes are distinguishable only if they predict different observations.
Weather at ground station g degrades every contact touching g. A pointing
error on satellite s degrades every contact touching s. If, during the fault
window, s has exactly one active contact and it is with g, AND g has exactly
one active contact and it is with s, then both hypotheses predict the
*identical* observation set:

    p(weather@g | o) / p(pointing@s | o)  =  p(weather@g) / p(pointing@s)

The posterior ratio equals the prior ratio for all t and all observations. No
estimator separates them -- not a Bayes filter, not a neural net, not an LLM.
This is ordinary statistical unidentifiability, not a modelling weakness.

    identifiable(s, g, t)  <=>  deg(g, t) > 1  OR  deg(s, t) > 1

Sparsity is what makes this bite. Contacts are scheduled and sparse, so degree
1 is the common case, not the exception.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .config import ScenarioConfig, DEFAULT
from .types import Contact, Strip, Cause

# Cause pairs and what it takes to separate them. Concurrency is necessary for
# the spatial-clique pairs; the others need a different observable entirely.
PAIR_RULES: dict[tuple[Cause, Cause], str] = {
    (Cause.WEATHER, Cause.POINTING): "concurrency",
    (Cause.WEATHER, Cause.NODE_DOWN): "concurrency",
    (Cause.POINTING, Cause.NODE_DOWN): "concurrency",
    (Cause.BUFFER, Cause.POINTING): "traffic_variation",
    (Cause.BUFFER, Cause.WEATHER): "traffic_variation",
    (Cause.STALE_SCHED, Cause.NODE_DOWN): "out_of_window_listening",
    (Cause.STALE_SCHED, Cause.WEATHER): "out_of_window_listening",
}


def time_grid(cfg: ScenarioConfig) -> list[int]:
    epoch_ms = _epoch_ms(cfg)
    step = cfg.grid_s * 1000
    return list(range(epoch_ms, epoch_ms + cfg.horizon_ms + step, step))


def _epoch_ms(cfg: ScenarioConfig) -> int:
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(cfg.epoch_iso.replace("Z", "+00:00"))
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def degree_matrix(contacts: list[Contact], grid: list[int], kind: str | None = None):
    """node -> np.int16 array of concurrent-contact count at each grid point.

    Counts undirected links: a bidirectional pair contributes 1, not 2.

    `kind` filters to one link type, and filtering is REQUIRED for correctness --
    see the note on channel scope in `compute`.
    """
    g = np.asarray(grid)
    deg: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(g), dtype=np.int16))

    seen: set[tuple[str, str, int, int]] = set()
    for c in contacts:
        if kind is not None and c.kind != kind:
            continue
        key = (min(c.src, c.dst), max(c.src, c.dst), c.t_open, c.t_close)
        if key in seen:
            continue
        seen.add(key)
        active = (g >= c.t_open) & (g <= c.t_close)
        deg[c.src] = deg[c.src] + active
        deg[c.dst] = deg[c.dst] + active
    return dict(deg)


def compute(contacts: list[Contact], cfg: ScenarioConfig = DEFAULT) -> Strip:
    """Per-link identifiability over the grid.

    CHANNEL SCOPE (see FINDINGS F-001). Concurrency only discriminates when the
    concurrent contact shares the failing antenna. A healthy inter-satellite
    link tells you nothing about whether a failed downlink is weather at the
    ground station or a pointing error on the satellite's high-gain antenna --
    the crosslink uses different hardware and would survive either. So degree
    is computed WITHIN link kind, not across all contacts.

    Counting all contacts made every downlink look identifiable, because each
    satellite carries 2-3 crosslinks continuously. That was wrong and it
    erased the entire structure the strip exists to show.
    """
    grid = time_grid(cfg)
    g = np.asarray(grid)
    zeros = np.zeros(len(g), dtype=np.int16)

    deg_by_kind = {
        "downlink": degree_matrix(contacts, grid, kind="downlink"),
        "isl": degree_matrix(contacts, grid, kind="isl"),
    }

    strip = Strip(t_grid=grid)

    # Group windows by undirected link in one pass rather than rescanning the
    # full contact list per link (that was O(links * contacts)).
    windows: dict[tuple[str, str], np.ndarray] = {}
    kinds: dict[tuple[str, str], str] = {}
    for c in contacts:
        key = (min(c.src, c.dst), max(c.src, c.dst))
        if key not in windows:
            windows[key] = np.zeros(len(g), dtype=bool)
            kinds[key] = c.kind
        windows[key] |= (g >= c.t_open) & (g <= c.t_close)

    for (a, b), win in windows.items():
        deg = deg_by_kind[kinds[(a, b)]]
        concurrent = (deg.get(a, zeros) > 1) | (deg.get(b, zeros) > 1)
        # Outside the link's own windows the question is undefined; report
        # False so the UI renders "cannot diagnose here", which is the
        # operationally honest reading.
        strip.per_link[f"{a}|{b}"] = (win & concurrent).tolist()

    # per_pair is deliberately NOT filled here (see FINDINGS F-002). Separability
    # for a cause pair is a property of the link that is actually faulted, so a
    # network-wide "is anything identifiable right now" aggregate is always
    # ~100% true and renders as a solid green bar carrying no information. The
    # engine populates per_pair for the faulted link once faults are known.
    return strip


def pair_separable(strip: Strip, link: tuple[str, str],
                   c1: Cause, c2: Cause) -> list[bool]:
    """Separability of one cause pair on one specific link, over the grid.

    This is what the UI strip should render: the faulted link, not the network.
    """
    a, b = sorted(link)
    geom = strip.per_link.get(f"{a}|{b}")
    if geom is None:
        return [False] * len(strip.t_grid)

    rule = PAIR_RULES.get((c1, c2)) or PAIR_RULES.get((c2, c1))
    if rule == "concurrency":
        return list(geom)
    if rule == "out_of_window_listening":
        return [True] * len(strip.t_grid)
    # traffic_variation: needs the offered-load trace, which geometry does not
    # have. Conservative default; the engine overwrites it.
    return [False] * len(strip.t_grid)


def summarize(strip: Strip, contacts: list[Contact]) -> dict:
    """Topology-tuning readout. This is the instrument -- run it over candidate
    configurations and pick the one where the strip has real structure AND
    gossip has somewhere to go.

    Every fraction below is WINDOW-RELATIVE: identifiable grid points divided by
    grid points where that link is actually open. Dividing by total grid points
    instead just measures duty cycle -- a downlink open 4% of the day reports
    ~0.04 no matter how identifiable it is. (FINDINGS F-002.)
    """
    n = len(strip.t_grid)
    grid = np.asarray(strip.t_grid)
    per_link = {k: np.asarray(v, dtype=bool) for k, v in strip.per_link.items()}

    win: dict[str, np.ndarray] = {}
    for c in contacts:
        k = f"{min(c.src, c.dst)}|{max(c.src, c.dst)}"
        m = (grid >= c.t_open) & (grid <= c.t_close)
        win[k] = m if k not in win else (win[k] | m)

    frac: dict[str, float] = {}
    for k, arr in per_link.items():
        w = win.get(k)
        open_pts = int(w.sum()) if w is not None else 0
        frac[k] = float(arr.sum() / open_pts) if open_pts else 0.0

    isl = [k for k in per_link if "GS_" not in k]
    dl = [k for k in per_link if "GS_" in k]

    any_open = np.zeros(n, dtype=bool)
    for w in win.values():
        any_open |= w

    def mean(keys):
        return float(np.mean([frac[k] for k in keys])) if keys else 0.0

    dl_vals = [frac[k] for k in dl]
    return {
        "grid_points": n,
        "unique_links": len(per_link),
        "isl_links": len(isl),
        "downlink_links": len(dl),
        "frac_time_any_contact_open": float(any_open.mean()),
        "isl_identifiable_in_window": mean(isl),
        "downlink_identifiable_in_window": mean(dl),
        # Structure check: a strip that is all-green or all-red is useless.
        # We want downlinks somewhere in the middle.
        "downlinks_always_identifiable": sum(1 for v in dl_vals if v > 0.99),
        "downlinks_never_identifiable": sum(1 for v in dl_vals if v < 0.01),
        "downlinks_mixed": sum(1 for v in dl_vals if 0.01 <= v <= 0.99),
    }


def write(strip: Strip, cfg: ScenarioConfig = DEFAULT) -> str:
    path = cfg.out_path("strip.json")
    path.write_text(json.dumps(strip.to_dict(), separators=(",", ":")), encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    import time
    from . import contacts as C

    cs = C.read()
    t = time.time()
    s = compute(cs)
    p = write(s)
    dt = time.time() - t
    summary = summarize(s, cs)
    print(f"strip computed in {dt:.2f}s -> {p}")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
