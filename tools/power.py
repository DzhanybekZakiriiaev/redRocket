"""Statistical power for the identifiability result.

The S3 headline -- accuracy 0.778 inside identifiable windows against 0.577
outside -- rests on n=9 and n=104 from a single seed. This runs the same
measurement over many seeds and puts intervals on it.

Three things it does that the single-run version did not:

1. **Clusters the bootstrap by seed.** Belief samples arrive every 5 minutes
   and a pointing fault runs for hours, so consecutive samples of the same
   episode are the same fact counted forty times. A per-sample bootstrap
   would report an interval several times too narrow. The seed is the
   independent replication unit, so that is what gets resampled.

2. **Splits the unidentifiable bin.** `strip.per_link` is `window AND
   concurrent`, and `diagnosability.compute` deliberately reports False
   outside the link's own contact windows because there the question is
   undefined. Downlink duty cycle is ~21%, so binning on that value alone
   puts every no-contact-open sample into "unidentifiable" -- mixing
   "geometry forbids diagnosis" with "nothing was observed". They are
   separated here into `deg1` (link open, degree 1 -- genuinely
   unidentifiable) and `closed` (no contact open).

3. **Reports the majority-class baseline.** Accuracy is only interpretable
   against the base rate of the bin it was measured in. Pointing faults last
   0.5-6 h and weather faults 1-30 min, so fault-active samples are
   overwhelmingly pointing and the "0.5 chance line for a binary pair" is not
   the right reference.

Usage:
    python tools/power.py                       # main table, 200 seeds
    python tools/power.py --seeds 400
    python tools/power.py --faults-sweep
    python tools/power.py --equalise            # P_STAY leak test
    python tools/power.py --all
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import contacts as C, faults as F, diagnosability as D   # noqa: E402
from sim.config import DEFAULT, ScenarioConfig                    # noqa: E402
from sim.engine import Engine                                     # noqa: E402
from sim.advisor import bayes as BAYES                            # noqa: E402
from sim.advisor.bayes import BayesAdvisor                        # noqa: E402
from sim.types import Cause                                       # noqa: E402

BINS = ("ident", "deg1", "closed")
BIN_LABEL = {
    "ident": "identifiable   (link open, degree>1)",
    "deg1": "unidentifiable (link open, degree 1)",
    "closed": "no contact open (strip undefined)",
}
PAIR = ("weather", "pointing")


# --------------------------------------------------------------------------
# geometry: identifiability and window masks, computed once
# --------------------------------------------------------------------------

class Geometry:
    def __init__(self, contacts, cfg: ScenarioConfig = DEFAULT):
        strip = D.compute(contacts, cfg)
        self.grid0 = strip.t_grid[0]
        self.step = cfg.grid_s * 1000
        self.n = len(strip.t_grid)
        g = np.asarray(strip.t_grid)

        self.ident = {k: np.asarray(v, dtype=bool) for k, v in strip.per_link.items()}
        self.open_: dict[str, np.ndarray] = {}
        for c in contacts:
            k = f"{min(c.src, c.dst)}|{max(c.src, c.dst)}"
            m = (g >= c.t_open) & (g <= c.t_close)
            self.open_[k] = m if k not in self.open_ else (self.open_[k] | m)

        # Exact-time version of the same condition. Belief samples land on the
        # grid, but OBSERVATIONS land at contact-open instants, which almost
        # never coincide with a grid point -- flooring to the grid would call
        # every observation "no contact open".
        pair_iv: dict[str, list] = defaultdict(list)
        node_iv: dict[tuple[str, str], list] = defaultdict(list)
        seen = set()
        for c in contacts:
            k = f"{min(c.src, c.dst)}|{max(c.src, c.dst)}"
            pair_iv[k].append((c.t_open, c.t_close))
            key = (k, c.t_open, c.t_close)
            if key in seen:               # undirected: a pair counts once
                continue
            seen.add(key)
            node_iv[(c.src, c.kind)].append((c.t_open, c.t_close))
            node_iv[(c.dst, c.kind)].append((c.t_open, c.t_close))
        self.pair_iv = {k: (np.array([a for a, _ in v]), np.array([b for _, b in v]))
                        for k, v in pair_iv.items()}
        self.node_iv = {k: (np.array([a for a, _ in v]), np.array([b for _, b in v]))
                        for k, v in node_iv.items()}

    def _deg(self, node: str, kind: str, t: int) -> int:
        iv = self.node_iv.get((node, kind))
        if iv is None:
            return 0
        return int(((iv[0] <= t) & (t <= iv[1])).sum())

    def exact_bin(self, node: str, peer: str, t: int) -> tuple[str, str]:
        k = f"{min(node, peer)}|{max(node, peer)}"
        kind = "downlink" if "GS_" in k else "isl"
        iv = self.pair_iv.get(k)
        if iv is None or not ((iv[0] <= t) & (t <= iv[1])).any():
            return "closed", kind
        ident = self._deg(node, kind, t) > 1 or self._deg(peer, kind, t) > 1
        return ("ident" if ident else "deg1"), kind

    def bin_of(self, node: str, target: str, t: int) -> tuple[str, str] | None:
        """Link-scoped, per PLAN C-003: the faulted link, never an aggregate.

        Returns (bin, link_kind). The kind matters: an ISL is open almost
        continuously and every satellite carries 2-3 of them, so ISL degree is
        >1 essentially always and every ISL sample lands in `ident` by
        construction. Pooling kinds turns the identifiability contrast into a
        downlink-vs-crosslink contrast wearing its clothes.
        """
        if not target:
            return None
        k = f"{min(node, target)}|{max(node, target)}"
        kind = "downlink" if "GS_" in k else "isl"
        idx = (t - self.grid0) // self.step
        if idx < 0 or idx >= self.n:
            return None
        op = self.open_.get(k)
        if op is None or not op[idx]:
            return "closed", kind
        return ("ident" if self.ident[k][idx] else "deg1"), kind


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------

def run_seed(contacts, geo: Geometry, seed: int, n_faults: int,
             kinds=None, cfg_base: ScenarioConfig = DEFAULT) -> list[dict]:
    cfg = replace(cfg_base, seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):     # generator's under-fill warning
        faults = F.generate(contacts, cfg, n_faults=n_faults, kinds=kinds)
    eng = Engine(contacts, faults, BayesAdvisor(grid_ms=cfg.grid_s * 1000), cfg)
    eng.run()

    rows = []
    for b in eng.beliefs:
        if b["truth"] == "nominal":
            continue
        z = geo.bin_of(b["node"], b["target"], b["t"])
        if z is None:
            continue
        rows.append({"seed": seed, "bin": z[0], "kind": z[1], "truth": b["truth"],
                     "pred": b["argmax"], "hit": b["argmax"] == b["truth"]})
    return rows


def mean_placed(contacts, seeds, n_faults, kinds=None) -> float:
    """`generate` enforces one active fault per target and can genuinely run
    out of room, so the requested count is not the delivered count."""
    tot = 0
    for s in seeds:
        cfg = replace(DEFAULT, seed=s)
        with contextlib.redirect_stdout(io.StringIO()):
            tot += len(F.generate(contacts, cfg, n_faults=n_faults, kinds=kinds))
    return tot / max(len(seeds), 1)


def collect(contacts, geo, seeds, n_faults, kinds=None, cfg_base=DEFAULT,
            progress=True) -> list[dict]:
    rows = []
    t0 = time.time()
    for i, s in enumerate(seeds):
        rows += run_seed(contacts, geo, s, n_faults, kinds, cfg_base)
        if progress and (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{len(seeds)} seeds, {time.time()-t0:.0f}s",
                  file=sys.stderr)
    return rows


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def keep(r, bin_, subset, kind) -> bool:
    if r["bin"] != bin_:
        return False
    if kind and r["kind"] != kind:
        return False
    if subset == "pair" and r["truth"] not in PAIR:
        return False
    return True


def per_seed_counts(rows, seeds, bin_: str, subset: str, kind=None) -> np.ndarray:
    """(S, 2) array of [hits, n] per seed."""
    acc = {s: [0, 0] for s in seeds}
    for r in rows:
        if not keep(r, bin_, subset, kind):
            continue
        acc[r["seed"]][0] += r["hit"]
        acc[r["seed"]][1] += 1
    return np.array([acc[s] for s in seeds], dtype=float)


def _boot_idx(S: int, B: int, rng) -> np.ndarray:
    return rng.integers(0, S, size=(B, S))


def cluster_ci(arr: np.ndarray, B: int = 4000, rng=None):
    """Seed-clustered percentile bootstrap. Returns (point, lo, hi, n)."""
    n_tot = arr[:, 1].sum()
    if n_tot == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = arr[:, 0].sum() / n_tot
    rng = rng or np.random.default_rng(0)
    idx = _boot_idx(len(arr), B, rng)
    h = arr[idx, 0].sum(1)
    n = arr[idx, 1].sum(1)
    ok = n > 0
    if ok.sum() < 100:
        return point, float("nan"), float("nan"), int(n_tot)
    a = h[ok] / n[ok]
    return point, float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), int(n_tot)


def cluster_ci_diff(a: np.ndarray, b: np.ndarray, B: int = 4000, rng=None):
    """CI on acc(a) - acc(b), resampling the SAME seeds for both bins."""
    if a[:, 1].sum() == 0 or b[:, 1].sum() == 0:
        return float("nan"), float("nan"), float("nan")
    point = a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum()
    rng = rng or np.random.default_rng(1)
    idx = _boot_idx(len(a), B, rng)
    ha, na = a[idx, 0].sum(1), a[idx, 1].sum(1)
    hb, nb = b[idx, 0].sum(1), b[idx, 1].sum(1)
    ok = (na > 0) & (nb > 0)
    if ok.sum() < 100:
        return point, float("nan"), float("nan")
    d = ha[ok] / na[ok] - hb[ok] / nb[ok]
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def wilson(h: int, n: int, z: float = 1.96):
    """Per-sample binomial interval. Reported only to show how much narrower
    it is than the clustered one -- it is the WRONG interval here."""
    if n == 0:
        return float("nan"), float("nan")
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - hw, c + hw


def baselines(rows, bin_: str, subset: str, kind=None) -> tuple[float, float, int]:
    """(majority-class rate, prior-matched random rate, n) inside a bin."""
    truths = [r["truth"] for r in rows if keep(r, bin_, subset, kind)]
    if not truths:
        return float("nan"), float("nan"), 0
    c = Counter(truths)
    n = len(truths)
    maj = max(c.values()) / n
    rand = sum((v / n) ** 2 for v in c.values())
    return maj, rand, n


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def report_table(rows, seeds, title: str, boot: int = 4000, kind=None) -> dict:
    print()
    print(title)
    print(f"  {'bin':<38}{'subset':<7}{'n':>7}{'acc':>7}"
          f"{'95% CI (seed-clustered)':>26}{'maj':>7}{'rand':>7}{'seeds>0':>9}")
    rng = np.random.default_rng(7)
    out = {}
    for b in BINS:
        for subset in ("pair", "all"):
            arr = per_seed_counts(rows, seeds, b, subset, kind)
            p, lo, hi, n = cluster_ci(arr, boot, rng)
            maj, rand, _ = baselines(rows, b, subset, kind)
            nz = int((arr[:, 1] > 0).sum())
            out[(b, subset)] = (arr, p, lo, hi, n)
            ci = f"[{lo:.3f}, {hi:.3f}]" if n else "--"
            print(f"  {BIN_LABEL[b]:<38}{subset:<7}{n:>7,}{p:>7.3f}"
                  f"{ci:>26}{maj:>7.3f}{rand:>7.3f}{nz:>9}")
    return out


def report_contrast(out, boot=4000) -> dict:
    print()
    print("  contrast -- the headline claim is that the first number exceeds the second")
    rng = np.random.default_rng(11)
    res = {}
    for subset in ("pair", "all"):
        for lo_bin, name in (("deg1", "ident - deg1  (open, degree 1)"),
                             ("closed", "ident - closed (no contact)")):
            a = out[("ident", subset)][0]
            b = out[(lo_bin, subset)][0]
            d, l, h = cluster_ci_diff(a, b, boot, rng)
            verdict = "supported" if (l == l and l > 0) else "NOT supported"
            print(f"    {subset:<5} {name:<32} {d:+.3f}  95% CI "
                  f"[{l:+.3f}, {h:+.3f}]  {verdict}")
            res[(subset, lo_bin)] = (d, l, h)
    return res


def report_confusion(rows, bins=BINS, kind=None) -> None:
    print()
    print("  where the weather/pointing decisions actually go "
          "(rows = truth, cols = argmax):")
    for b in bins:
        sub = [r for r in rows if keep(r, b, "pair", kind)]
        if not sub:
            continue
        preds = sorted({r["pred"] for r in sub})
        print(f"    {b:<8} " + "".join(f"{p:>12}" for p in preds) + f"{'n':>8}")
        for t in PAIR:
            row = [r for r in sub if r["truth"] == t]
            if not row:
                continue
            c = Counter(r["pred"] for r in row)
            print(f"      {t:<10}" + "".join(f"{c.get(p,0):>12}" for p in preds)
                  + f"{len(row):>8}")


def report_power(out, diffs, n_seeds: int, n_faults: int) -> None:
    print()
    print("  power. Two criteria: (a) the identifiable-bin CI narrower than the")
    print("  observed gap, and (b) the stronger and actually correct one -- the CI")
    print("  on the DIFFERENCE excluding zero. Both scale as 1/sqrt(seeds).")
    for subset in ("pair", "all"):
        arr, p, lo, hi, n = out[("ident", subset)]
        if n == 0 or hi != hi:
            print(f"    {subset}: no samples")
            continue
        hw = (hi - lo) / 2
        wl, wh = wilson(int(arr[:, 0].sum()), int(arr[:, 1].sum()))
        d, dlo, dhi = diffs[(subset, "deg1")]
        dhw = (dhi - dlo) / 2
        need_a = n_seeds * (hw / abs(d)) ** 2 if d else float("inf")
        need_b = n_seeds * (dhw / abs(d)) ** 2 if d else float("inf")
        print(f"    {subset:<5} ident n={n:>6,}  bin half-width {hw:.3f} at "
              f"{n_seeds} seeds x {n_faults} faults")
        print(f"          per-sample Wilson would claim {(wh-wl)/2:.3f} -- "
              f"{hw/max((wh-wl)/2,1e-9):.1f}x too narrow; belief samples inside one "
              f"fault episode are not independent")
        print(f"          gap {d:+.3f}  (a) needs ~{need_a:,.0f} seeds   "
              f"(b) needs ~{need_b:,.0f} seeds "
              f"-> ~{n/max(n_seeds,1)*need_b:,.0f} identifiable samples")


# --------------------------------------------------------------------------
# P_STAY equalisation -- how much of the deg1 excess is dwell-time information
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# feature availability -- does the evidence the geometry permits ever arrive?
# --------------------------------------------------------------------------

class FeatureSpy(BayesAdvisor):
    """Records which discriminating features were actually available.

    `identifiable(s, g, t)` is a statement about what evidence EXISTS in the
    world. The filter only benefits if that evidence reaches it: the
    concurrent contact has to have been observed within `_self_cofail`'s
    15-minute window, or a peer's report has to have arrived by gossip. This
    measures how often either is true at the instant of a fault-active
    observation.
    """

    def __init__(self, geo: Geometry, faults, **kw):
        super().__init__(**kw)
        self.geo = geo
        self.faults = faults
        self.log: list[dict] = []

    def _truth(self, node, peer, t) -> str:
        for f in self.faults:
            if f.t_start <= t <= f.t_end and f.target in (node, peer):
                return f.kind.value
        return "nominal"

    def _likelihood(self, obs):
        z = self.geo.exact_bin(obs.node, obs.peer, obs.t)
        self.log.append({
            "bin": z[0], "kind": z[1],
            "truth": self._truth(obs.node, obs.peer, obs.t),
            "self_cf": self._self_cofail(obs) is not None,
            "peer_cf": self._peer_cofail(obs) is not None,
        })
        return super()._likelihood(obs)


def report_features(contacts, geo, seeds, n_faults, cfg_base=DEFAULT) -> None:
    print()
    print("FEATURE AVAILABILITY  -- the evidence the strip says exists, "
          "vs the evidence the filter got")
    log = []
    for s in seeds:
        cfg = replace(cfg_base, seed=s)
        with contextlib.redirect_stdout(io.StringIO()):
            faults = F.generate(contacts, cfg, n_faults=n_faults)
        adv = FeatureSpy(geo, faults, grid_ms=cfg.grid_s * 1000)
        Engine(contacts, faults, adv, cfg).run()
        log += adv.log
    print(f"  {'bin':<10}{'obs':>9}{'self_cofail':>13}{'peer_cofail':>13}{'either':>9}")
    for b in BINS:
        for tag, filt in (("", lambda r: r["truth"] != "nominal"),
                          (" (wx)", lambda r: r["truth"] == "weather")):
            sub = [r for r in log if r["bin"] == b and r["kind"] == "downlink"
                   and filt(r)]
            if not sub:
                continue
            n = len(sub)
            sc = sum(r["self_cf"] for r in sub)
            pc = sum(r["peer_cf"] for r in sub)
            ei = sum(r["self_cf"] or r["peer_cf"] for r in sub)
            print(f"  {b+tag:<10}{n:>9,}{sc/n:>13.3f}{pc/n:>13.3f}{ei/n:>9.3f}")
    print("  (fault-active downlink observations only). If `ident` and `deg1` show")
    print("  similar availability, the bins differ in geometry but not in the")
    print("  information the filter actually holds -- and no accuracy gap can appear.")


@contextlib.contextmanager
def equalised_p_stay(value: float):
    """Force weather and pointing to share a self-transition probability.

    C-009 argues the deg1 excess above chance is legitimate: the kernel encodes
    genuinely different dwell times (weather 0.85, pointing 0.97), so a
    long-running fault really is more likely pointing. If that is the whole
    story, equalising the two must collapse the excess. Whatever survives is
    information reaching the posterior by some other route -- i.e. a leak.

    Mutates module state in memory only; sim/ on disk is untouched.
    """
    old = dict(BAYES.P_STAY)
    old_T = BAYES.T_MATRIX
    BAYES.P_STAY[Cause.WEATHER] = value
    BAYES.P_STAY[Cause.POINTING] = value
    BAYES.T_MATRIX = BAYES._transition()
    try:
        yield
    finally:
        BAYES.P_STAY.clear()
        BAYES.P_STAY.update(old)
        BAYES.T_MATRIX = old_T


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--faults", type=int, default=8)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--faults-sweep", action="store_true")
    ap.add_argument("--weather", action="store_true",
                    help="also run a weather-weighted kinds pool")
    ap.add_argument("--equalise", action="store_true",
                    help="P_STAY leak test")
    ap.add_argument("--features", action="store_true",
                    help="was the discriminating evidence even available?")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        args.faults_sweep = args.weather = args.equalise = args.features = True

    cs = C.read()
    geo = Geometry(cs, DEFAULT)
    seeds = list(range(args.seeds))
    print(f"contact plan {len(cs)} contacts · {len(geo.ident)} links · "
          f"{geo.n} grid points")
    print(f"{args.seeds} seeds x {args.faults} faults, "
          f"link-scoped identifiability, seed-clustered bootstrap B={args.boot}")

    t = time.time()
    rows = collect(cs, geo, seeds, args.faults)
    print(f"collected {len(rows):,} fault-active belief samples in {time.time()-t:.0f}s")

    print()
    print("  truth mix over all fault-active samples: "
          + "  ".join(f"{k}={v:,}" for k, v in
                      Counter(r['truth'] for r in rows).most_common()))
    print(f"  {'bin':<10}{'all':>9}{'downlink':>10}{'isl':>8}{'wx':>6}{'wx dl':>7}")
    for b in BINS:
        sub = [r for r in rows if r["bin"] == b]
        print(f"  {b:<10}{len(sub):>9,}"
              f"{sum(1 for r in sub if r['kind']=='downlink'):>10,}"
              f"{sum(1 for r in sub if r['kind']=='isl'):>8,}"
              f"{sum(1 for r in sub if r['truth']=='weather'):>6,}"
              f"{sum(1 for r in sub if r['truth']=='weather' and r['kind']=='downlink'):>7,}")
    print("  NOTE: an ISL is open ~continuously with 2-3 per satellite, so ISL degree")
    print("  is >1 essentially always and every ISL sample falls in `ident` by")
    print("  construction. Weather cannot even occur on an ISL (it targets stations).")
    print("  The headline table below is therefore DOWNLINK-ONLY -- the links PLAN 5.2")
    print("  actually argues about. The pooled table follows for comparison.")

    out = report_table(rows, seeds,
                       f"MAIN, DOWNLINK LINKS ONLY  "
                       f"({args.seeds} seeds, {args.faults} faults/seed)",
                       args.boot, kind="downlink")
    diffs = report_contrast(out, args.boot)
    report_confusion(rows, kind="downlink")
    report_power(out, diffs, args.seeds, args.faults)

    pooled = report_table(rows, seeds,
                          "POOLED over link kinds (what the S3 result reported) "
                          "-- confounded, see NOTE", args.boot)
    report_contrast(pooled, args.boot)

    # -- feature availability ---------------------------------------------
    if args.features:
        report_features(cs, geo, list(range(min(args.seeds, 100))), args.faults)

    # -- fault count sweep -------------------------------------------------
    if args.faults_sweep:
        print()
        print("FAULTS-PER-SEED SWEEP  (downlink links, weather/pointing pair)")
        print("  more faults per seed buys samples but NOT independence -- the")
        print("  between-seed variance component does not shrink with it.")
        print(f"  {'faults':>7}{'seeds':>7}{'placed':>8}{'ident n':>9}{'acc':>7}"
              f"{'deg1 n':>8}{'acc':>7}{'gap':>8}{'gap 95% CI':>22}{'wx n':>7}{'s':>6}")
        ns = min(args.seeds, 120)
        sw = list(range(ns))
        for k in (8, 16, 32, 64):
            t0 = time.time()
            r = collect(cs, geo, sw, k, progress=False)
            placed = mean_placed(cs, sw[:20], k)
            a = per_seed_counts(r, sw, "ident", "pair", "downlink")
            b = per_seed_counts(r, sw, "deg1", "pair", "downlink")
            pa, _, _, na = cluster_ci(a, args.boot)
            pb, _, _, nb = cluster_ci(b, args.boot)
            d, lo, hi = cluster_ci_diff(a, b, args.boot)
            print(f"  {k:>7}{ns:>7}{placed:>8.1f}{na:>9,}{pa:>7.3f}{nb:>8,}{pb:>7.3f}"
                  f"{d:>+8.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}"
                  f"{sum(1 for x in r if x['truth']=='weather'):>7,}"
                  f"{time.time()-t0:>6.0f}")

    # -- weather-weighted kinds --------------------------------------------
    if args.weather:
        print()
        print("WEATHER-WEIGHTED KINDS POOL")
        print("  weather is scarce for two compounding reasons: it is 1 of 5 kinds in")
        print("  the pool, AND its dwell is 1-30 min against pointing's 0.5-6 h, so it")
        print("  contributes ~20-40x fewer belief samples per fault drawn.")
        wxpool = ((Cause.WEATHER,) * 6 + (Cause.POINTING,) * 2
                  + (Cause.NODE_DOWN, Cause.BUFFER, Cause.STALE_SCHED))
        ns = min(args.seeds, 120)
        sw = list(range(ns))
        print(f"  {'pool':>8}{'faults':>7}{'placed':>8}{'wx n':>8}{'pt n':>8}"
              f"{'wx share':>10}{'ident pair n':>13}{'acc':>7}{'gap 95% CI':>22}")
        for pool, tag in ((None, "default"), (wxpool, "wx x6")):
            for k in (8, 32, 64):
                r = collect(cs, geo, sw, k, kinds=pool, progress=False)
                a = per_seed_counts(r, sw, "ident", "pair", "downlink")
                b = per_seed_counts(r, sw, "deg1", "pair", "downlink")
                p, lo, hi, n = cluster_ci(a, args.boot)
                d, dlo, dhi = cluster_ci_diff(a, b, args.boot)
                wx = sum(1 for x in r if x["truth"] == "weather")
                pt = sum(1 for x in r if x["truth"] == "pointing")
                print(f"  {tag:>8}{k:>7}{mean_placed(cs, sw[:20], k, pool):>8.1f}"
                      f"{wx:>8,}{pt:>8,}{wx/max(wx+pt,1):>10.3f}{n:>13,}{p:>7.3f}"
                      f"{f'[{dlo:+.3f}, {dhi:+.3f}]':>22}")

    # -- P_STAY equalisation ----------------------------------------------
    if args.equalise:
        print()
        print("P_STAY EQUALISATION  -- is the deg1 excess dwell-time information "
              "or a leak?")
        print(f"  baseline kernel: weather {BAYES.P_STAY[Cause.WEATHER]}, "
              f"pointing {BAYES.P_STAY[Cause.POINTING]}")
        print("  The claim under test (C-009): the deg1 excess is legitimate dwell-time")
        print("  information, not a leak. If so, equalising the two dwell times must")
        print("  collapse it. Reference point is NOT 0.5 -- it is the always-pointing")
        print("  rate, because pointing dwells 0.5-6 h and weather 1-30 min.")
        print("  P(weather correct) is the number that isolates the leak: with the")
        print("  outcome row shared and the dwell times equal, weather is recoverable")
        print("  only from co-failure features -- which at degree 1 do not exist.")
        sw = list(range(min(args.seeds, 200)))
        rows_dl = rows

        def line(tag, r, seedlist):
            a = per_seed_counts(r, seedlist, "deg1", "pair", "downlink")
            p, lo, hi, n = cluster_ci(a, args.boot)
            maj, _, _ = baselines(r, "deg1", "pair", "downlink")
            sub = [x for x in r if keep(x, "deg1", "pair", "downlink")]
            wx = [x for x in sub if x["truth"] == "weather"]
            pt = [x for x in sub if x["truth"] == "pointing"]
            wacc = sum(x["hit"] for x in wx) / len(wx) if wx else float("nan")
            pacc = sum(x["hit"] for x in pt) / len(pt) if pt else float("nan")
            print(f"  {tag:>10}{n:>8,}{p:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}"
                  f"{maj:>9.3f}{p-maj:>+9.3f}{len(wx):>7,}{wacc:>9.3f}"
                  f"{len(pt):>7,}{pacc:>9.3f}")

        print(f"  {'P_STAY':>10}{'n':>8}{'acc':>8}{'95% CI':>20}{'always-pt':>9}"
              f"{'lift':>9}{'wx n':>7}{'wx acc':>9}{'pt n':>7}{'pt acc':>9}")
        line("as built", rows_dl, seeds)
        for v in (0.85, 0.91, 0.97):
            with equalised_p_stay(v):
                r = collect(cs, geo, sw, args.faults, progress=False)
            line(f"{v:.2f}", r, sw)
            if v == 0.91:
                report_confusion(r, bins=("deg1",), kind="downlink")


if __name__ == "__main__":
    main()
