"""Run the diagnosability-gated peer cross-check arm and score it HONESTLY.

Writes web/public/trace_gemma_peer_gated.json -- a NEW file. Every existing
trace is left alone, so a negative result costs nothing but API calls.

    python -m tools.run_gemma_peer_gated

What this script refuses to do
------------------------------
Present raw cross-arm accuracy as a controlled comparison. The arms do not run
the same trajectory: a belief becomes an action, an action can exclude a
contact, and from there the two runs see different observations. The faulted
denominators already differ across the existing traces (gemma_cot 179,
gemma_peer 181, bayes 1089), so "65.9% vs 50.8%" is a difference of two
populations as much as a difference of two methods.

So every headline number here is reported twice:

  * RAW      -- each arm on its own trajectory, denominator stated every time.
                Descriptive. Not a comparison.
  * MATCHED  -- restricted to the (node, t) decision points that exist in BOTH
                traces AND reported the SAME target, so ground truth is
                identical on both sides and the only thing that varies is the
                belief rule. This is the only figure that supports a causal
                claim, and its n is printed with it.

If the gate does not recover accuracy the matched block will say so with
numbers. Do not tune the gate against it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sim.config import DEFAULT
from sim import contacts as C
from sim import diagnosability as D
from sim import trace as T
from sim.advisor.gemma_peer_gated import GemmaPeerGatedAdvisor

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "web" / "public"
OUT = PUB / "trace_gemma_peer_gated.json"

# The frontend's UNCERTAIN threshold (CONF in web/src/trace.ts). A belief whose
# top probability sits below it renders the node as UNCERTAIN.
CONF = 0.65

# Loaded for context. gemma_peer_gated is appended after the run.
ARMS = [
    ("bayes", PUB / "trace_bayes.json"),
    ("gemma", PUB / "trace_gemma.json"),
    ("gemma_cot", PUB / "trace_gemma_cot.json"),
    ("gemma_peer", PUB / "trace_gemma_peer.json"),
]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def index(blob: dict) -> dict[tuple[str, int], dict]:
    """(node, t) -> the belief sample, for joining two arms decision by decision."""
    out: dict[tuple[str, int], dict] = {}
    for node, series in blob["beliefs"].items():
        for b in series:
            out[(node, int(b["t"]))] = b
    return out


def accuracy(blob: dict) -> tuple[int, int]:
    """Correct / total over samples whose reported link was genuinely faulted."""
    correct = total = 0
    for series in blob["beliefs"].values():
        for b in series:
            if b["truth"] != "nominal":
                total += 1
                correct += (b["x"] == b["truth"])
    return correct, total


def uncertain(blob: dict) -> tuple[int, int, int, int]:
    """(sub-threshold, all samples, sub-threshold on faulted, faulted)."""
    sub = tot = sub_f = tot_f = 0
    for series in blob["beliefs"].values():
        for b in series:
            low = max(b["d"]) < CONF
            tot += 1
            sub += low
            if b["truth"] != "nominal":
                tot_f += 1
                sub_f += low
    return sub, tot, sub_f, tot_f


def _pct(n: int, d: int) -> str:
    return f"{n / d * 100:5.1f}%" if d else "    -"


def mcnemar_p(b01: int, b10: int) -> float:
    """Exact two-sided McNemar on the PAIRED discordant counts.

    The two arms answer the same decision points, so the samples are paired and
    an unpaired proportion test would overstate the evidence. With single-digit
    discordant counts the normal approximation is not usable either; this is the
    exact binomial. A delta of a few samples on ~180 paired points is very
    likely noise, and this number is here to stop that being reported as a win.
    """
    from math import comb
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def matched(a: dict, b: dict) -> dict:
    """Accuracy of two arms on the decision points they actually share.

    A point qualifies only if both traces decided at the same (node, t) AND
    reported the same target -- ground truth is a function of (node, target, t),
    so equal targets is what makes the two sides answer the same question.
    """
    A, B = index(a), index(b)
    common = [k for k in A if k in B and A[k]["tg"] == B[k]["tg"]]
    disputed = [k for k in common if A[k]["truth"] != B[k]["truth"]]
    faulted = [k for k in common if A[k]["truth"] != "nominal"]
    return {
        "n_a": len(A), "n_b": len(B),
        "n_common": len(common), "n_faulted": len(faulted),
        "truth_mismatch": len(disputed),
        "a_correct": sum(A[k]["x"] == A[k]["truth"] for k in faulted),
        "b_correct": sum(B[k]["x"] == B[k]["truth"] for k in faulted),
        "a_unc": sum(max(A[k]["d"]) < CONF for k in faulted),
        "b_unc": sum(max(B[k]["d"]) < CONF for k in faulted),
        # paired discordant counts -- b right where a was wrong, and vice versa
        "b01": sum(A[k]["x"] != A[k]["truth"] and B[k]["x"] == B[k]["truth"]
                   for k in faulted),
        "b10": sum(A[k]["x"] == A[k]["truth"] and B[k]["x"] != B[k]["truth"]
                   for k in faulted),
        "keys": faulted,
        "A": A, "B": B,
    }


def main() -> int:
    t_start = time.time()

    cs = C.read(DEFAULT)
    strip = D.compute(cs, DEFAULT)
    adv = GemmaPeerGatedAdvisor(strip)
    print(f"[gated] strip: {len(strip.per_link)} links x "
          f"{len(strip.t_grid)} grid points "
          f"({(strip.t_grid[1] - strip.t_grid[0]) // 1000} s step)")
    print(f"[gated] cache seeded with {adv.seeded_cache_entries} response(s) "
          f"already paid for by gemma_peer / gemma_cot")
    print(f"[gated] running the gated peer cross-check arm against {adv.model} ...")

    blob = T.build(DEFAULT, advisor=adv)

    attached = 0
    for node, series in blob["beliefs"].items():
        for b in series:
            rec = adv.records.get((node, int(b["t"])))
            if rec is None:
                continue
            b["ev"] = {
                "peer": rec["peer"], "prompt": rec["prompt"], "raw": rec["raw"],
                "steps": rec["steps"], "peers": rec["peers"],
                "consensus": rec["consensus"],
                "identifiable": rec["identifiable"],
                "gated": rec["gated"],
            }
            attached += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(blob, separators=(",", ":")), encoding="utf-8")
    print(f"[gated] wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"[gated] reasoning + peers + gate state attached to {attached} decisions")
    print(f"[gated] run took {time.time() - t_start:.0f}s")

    # ---- load the other arms ---------------------------------------------
    loaded: list[tuple[str, dict]] = []
    for name, path in ARMS:
        if path.exists():
            loaded.append((name, json.loads(path.read_text(encoding="utf-8"))))
        else:
            print(f"[gated] note: {path.name} missing -- {name} omitted")
    loaded.append(("gemma_peer_gated", blob))

    # ==================================================================
    print()
    print("=" * 78)
    print("RAW -- each arm on its OWN trajectory. Denominators differ because a")
    print("belief becomes an action and an action changes which contacts happen.")
    print("These columns are DESCRIPTIVE. They are not a controlled comparison.")
    print("=" * 78)
    print(f"  {'ARM':<18} {'ACC ON FAULTED':>18}   {'<0.65 (ALL)':>16}   "
          f"{'<0.65 (FAULTED)':>18}")
    for name, b in loaded:
        c, tot = accuracy(b)
        s, st, sf, tf = uncertain(b)
        print(f"  {name:<18} {_pct(c, tot)} ({c:4d}/{tot:<5d}) "
              f"{s:5d}/{st:<5d} {_pct(s, st)}   {sf:5d}/{tf:<5d} {_pct(sf, tf)}")

    # ==================================================================
    print()
    print("=" * 78)
    print("GATE DIAGNOSTICS -- gemma_peer_gated only")
    print("=" * 78)
    recs = list(adv.records.values())
    with_peers = [r for r in recs if r["peers"]]
    disagreed = [r for r in with_peers if not r["consensus"]]
    gated = [r for r in recs if r["gated"]]
    ident = [r for r in recs if r["identifiable"]]
    unident = [r for r in recs if not r["identifiable"]]

    print(f"  decisions with an LLM call            {len(recs)}")
    print(f"    of those, had >=1 peer report       {len(with_peers)} "
          f"({_pct(len(with_peers), len(recs)).strip()})")
    print(f"    of those, peers DISAGREED           {len(disagreed)} "
          f"({_pct(len(disagreed), len(with_peers)).strip()} of consulted)")
    print()
    print(f"  window IDENTIFIABLE                   {len(ident)} "
          f"({_pct(len(ident), len(recs)).strip()})")
    print(f"  window UNIDENTIFIABLE                 {len(unident)} "
          f"({_pct(len(unident), len(recs)).strip()})")
    print()
    print(f"  GATE FIRED (disagreement overridden)  {adv.n_gated}")
    print(f"  spread allowed (disagree + unident.)  {adv.n_spread}")
    if disagreed:
        print(f"  -> {_pct(adv.n_gated, len(disagreed)).strip()} of all "
              f"disagreements were overridden as decidable-in-principle")
    print()
    print(f"  strip-lookup validation: the strip read at the observation's frame")
    print(f"  agreed with obs.self_degree>1 or obs.peer_degree>1 (the same")
    print(f"  predicate, measured by the engine with no grid rounding) on")
    print(f"  {adv.n_degree_agree}/{adv.n_degree_checked} decisions "
          f"({_pct(adv.n_degree_agree, adv.n_degree_checked).strip()}).")

    # ---- accuracy split by window ----------------------------------------
    idx = index(blob)
    buckets: dict[tuple[bool, bool], list[tuple[bool, bool]]] = {}
    for (node, t), rec in adv.records.items():
        b = idx.get((node, t))
        if b is None or b["truth"] == "nominal":
            continue
        key = (rec["identifiable"], rec["gated"])
        buckets.setdefault(key, []).append(
            (b["x"] == b["truth"], max(b["d"]) < CONF))

    def roll(pred) -> tuple[int, int, int]:
        rows = [r for k, v in buckets.items() if pred(k) for r in v]
        return sum(r[0] for r in rows), len(rows), sum(r[1] for r in rows)

    print()
    print("  ACCURACY BY WINDOW (faulted samples that had an LLM call)")
    print(f"  {'BUCKET':<34} {'ACC':>7} {'n':>7}   {'<0.65':>7}")
    for label, pred in (
        ("identifiable window",        lambda k: k[0]),
        ("  of which the gate fired",  lambda k: k[0] and k[1]),
        ("  of which it did not",      lambda k: k[0] and not k[1]),
        ("unidentifiable window",      lambda k: not k[0]),
        ("ALL",                        lambda k: True),
    ):
        c, n, u = roll(pred)
        print(f"  {label:<34} {_pct(c, n)} {n:7d}   {u:7d}")

    # ==================================================================
    print()
    print("=" * 78)
    print("MATCHED -- restricted to (node, t) decision points present in BOTH")
    print("traces with the SAME reported target, so ground truth is identical on")
    print("both sides. This is the only block that supports a causal claim.")
    print("=" * 78)
    others = [(n, b) for n, b in loaded if n != "gemma_peer_gated"]
    print(f"  {'vs ARM':<16} {'n(matched,faulted)':>19}  "
          f"{'THAT ARM':>12}  {'GATED':>12}  {'DELTA':>8}")
    for name, other in others:
        m = matched(other, blob)
        n = m["n_faulted"]
        if m["truth_mismatch"]:
            print(f"  !! {name}: {m['truth_mismatch']} matched points disagree "
                  f"on ground truth -- join is unsound, ignore this row")
            continue
        if n == 0:
            print(f"  {name:<16} {0:>19}  -- no shared faulted decision points --")
            continue
        pa, pb = m["a_correct"] / n, m["b_correct"] / n
        print(f"  {name:<16} {n:>19}  {pa * 100:11.1f}%  {pb * 100:11.1f}%  "
              f"{(pb - pa) * 100:+7.1f}")
    print()
    print(f"  {'vs ARM':<16} {'n(matched,faulted)':>19}  "
          f"{'THAT ARM <0.65':>15}  {'GATED <0.65':>12}")
    for name, other in others:
        m = matched(other, blob)
        n = m["n_faulted"]
        if n == 0 or m["truth_mismatch"]:
            continue
        print(f"  {name:<16} {n:>19}  {m['a_unc']:>15}  {m['b_unc']:>12}")

    # ---- the one comparison the gate is actually about --------------------
    m = matched(dict(loaded)["gemma_peer"], blob) if any(
        n == "gemma_peer" for n, _ in loaded) else None
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if m is None:
        print("  trace_gemma_peer.json is missing; the gate cannot be scored "
              "against the arm it modifies.")
    elif m["n_faulted"] == 0:
        print("  gemma_peer and gemma_peer_gated share NO faulted decision "
              "point with the same target.")
        print("  The trajectories diverged completely; nothing here is a "
              "controlled comparison.")
    else:
        n = m["n_faulted"]
        d = (m["b_correct"] - m["a_correct"]) / n * 100
        pv = mcnemar_p(m["b01"], m["b10"])
        print(f"  On the {n} faulted decision points gemma_peer and "
              f"gemma_peer_gated share:")
        print(f"    gemma_peer        {_pct(m['a_correct'], n).strip()}  "
              f"({m['a_correct']}/{n}), {m['a_unc']} below {CONF}")
        print(f"    gemma_peer_gated  {_pct(m['b_correct'], n).strip()}  "
              f"({m['b_correct']}/{n}), {m['b_unc']} below {CONF}")
        print(f"    delta {d:+.1f} pts. Paired: gate fixed {m['b01']}, "
              f"broke {m['b10']}; exact McNemar p = {pv:.4f}")
        if pv > 0.05:
            print(f"    p > 0.05 on {m['b01'] + m['b10']} discordant pair(s). "
                  f"The accuracy move is NOT")
            print(f"    significant. Report the direction if you like; do not "
                  f"report it as a recovery.")
        elif d < 0:
            print(f"    THE GATE DID NOT RECOVER ACCURACY. Report this as it "
                  f"stands; peer disagreement")
            print(f"    evidently carries information the strip does not. Do "
                  f"not move the rule until")
            print(f"    the number improves -- that is fitting the metric.")
        else:
            print(f"    The gate recovered accuracy on the shared subset.")

        # The counterfactual that isolates the gate: the SAME decision points
        # where it fired, scored both ways. Everything outside this set is
        # identical by construction, so this is the whole of the gate's effect.
        A, B = m["A"], m["B"]
        gk = [k for k in m["keys"]
              if (B[k].get("ev") or {}).get("gated")]
        print()
        if not gk:
            print("  The gate fired on no faulted decision point; it changed "
                  "nothing measurable.")
        else:
            ga = sum(A[k]["x"] == A[k]["truth"] for k in gk)
            gb = sum(B[k]["x"] == B[k]["truth"] for k in gk)
            ua = sum(max(A[k]["d"]) < CONF for k in gk)
            ub = sum(max(B[k]["d"]) < CONF for k in gk)
            print(f"  ISOLATED -- the {len(gk)} faulted points where the gate "
                  f"actually fired, scored")
            print(f"  both ways. Outside this set the two arms are identical "
                  f"by construction.")
            print(f"    spread (gemma_peer)   {_pct(ga, len(gk)).strip()}  "
                  f"({ga}/{len(gk)}), {ua} below {CONF}")
            print(f"    kept own (gated)      {_pct(gb, len(gk)).strip()}  "
                  f"({gb}/{len(gk)}), {ub} below {CONF}")
            print(f"  Keeping the node's own reading in a window the geometry "
                  f"calls decidable is")
            print(f"  right {_pct(gb, len(gk)).strip()} of the time. The gate's "
                  f"premise -- that a disagreeing peer")
            print(f"  in an identifiable window is simply wrong -- would "
                  f"predict much better than")
            print(f"  that. Often it is the NODE that is wrong, and the strip "
                  f"cannot tell which.")
    if adv.n_gated == 0:
        print()
        print("  NOTE: the gate never fired, so gemma_peer_gated is "
              "gemma_peer with extra bookkeeping.")
        print("  Any difference above is trajectory noise, not the gate.")

    # ---- examples ---------------------------------------------------------
    print()
    print("=" * 78)
    print("EXAMPLES WHERE THE GATE FIRED (first 4)")
    print("=" * 78)
    shown = 0
    for (node, t), rec in sorted(adv.records.items(), key=lambda kv: kv[0][1]):
        if not rec["gated"] or shown >= 4:
            continue
        b = idx.get((node, t))
        truth = b["truth"] if b else "?"
        got = b["x"] if b else "?"
        print(f"  [{shown + 1}] {node} @ t={t} link {rec['link']} "
              f"(frame {rec['frame']}, identifiable)")
        print(f"      own reading : {rec['own_cause']} @ {rec['own_conf']:.2f}")
        for e in rec["peers"]:
            flag = "  <-- disagrees" if e["cause"] != rec["own_cause"] else ""
            print(f"      peer        : {e['node']} says {e['cause']} "
                  f"@ {e['conf']:.2f}{flag}")
        print(f"      TRUTH       : {truth}   |   kept: {got}   |   "
              f"{'CORRECT' if got == truth else 'WRONG'}")
        shown += 1
    if shown == 0:
        print("  none -- the gate never fired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
