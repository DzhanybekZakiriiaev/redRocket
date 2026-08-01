"""Peer cross-check, GATED on the project's own diagnosability result.

Why this arm exists
-------------------
`gemma_peer` fixed calibration and paid for it in accuracy. Measured:

    ARM          ACC(faulted)   UNCERTAIN(<0.65)
    gemma_cot        65.9%            0
    gemma_peer       50.8%           98

The mechanism that produced the doubt is the same mechanism that produced the
errors. When two nodes disagree about one target, `_consensus_belief` spreads
mass across the competing causes -- and a peer that is CONFIDENTLY WRONG pulls
the node off a correct answer just as hard as a peer that is confidently right.
Observed directly in the peer trace: two peers asserted POINTING at 0.95 and
0.98, the truth was WEATHER, and the node conformed.

Spreading belief is the correct response to *irreducible* ambiguity and the
wrong response to *a peer being mistaken*. The peer arm cannot tell those apart,
because a disagreement looks identical in both cases. So it hedges on all of
them.

The gate
--------
The project can tell them apart, and it can do it without the simulator, the
faults, or the model -- from schedule geometry alone (`sim/diagnosability.py`):

    identifiable(s, g, t)  <=>  deg(g, t) > 1  OR  deg(s, t) > 1

Where that is FALSE, the competing hypotheses predict the identical observation
set: the posterior ratio equals the prior ratio for every observation, so no
estimator separates them. A disagreement there is honest ambiguity and hedging
is the only defensible posterior.

Where it is TRUE, evidence exists that could settle the question. A peer that
disagrees in an identifiable window is not surfacing ambiguity -- it is simply
wrong, and so is deferring to it. The node keeps its own conclusion.

    disagreement AND NOT identifiable  ->  spread (inherit GemmaPeerAdvisor)
    disagreement AND identifiable      ->  keep this node's own sharp belief
    agreement (or no peers)            ->  unchanged from the parent

Nothing else changes. Same prompt (byte-identical to `gemma_peer`'s, including
the peer block), same model, same call path, same decision rule, same thing
posted to the board. The gate touches ONLY how a disagreement is turned into a
belief, so the comparison isolates the gate and nothing else.

WHICH INSTANT THE STRIP IS READ AT -- read this before changing it
------------------------------------------------------------------
The strip is read at the OBSERVATION's timestamp, not at the decide tick.

Decide events fire every 5 minutes; `_last_obs` holds an adverse observation
until something replaces it, so a decision is routinely made hours after the
contact it is about, when the link is long closed. Reading the strip at the
decide tick would therefore report "not identifiable" for almost every decision
-- not because the evidence was ambiguous but because we asked about the wrong
moment. The identifiability of the evidence is a property of when the evidence
was GATHERED.

Grid alignment: `Strip.t_grid` is a 60 s grid anchored at the scenario epoch,
while a contact opens on the 10 s search granularity, so `obs.t` is generally
not a grid point. We take the first grid point AT OR AFTER `obs.t` (ceil, not
round). The strip masks each link to `g >= t_open & g <= t_close`, so ceil is
the first frame at which the strip describes this link at all; rounding to the
nearest frame lands before `t_open` about half the time and reads back a
spurious False from the window mask rather than from the geometry.

This is checkable, and it is checked. `Observation.self_degree` /
`peer_degree` are the same per-kind concurrent-contact counts the strip is built
from, measured by the engine at exactly `obs.t` (sim/engine.py `_on_contact`,
FINDINGS F-001). So `obs.self_degree > 1 or obs.peer_degree > 1` is the same
predicate evaluated with no grid rounding at all. Every decision records both;
the runner prints how often they agree. A low agreement rate means this lookup
is wrong and the gate's diagnostics are not to be believed.

Is the strip legitimate input for an advisor?
---------------------------------------------
Yes, and it is the narrow case. `sim/diagnosability.py` is computed from the
contact plan ALONE -- no simulator, no faults, no ground truth. A contact plan
is uploaded to a spacecraft in advance; knowing your own schedule is not
privileged information the way a FaultEvent is. The invariant that matters
(tests/test_bayes.py::test_advisor_never_sees_ground_truth) is untouched: this
arm still never sees a fault, and the degree agreement above proves the same
signal was already reachable from the Observation stream on its own.

What this arm gives up
----------------------
Uncertainty, on purpose, in exactly the windows where the geometry says the
evidence could have settled it. If accuracy does not go up, the gate is wrong
and the honest reading is that peer disagreement carries information the strip
does not -- report that, do not move the rule until a number improves.

Its own cache file. It is seeded from `gemma_peer`'s and `gemma_cot`'s caches
because `_call_raw`'s key is `sha256("cot\\n<model>\\n<prompt>")` and this arm's
prompts are unchanged, so a response already paid for is reused rather than
re-billed. Trajectories diverge once the gate changes an action, and the
prompts that only exist on this arm's trajectory are fetched fresh.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import ScenarioConfig, DEFAULT
from ..types import Action, Cause, Outcome, Policy, Strip, unpack
from .base import bayes_action, nominal_belief
from .gemma import _LETTER_TO_CAUSE, _REPO_ROOT, _belief_from
from .gemma_peer import GemmaPeerAdvisor

# Caches whose keys are produced by GemmaCotAdvisor._call_raw and therefore
# interchangeable with this arm's, prompt for prompt. gemma_cache.json is NOT
# here: the baseline's _call uses a different key prefix AND stores a bare
# parsed dict instead of {"parsed", "raw"}, so merging it would silently drop
# the raw reply the trace panel shows.
_SEED_CACHES = ("gemma_peer_cache.json", "gemma_cot_cache.json")


class GemmaPeerGatedAdvisor(GemmaPeerAdvisor):
    """`gemma_peer`, but disagreement only spreads belief where the geometry
    says the causes are genuinely inseparable."""

    name = "gemma_peer_gated"

    def __init__(self, strip: Strip | None = None, *,
                 contacts=None, cfg: ScenarioConfig = DEFAULT,
                 seed_from_peer_cache: bool = True, **kw) -> None:
        kw.setdefault("cache_path",
                      str(_REPO_ROOT / "out" / "gemma_peer_gated_cache.json"))
        super().__init__(**kw)

        if strip is None:
            if contacts is None:
                from .. import contacts as C
                contacts = C.read(cfg)
            from .. import diagnosability as D
            strip = D.compute(contacts, cfg)

        grid = list(strip.t_grid)
        if len(grid) < 2:
            raise RuntimeError(
                "gemma_peer_gated: the strip has fewer than two grid points, so "
                "there is no grid step to index with. Pass a strip built by "
                "sim.diagnosability.compute.")
        self.strip = strip
        self._grid0 = int(grid[0])
        self._step = int(grid[1] - grid[0])
        self._n = len(grid)
        # Accept either an in-memory Strip (list[bool]) or a round-tripped one
        # (RLE string), so a caller can hand us strip.to_dict()["per_link"] or
        # the trace's own "strip" block without a conversion dance.
        self._bits: dict[str, list[bool]] = {
            k: (unpack(v) if isinstance(v, str) else list(v))
            for k, v in strip.per_link.items()
        }

        # Gate bookkeeping, for the runner's diagnostics.
        self.n_identifiable = 0
        self.n_unidentifiable = 0
        self.n_gated = 0
        self.n_spread = 0
        self.n_degree_agree = 0
        self.n_degree_checked = 0

        if seed_from_peer_cache:
            self._seed_cache()

    # -- cache reuse --------------------------------------------------------

    def _seed_cache(self) -> None:
        """Fold in responses the peer / reasoning-first arms already paid for.

        Only ever ADDS keys -- anything already in this arm's own cache wins, so
        seeding can never rewrite a response this arm recorded itself.
        """
        added = 0
        for fn in _SEED_CACHES:
            p = Path(_REPO_ROOT) / "out" / fn
            if not p.exists():
                continue
            try:
                other = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            for k, v in other.items():
                if k not in self._cache and isinstance(v, dict) and "parsed" in v:
                    self._cache[k] = v
                    added += 1
        self.seeded_cache_entries = added
        # Persist immediately. _call_raw only writes on a MISS, so an arm whose
        # prompts are all covered by the seed would finish having never created
        # its own cache file -- and the next run would silently depend on the
        # other arms' caches still being present. Write it once here so this
        # arm is reproducible on its own.
        if added:
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._cache_path.write_text(json.dumps(self._cache),
                                            encoding="utf-8")
            except Exception:
                pass

    # -- the gate -----------------------------------------------------------

    def _frame(self, t: int) -> int:
        """First strip grid point at or after `t`, clamped to the grid.

        Ceil, not round -- see the module docstring. `t` below the grid start
        clamps to 0 and `t` past the end clamps to the last frame; both are
        outside every contact window, where the strip reports False anyway.
        """
        off = int(t) - self._grid0
        if off <= 0:
            return 0
        i = (off + self._step - 1) // self._step
        return i if i < self._n else self._n - 1

    def link_key(self, a: str, b: str) -> str:
        """Strip key for an undirected link. `Strip.per_link` is keyed
        f"{min}|{max}" (sim/diagnosability.py), so the direction the observation
        happened to be made in must not change the lookup."""
        return f"{min(a, b)}|{max(a, b)}"

    def identifiable_at(self, a: str, b: str, t: int) -> bool:
        """Does the schedule geometry permit separating the causes on this link
        at this instant? Unknown link -> False (the conservative reading: we
        cannot claim evidence exists that we have no schedule for)."""
        bits = self._bits.get(self.link_key(a, b))
        if not bits:
            return False
        return bool(bits[self._frame(t)])

    # -- output -------------------------------------------------------------

    def decide(self, node: str, t: int) -> Policy:
        adverse = [(p, o) for (n, p), o in self._last_obs.items()
                   if n == node and o.outcome is not Outcome.OK]
        if not adverse:
            return Policy(nominal_belief(), Action.NONE, "", t, 1.0, "")

        target, obs = max(adverse, key=lambda kv: kv[1].t)
        peers = self._peers_for(target, node, t)
        prompt = self._prompt_with_peers(obs, peers, t)
        parsed, raw = self._call_raw(prompt)

        if parsed is None:
            b = nominal_belief()
            return Policy(b, bayes_action(b)[0], target, t, 0.0,
                          "[gemma_peer_gated unavailable]")

        cause = _LETTER_TO_CAUSE.get(
            str(parsed.get("cause", "")).strip()[:1].upper(), Cause.NOMINAL)
        conf = min(0.98, max(0.2, float(parsed.get("confidence", 0.6))))

        # --- cross-check (identical to the parent) --------------------------
        votes: dict[Cause, float] = {cause: conf}
        for e in peers:
            pc = Cause(e["cause"])
            votes[pc] = votes.get(pc, 0.0) + float(e["conf"])
        consensus = len(votes) == 1

        # --- the gate -------------------------------------------------------
        # Read at obs.t, the moment the evidence was gathered, NOT at t.
        identifiable = self.identifiable_at(node, target, obs.t)
        # Same predicate straight off the Observation, with no grid rounding --
        # a standing check that the strip lookup above is reading the right
        # frame. Never used to decide anything; only reported.
        deg_identifiable = bool(obs.self_degree > 1 or obs.peer_degree > 1)
        self.n_degree_checked += 1
        self.n_degree_agree += (identifiable == deg_identifiable)
        if identifiable:
            self.n_identifiable += 1
        else:
            self.n_unidentifiable += 1

        gated = (not consensus) and identifiable
        if consensus or gated:
            # gated: the peers disagree, but the geometry says the evidence
            # could have settled it, so at least one of us is plainly wrong and
            # hedging would not be honest doubt -- it would be deference.
            belief = _belief_from(cause, conf)
            reported = conf
        else:
            belief = self._consensus_belief_for(votes)
            reported = max(belief.values())

        if gated:
            self.n_gated += 1
        elif not consensus:
            self.n_spread += 1

        action, _ = bayes_action(belief)

        steps = parsed.get("steps") or []
        if isinstance(steps, str):
            steps = [steps]
        steps = [str(s)[:400] for s in steps][:12]

        rationale = str(parsed.get("rationale", ""))
        if gated:
            others = ", ".join(sorted({e["node"] for e in peers
                                       if e["cause"] != cause.value}))
            rationale = (f"[peers disagree: {others}; window identifiable -- "
                         f"holding own reading] {rationale}")
        elif not consensus:
            others = ", ".join(sorted({e["node"] for e in peers
                                       if e["cause"] != cause.value}))
            rationale = f"[peers disagree: {others}] {rationale}"

        self.records[(node, t)] = {
            "peer": target, "prompt": prompt, "raw": raw, "steps": steps,
            "peers": [dict(e) for e in peers],
            "consensus": consensus,
            # -- what this arm adds --
            "identifiable": bool(identifiable),
            "gated": bool(gated),
            # diagnostics, not UI contract
            "link": self.link_key(node, target),
            "obs_t": int(obs.t),
            "frame": self._frame(obs.t),
            "deg_identifiable": deg_identifiable,
            "own_cause": cause.value,
            "own_conf": round(conf, 4),
        }

        # Posted AFTER deciding, and it is this node's OWN reading -- never the
        # consensus and never the gated result. Republishing an aggregate would
        # let one opinion re-enter the pool under another node's name, which is
        # the belief-forwarding incest the gossip layer is built to avoid.
        self._post(target, node, cause, conf, t)

        return Policy(
            belief=belief, action=action, target=target,
            until=t + 30 * 60_000, confidence=reported,
            rationale=rationale[:160],
        )

    # -- seam ---------------------------------------------------------------

    def _consensus_belief_for(self, votes: dict[Cause, float]) -> dict[str, float]:
        """Indirection so the ungated path is provably the parent's rule.

        Overriding this in a subclass changes the spread; it does not change the
        gate. Keeping them separable is the only way a future change to one can
        be attributed without re-running both arms.
        """
        from .gemma_peer import _consensus_belief
        return _consensus_belief(votes)
