"""Gemma advisor with peer cross-check -- nodes check each other's diagnoses.

Why this arm exists
-------------------
Measured on the baseline and reasoning-first arms: across 2304 belief samples
Gemma NEVER reports a confidence below 0.70, including in windows the
diagnosability strip proves are undecidable from the evidence alone. The Bayes
filter on the same evidence splits down to 0.33 there. The frontend's UNCERTAIN
threshold (`CONF = 0.65` in web/src/trace.ts) is therefore never crossed on the
Gemma arms, and no node ever displays doubt. The model is confidently wrong and
looks exactly like the model being confidently right.

A single overconfident model cannot detect its own overconfidence -- there is
nothing in its context that disagrees with it. So give it something that does:
the conclusions other orbiters reached about the SAME target. When two nodes
looking at one target reach incompatible conclusions, at least one is wrong,
and that fact is available even though neither node can tell which.

Two mechanisms, and only one of them is trusted
-----------------------------------------------
1. The prompt tells the model what its peers concluded and asks it to reconcile.
   Whether the model actually lowers its own number in response is an open
   question this arm measures rather than assumes.
2. The BELIEF is built by ensembling the conclusions. This is the mechanism that
   does the work, because it does not depend on the model cooperating.

Ensembling CONCLUSIONS, not evidence -- read this before changing it
--------------------------------------------------------------------
The consensus belief spreads mass across the distinct causes that were named,
weighted by the confidence attached to each. That is an ensemble over
independent *conclusions*: N diagnosticians voted, they disagreed, so the
posterior is not sharp.

It is NOT a merge of evidence. Peer observations are never folded into this
node's posterior, no likelihoods are multiplied, and nothing here makes the
belief SHARPER than the node's own reading. That restraint is deliberate:
merging posteriors double-counts evidence. A's opinion reaches C directly and
via B, C reads it as two independent confirmations, and the network converges
confidently wrong -- the exact failure PLAN.md:229 designs the Evidence tuple
around ("gossip evidence, not beliefs") and that
tests/test_bayes.py::test_a_nodes_own_gossip_does_not_confirm_itself pins down.
The invariant to preserve: disagreement may only ever REMOVE confidence here.
Agreement adds none.

Honest limitation -- the board is a shared pool, not a delivered message
------------------------------------------------------------------------
The engine constructs ONE advisor instance and shares it across every node
(sim/engine.py: `self.advisor.decide(node, t)` in a loop), and `receive()` is
handed gossip without any parameter saying which node received it. The
inherited `_peer_reports` already has this property; the diagnosis board here
inherits it too.

So a "peer report" is NOT something that provably traversed a link and arrived
before the deadline. It is a diagnosis some other node made recently, read out
of a pool every node can see. On a real delay-tolerant relay it would have to be
carried by an actual contact, and some of these reports would not have arrived
in time -- or at all. This arm therefore models an upper bound on what peer
cross-check can buy, not the delivered-in-time reality.

What bounds it: the same 4-hour recency window the gossip path already uses
(`GemmaAdvisor._peer_cofail`), so a report is at least contemporaneous with the
decision rather than arbitrarily old. That is a real constraint and it is the
only one. Do not describe this arm as delay-tolerant peer consensus.

Everything else is inherited from GemmaCotAdvisor unchanged -- same evidence
bullets, same reasoning-first output contract, same decision rule. When no peer
has diagnosed the target recently the prompt is byte-identical to the
reasoning-first arm's, so the comparison isolates the cross-check and nothing
else. Its own cache file, so its responses can never be confused with another
arm's.
"""

from __future__ import annotations

import re

from ..types import Action, Cause, CAUSES, Outcome, Policy
from .base import bayes_action, nominal_belief
from .gemma import (
    _CAUSE_MENU, _LETTER_TO_CAUSE, _REPO_ROOT, _belief_from, LETTERS,
)
from .gemma_cot import GemmaCotAdvisor

# Recency window for peer reports. Deliberately the same 4 hours
# GemmaAdvisor._peer_cofail uses for gossiped observations: a diagnosis is no
# more current than the observation it was made from, so it cannot justify a
# longer window.
PEER_WINDOW_MS = 4 * 3600 * 1000

# At most this many peers are quoted, most recent first -- one per peer node.
# Bounds prompt size and, because the SAME list feeds the consensus belief, the
# prompt and the belief can never disagree about who was consulted.
MAX_PEERS = 6

# Mass parked on causes NOBODY named, so a disagreement belief stays a valid
# interior point of the simplex rather than assigning literal zero to a cause
# that merely went unmentioned. Mirrors the spirit of _belief_from's spread.
CONSENSUS_FLOOR = 0.02

_CAUSE_TO_LETTER = {c: LETTERS[i] for i, c in enumerate(CAUSES)}


def _menu_labels() -> dict[str, str]:
    """Headline label per letter, parsed out of the shared cause menu.

    Parsed rather than retyped so a peer report can never quote a cause by a
    name the menu no longer uses. If the menu is reworded past recognition this
    raises instead of quietly emitting labels the model has never seen.
    """
    labels: dict[str, str] = {}
    for line in _CAUSE_MENU.splitlines():
        m = re.match(r"^([A-F])\.\s+([A-Z][A-Z0-9 /-]*[A-Z])", line)
        if m:
            labels[m.group(1)] = m.group(2)
    if len(labels) != len(LETTERS):
        raise RuntimeError(
            "gemma_peer: could not parse a headline label for every cause out "
            f"of _CAUSE_MENU (got {sorted(labels)}). The menu changed; update "
            "this parser rather than quoting peers with invented labels.")
    return labels


_LABELS = _menu_labels()


def _consensus_belief(votes: dict[Cause, float]) -> dict[str, float]:
    """Spread mass over the causes that were NAMED, weighted by confidence.

    `votes` is cause -> summed confidence across everyone who named it. This is
    a vote count, NOT a likelihood product: two nodes naming the same cause make
    that cause hold a larger SHARE of a fixed unit of mass, they never push the
    total above what one node alone could claim. See the module docstring --
    multiplying here would be evidence double-counting.

    Sums to exactly 1.0; the float residue lands on the largest component.
    """
    named = {c: w for c, w in votes.items() if w > 0.0}
    if not named:                       # every vote arrived weightless
        named = {c: 1.0 for c in votes}
    total = sum(named.values())

    others = [c for c in CAUSES if c not in named]
    share = 1.0 - CONSENSUS_FLOOR * len(others)

    b = {c.value: CONSENSUS_FLOOR for c in others}
    for c, w in named.items():
        b[c.value] = share * w / total

    # Exact simplex. The float residue is dumped on the largest component;
    # re-adding can itself round, so repeat until it lands or give up after a
    # few passes (it converges in one or two).
    top = max(b, key=b.__getitem__)
    for _ in range(4):
        residue = 1.0 - sum(b.values())
        if residue == 0.0:
            break
        b[top] += residue
    return b


class GemmaPeerAdvisor(GemmaCotAdvisor):
    """Reasoning-first Gemma, told what its peers concluded about the same target."""

    name = "gemma_peer"

    def __init__(self, **kw) -> None:
        kw.setdefault("cache_path", str(_REPO_ROOT / "out" / "gemma_peer_cache.json"))
        super().__init__(**kw)
        # TARGET (the site/orbiter being diagnosed) -> recent diagnoses of it.
        # Shared across nodes; see the docstring's honesty caveat.
        self._board: dict[str, list[dict]] = {}

    # -- board --------------------------------------------------------------

    def _post(self, target: str, node: str, cause: Cause, conf: float, t: int) -> None:
        """Publish a diagnosis and drop everything that fell out of the window."""
        entries = self._board.setdefault(target, [])
        entries.append({"node": node, "cause": cause.value,
                        "conf": round(float(conf), 4), "t": int(t)})
        # Decide events are non-decreasing in t, so pruning against the newest
        # entry is safe and keeps the board bounded over a long horizon.
        self._board[target] = [e for e in entries if t - e["t"] <= PEER_WINDOW_MS]

    def _peers_for(self, target: str, node: str, t: int) -> list[dict]:
        """Recent diagnoses of `target` made by nodes OTHER than `node`.

        One entry per peer, its latest. A peer that re-diagnosed the same target
        three times has not become three witnesses -- counting it that way is
        the same incest the (origin, seq) dedup exists to prevent.
        """
        latest: dict[str, dict] = {}
        for e in self._board.get(target, ()):
            if e["node"] == node or not (0 <= t - e["t"] <= PEER_WINDOW_MS):
                continue
            prev = latest.get(e["node"])
            if prev is None or e["t"] >= prev["t"]:
                latest[e["node"]] = e
        peers = sorted(latest.values(), key=lambda e: (-e["t"], e["node"]))
        return peers[:MAX_PEERS]

    # -- prompt -------------------------------------------------------------

    def _peer_block(self, peers: list[dict], t: int) -> str:
        lines = []
        for e in peers:
            letter = _CAUSE_TO_LETTER.get(Cause(e["cause"]), "?")
            age = max(0, (t - e["t"]) // 60_000)
            when = "just now" if age == 0 else f"{age} min ago"
            lines.append(f"- {e['node']} concluded {letter} "
                         f"({_LABELS.get(letter, letter)}), "
                         f"confidence {e['conf']:.2f}, {when}")
        return (
            "PEER REPORTS on this same target (other orbiters that diagnosed it "
            "recently):\n"
            + "\n".join(lines) + "\n"
            "Reconcile with your own reading. If a peer's conclusion is "
            "incompatible with\nyours and the evidence cannot settle which is "
            "right, say so and lower your\nconfidence rather than insisting.\n\n"
        )

    def _prompt_with_peers(self, obs, peers: list[dict], t: int) -> str:
        """Parent's prompt with the peer block spliced in before the reply
        instruction. With no peers it is byte-identical to the parent's, which
        is what keeps the two arms comparable."""
        base = super()._prompt(obs)
        if not peers:
            return base
        marker = "Reply with ONLY"
        i = base.find(marker)
        if i == -1:
            raise RuntimeError(
                "gemma_peer: could not find the reply instruction in the "
                "reasoning-first prompt. gemma_cot._REASONING_INSTRUCTION "
                "changed; update this splice rather than shipping a prompt "
                "that is no longer comparable.")
        return base[:i] + self._peer_block(peers, t) + base[i:]

    # -- output -------------------------------------------------------------

    def decide(self, node: str, t: int) -> Policy:
        adverse = [(p, o) for (n, p), o in self._last_obs.items()
                   if n == node and o.outcome is not Outcome.OK]
        if not adverse:
            # Nothing adverse -> no diagnosis was made, so there is nothing to
            # post to the board and no LLM call to spend.
            return Policy(nominal_belief(), Action.NONE, "", t, 1.0, "")

        target, obs = max(adverse, key=lambda kv: kv[1].t)
        peers = self._peers_for(target, node, t)
        prompt = self._prompt_with_peers(obs, peers, t)
        # Inherited call path (and cache-key prefix) -- a failure here prints
        # the parent's [gemma_cot] tag; the cache FILE is this arm's own.
        parsed, raw = self._call_raw(prompt)

        if parsed is None:
            b = nominal_belief()
            return Policy(b, bayes_action(b)[0], target, t, 0.0,
                          "[gemma_peer unavailable]")

        cause = _LETTER_TO_CAUSE.get(
            str(parsed.get("cause", "")).strip()[:1].upper(), Cause.NOMINAL)
        conf = min(0.98, max(0.2, float(parsed.get("confidence", 0.6))))

        # --- cross-check ---------------------------------------------------
        # One vote per diagnostician: this node, plus each peer's latest. Votes
        # are summed per cause. Nothing is multiplied and no peer OBSERVATION
        # enters -- only the conclusion it produced.
        votes: dict[Cause, float] = {cause: conf}
        for e in peers:
            pc = Cause(e["cause"])
            votes[pc] = votes.get(pc, 0.0) + float(e["conf"])

        consensus = len(votes) == 1          # true when no peers, or all agreed
        if consensus:
            belief = _belief_from(cause, conf)
            reported = conf
        else:
            belief = _consensus_belief(votes)
            # The advisor's confidence IS the top of its belief now. Reporting
            # the model's own unmoved number next to a split belief would be
            # the overconfidence this arm exists to expose.
            reported = max(belief.values())

        action, _ = bayes_action(belief)

        steps = parsed.get("steps") or []
        if isinstance(steps, str):
            steps = [steps]
        steps = [str(s)[:400] for s in steps][:12]

        rationale = str(parsed.get("rationale", ""))
        if not consensus:
            others = ", ".join(sorted({e["node"] for e in peers
                                       if e["cause"] != cause.value}))
            rationale = f"[peers disagree: {others}] {rationale}"

        self.records[(node, t)] = {
            "peer": target, "prompt": prompt, "raw": raw, "steps": steps,
            "peers": [dict(e) for e in peers],
            "consensus": consensus,
        }

        # Post AFTER deciding, so a node never reads its own current conclusion
        # back as a peer report. What is posted is this node's OWN reading --
        # `cause`/`conf` as the model gave them, never the consensus belief. A
        # consensus is already an aggregate of other nodes' conclusions;
        # republishing it would let one opinion re-enter the pool wearing a
        # different node's name, which is the belief-forwarding incest the
        # gossip layer is built to avoid.
        self._post(target, node, cause, conf, t)

        return Policy(
            belief=belief, action=action, target=target,
            until=t + 30 * 60_000, confidence=reported,
            rationale=rationale[:160],
        )
