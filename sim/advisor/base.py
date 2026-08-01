"""The swap point. PLAN.md section 5.4.

Three implementations sit behind this one interface: no diagnosis, a Bayes
filter, and Gemma. The comparison between them is only meaningful if they are
genuinely interchangeable -- same observations in, same policy out, same code
path around them.

`NullAdvisor` must run the identical path: it receives every observation and
every piece of evidence, it is simply asked to conclude nothing from them. If
the null case skipped observation construction, the arms would differ in more
than the advisor and every number would be contaminated.

An advisor NEVER sees FaultTrace, the engine, the contact plan beyond what
arrives in an Observation, or anything else that constitutes ground truth.
`tests` enforces this.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Observation, Evidence, Policy, Action, Cause, CAUSES


def uniform_belief() -> dict[str, float]:
    p = 1.0 / len(CAUSES)
    return {c.value: p for c in CAUSES}


def nominal_belief() -> dict[str, float]:
    """Prior for a link with no adverse observation yet. Not uniform -- most
    links are fine most of the time, and a filter that starts at uniform spends
    its first several updates just relearning that."""
    b = {c.value: 0.02 for c in CAUSES}
    b[Cause.NOMINAL.value] = 1.0 - 0.02 * (len(CAUSES) - 1)
    return b


@runtime_checkable
class Advisor(Protocol):
    name: str

    def observe(self, obs: Observation) -> None:
        """Record a local observation. Called once per contact outcome."""
        ...

    def receive(self, evidence: list[Evidence]) -> None:
        """Absorb evidence gossiped from a peer. Already deduplicated by
        (origin, seq) upstream -- an advisor must not assume otherwise, but it
        also must not re-derive confidence from repeats."""
        ...

    def decide(self, node: str, t: int) -> Policy:
        """Current belief and chosen action for this node."""
        ...


# Cost of taking action A when the true cause is C. Rows = cause, cols = action.
# This matrix is what makes an action "correct" -- and using the same one for
# every advisor is what makes beating the Bayes filter mean something, since
# the filter's decision rule is provably optimal against it.
#
# The asymmetries are the point: blacklisting a link that was only rained on
# discards a working asset for nothing, while waiting on a dead node stalls
# traffic until the horizon.
COST: dict[Cause, dict[Action, float]] = {
    Cause.NOMINAL:     {Action.NONE: 0.0, Action.WAIT: 0.3, Action.REROUTE: 1.0,
                        Action.THROTTLE: 0.8, Action.BLACKLIST: 4.0},
    Cause.WEATHER:     {Action.NONE: 1.5, Action.WAIT: 0.0, Action.REROUTE: 0.6,
                        Action.THROTTLE: 1.2, Action.BLACKLIST: 5.0},
    Cause.NODE_DOWN:   {Action.NONE: 3.0, Action.WAIT: 4.0, Action.REROUTE: 0.8,
                        Action.THROTTLE: 3.5, Action.BLACKLIST: 0.0},
    Cause.POINTING:    {Action.NONE: 2.0, Action.WAIT: 1.5, Action.REROUTE: 0.0,
                        Action.THROTTLE: 2.0, Action.BLACKLIST: 2.5},
    Cause.BUFFER:      {Action.NONE: 2.0, Action.WAIT: 1.0, Action.REROUTE: 1.2,
                        Action.THROTTLE: 0.0, Action.BLACKLIST: 3.0},
    Cause.STALE_SCHED: {Action.NONE: 1.5, Action.WAIT: 0.5, Action.REROUTE: 1.0,
                        Action.THROTTLE: 2.0, Action.BLACKLIST: 3.0},
}

ACTIONS: tuple[Action, ...] = (Action.NONE, Action.WAIT, Action.REROUTE,
                               Action.THROTTLE, Action.BLACKLIST)


def bayes_action(belief: dict[str, float]) -> tuple[Action, float]:
    """Minimum expected cost under COST. Returns (action, expected_cost).

    Decision-theoretically optimal given the belief, which is exactly why it is
    the right rule for every advisor to share: differences in outcome then come
    from differences in *belief quality*, not from differences in how beliefs
    were turned into actions.
    """
    best, best_cost = Action.NONE, float("inf")
    for a in ACTIONS:
        c = sum(belief.get(cause.value, 0.0) * COST[cause][a] for cause in COST)
        if c < best_cost:
            best, best_cost = a, c
    return best, best_cost
