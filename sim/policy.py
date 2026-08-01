"""Policy enforcement. PLAN.md section 5.6.

The ONLY path from a diagnosis to a routing decision. Nothing else in the
simulator reads advisor output. That is what keeps the comparison between
advisors honest -- if a belief could influence behaviour through some other
channel, the arms would differ in ways the experiment cannot attribute.

Policy is an overlay on routing, never a mutation of the contact plan. The plan
is ground truth about what the network *can* do; policy is belief about what it
*should* do. Conflating them destroys the ability to measure diagnosis quality,
because the thing being diagnosed would change as a result of the diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Policy, Action


@dataclass
class Constraints:
    """What routing is allowed to do right now, per node."""
    excluded: dict[str, int] = field(default_factory=dict)     # peer -> until_ms
    throttled: dict[str, int] = field(default_factory=dict)    # peer -> until_ms
    holding: dict[str, int] = field(default_factory=dict)      # peer -> until_ms

    def _live(self, d: dict[str, int], peer: str, t: int) -> bool:
        until = d.get(peer)
        if until is None:
            return False
        if t >= until:
            del d[peer]
            return False
        return True

    def is_excluded(self, peer: str, t: int) -> bool:
        return self._live(self.excluded, peer, t)

    def is_throttled(self, peer: str, t: int) -> bool:
        return self._live(self.throttled, peer, t)

    def is_holding(self, peer: str, t: int) -> bool:
        return self._live(self.holding, peer, t)


class PolicyEnforcer:
    # A blacklist should outlive a reroute -- it expresses "this peer is gone",
    # not "try elsewhere for now".
    BLACKLIST_MS = 6 * 3600 * 1000
    REROUTE_MS = 45 * 60 * 1000
    THROTTLE_MS = 30 * 60 * 1000
    WAIT_MS = 20 * 60 * 1000
    THROTTLE_FACTOR = 0.25

    def __init__(self) -> None:
        self.by_node: dict[str, Constraints] = {}
        self.applied: int = 0

    def constraints(self, node: str) -> Constraints:
        return self.by_node.setdefault(node, Constraints())

    def apply(self, node: str, policy: Policy, t: int) -> None:
        if policy.action is Action.NONE or not policy.target:
            return
        c = self.constraints(node)
        peer = policy.target
        if policy.action is Action.BLACKLIST:
            c.excluded[peer] = t + self.BLACKLIST_MS
        elif policy.action is Action.REROUTE:
            c.excluded[peer] = t + self.REROUTE_MS
        elif policy.action is Action.THROTTLE:
            c.throttled[peer] = t + self.THROTTLE_MS
        elif policy.action is Action.WAIT:
            c.holding[peer] = t + self.WAIT_MS
        self.applied += 1
