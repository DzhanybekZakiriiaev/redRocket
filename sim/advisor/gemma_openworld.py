"""Open-world Gemma advisor -- the ONE thing the Bayes filter structurally cannot do.

The Bayes filter in `bayes.py` has a fixed likelihood row per cause. Its posterior
is a distribution over exactly six hardcoded causes; it has no mass to place
anywhere else, so a fault outside those six is *forced* to be re-expressed as the
nearest hardcoded bin. It cannot say "this is none of the above".

This variant hands Gemma one extra option the closed model never had:

    G. UNKNOWN / ANOMALY -- evidence fits no known signature; flag for operator
       review.

Everything else -- the evidence framing, the co-failure features, the caching,
the .env loading -- is inherited unchanged from `GemmaAdvisor`, so the ONLY
difference between the two arms is whether "none of the above" is on the menu.
When a novel fault (RFI/uplink jamming, an SEU frame-corruption event) presents
evidence that matches no known tell, the open-world model can pick G and reason
about it, while the Bayes filter confidently names a wrong known cause.

Letter G maps to the sentinel string "unknown" -- deliberately NOT a Cause enum
value. The absence of a Cause for it is the point: the simulator was never told
this fault exists.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request

from ..types import Observation, Policy, Action, Cause, Outcome
from .base import bayes_action, nominal_belief, uniform_belief
from .gemma import (
    GemmaAdvisor, _CAUSE_MENU, _bucket_silence, _LETTER_TO_CAUSE,
)

# The sentinel truth-label for a fault with no Cause enum value.
UNKNOWN = "unknown"

# One more letter than the closed model. G is the escape hatch.
LETTERS_OPEN = "ABCDEFG"

_MENU_OPEN = _CAUSE_MENU + (
    "\nG. UNKNOWN / ANOMALY - evidence fits NO known signature above (e.g. the\n"
    "   outcome is adverse yet the beacon is healthy, no peer is co-failing, the\n"
    "   queue is shallow, and there is no time shift -- so B-F each have their\n"
    "   defining tell ABSENT). Pick this rather than force-fitting a known cause;\n"
    "   recommend flagging the link for operator review."
)

# Letter -> cause map with the escape hatch. G is a plain string, NOT a Cause.
_LETTER_TO_CAUSE_OPEN: dict[str, object] = {**_LETTER_TO_CAUSE, "G": UNKNOWN}


def _parse_open(content: str) -> dict | None:
    """Like gemma._parse, but the bare-letter fallback also accepts G."""
    i, j = content.find("{"), content.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            obj = json.loads(content[i:j + 1])
            if isinstance(obj, dict) and "cause" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    for ch in content.strip():
        if ch.upper() in LETTERS_OPEN:
            return {"cause": ch.upper(), "confidence": 0.6, "rationale": ""}
    return None


class OpenWorldGemmaAdvisor(GemmaAdvisor):
    """GemmaAdvisor with an open-world 'UNKNOWN / ANOMALY' option (letter G)."""

    name = "gemma_openworld"

    # -- prompt (parent's, with the extended menu and an A-G answer space) --

    def _prompt(self, obs: Observation) -> str:
        self_cf = self._self_cofail(obs)
        peer_cf = self._peer_cofail(obs)
        beacon = self._beacon_ok(obs)

        def tri(x: bool | None) -> str:
            return "unknown" if x is None else ("yes" if x else "no")

        rate = ("silent" if obs.measured_rate == 0
                else f"{obs.measured_rate}/{obs.expected_rate} bps")
        evidence = "\n".join([
            f"- observed outcome: {obs.outcome.value.upper()} ({rate})",
            f"- elapsed silence on this link: {_bucket_silence(obs.ms_since_ok)}",
            f"- other links of THIS orbiter failing now (self co-failure): {tri(self_cf)}",
            f"- OTHER orbiters failing at this SAME peer (gossiped): {tri(peer_cf)}",
            f"- low-gain beacon healthy: {tri(beacon)}",
            f"- local send queue deep: {'yes' if obs.queue_bytes >= 1_000_000 else 'no'}",
            f"- carrier seen OUTSIDE the scheduled window (time shift): "
            f"{'yes' if obs.shift_observed else 'no'}",
            f"- concurrent contacts: this orbiter {obs.self_degree}, peer {obs.peer_degree}",
        ])
        return (
            "You are the onboard fault-diagnosis model on a Mars relay orbiter. "
            "A scheduled contact under-performed. From the local evidence alone, "
            "name the single most likely cause. If the evidence does not match any "
            "of the known signatures B-F, do NOT force-fit one -- choose G, and "
            "then use your engineering knowledge to HYPOTHESIZE what the fault "
            "actually is (e.g. external RF interference/jamming, a radiation-induced "
            "single-event upset, a thermal anomaly) even though it is not on the "
            "menu.\n\n"
            f"CAUSES:\n{_MENU_OPEN}\n\n"
            f"EVIDENCE for orbiter {obs.node} <-> {obs.peer}:\n{evidence}\n\n"
            'Reply with ONLY a JSON object, no prose:\n'
            '{"cause": "<one letter A-G>", "confidence": <0..1>, '
            '"hypothesis": "<if G: your best guess of the actual fault, a few '
            'words; otherwise empty>", "rationale": "<one short sentence>"}'
        )

    # -- LLM call (parent's, but tolerant of a bare 'G' in the fallback) ----

    def _call(self, prompt: str) -> dict | None:
        ck = hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()
        if ck in self._cache:
            return self._cache[ck]

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 200,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                resp = json.loads(r.read().decode())
            content = resp["choices"][0]["message"]["content"]
            parsed = _parse_open(content)
            if parsed is None:
                raise ValueError(f"could not parse cause from: {content!r}")
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
            if self.raise_on_error:
                raise
            if not self._warned:
                print(f"[gemma_openworld] call failed ({e}); falling back to "
                      f"NOMINAL for failed calls. Fix .env / endpoint and rerun.",
                      file=sys.stderr)
                self._warned = True
            return None

        self._cache[ck] = parsed
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except Exception:
            pass
        return parsed

    # -- output (handles the 'unknown' sentinel, which has no Cause) --------

    def diagnose(self, obs: Observation) -> dict:
        """Convenience for the demo: run prompt+call on ONE observation and
        return {letter, cause, confidence, rationale, flagged}. `cause` is a
        Cause value string, or the sentinel 'unknown' for letter G."""
        parsed = self._call(self._prompt(obs))
        if parsed is None:
            return {"letter": "?", "cause": Cause.NOMINAL.value, "confidence": 0.0,
                    "hypothesis": "", "rationale": "[gemma unavailable]",
                    "flagged": False}
        letter = str(parsed.get("cause", "")).strip()[:1].upper()
        mapped = _LETTER_TO_CAUSE_OPEN.get(letter, Cause.NOMINAL)
        cause_val = mapped if isinstance(mapped, str) else mapped.value
        return {
            "letter": letter,
            "cause": cause_val,
            "confidence": float(parsed.get("confidence", 0.6)),
            # The LLM's free-text guess at the ACTUAL fault when it flags G --
            # this is the open-world payoff: it doesn't just reject the known
            # set, it names what the novel fault probably is.
            "hypothesis": str(parsed.get("hypothesis", ""))[:160],
            "rationale": str(parsed.get("rationale", ""))[:200],
            "flagged": cause_val == UNKNOWN,
        }

    def decide(self, node: str, t: int) -> Policy:
        adverse = [(p, o) for (n, p), o in self._last_obs.items()
                   if n == node and o.outcome is not Outcome.OK]
        if not adverse:
            b = nominal_belief()
            return Policy(b, Action.NONE, "", t, 1.0, "")

        peer, obs = max(adverse, key=lambda kv: kv[1].t)
        d = self.diagnose(obs)
        conf = min(0.98, max(0.2, d["confidence"]))

        if d["flagged"]:
            # No Cause enum value exists for this -- spread belief uniformly over
            # the known causes (the honest "I can't localize this to a known bin")
            # and take no committing action beyond escalating to an operator.
            belief = uniform_belief()
            return Policy(
                belief=belief, action=Action.NONE, target=peer,
                until=t + 30 * 60_000, confidence=conf,
                rationale="ANOMALY - fits no known signature; flag for operator "
                          "review. " + d["rationale"],
            )

        from .gemma import _belief_from
        cause = Cause(d["cause"])
        belief = _belief_from(cause, conf)
        action, _ = bayes_action(belief)
        return Policy(belief=belief, action=action, target=peer,
                      until=t + 30 * 60_000, confidence=conf,
                      rationale=d["rationale"])
