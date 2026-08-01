"""Gemma advisor, reasoning-first.

The baseline arm asks for `{"cause", "confidence", "rationale"}` in that order.
Because a language model emits JSON keys in template order, the cause token is
sampled BEFORE the rationale -- so the rationale cannot have influenced it. It
is a justification written after the fact, not the reasoning that produced the
answer. That is fine for a caption and misleading as an explanation.

This arm moves the reasoning in front:

    {"steps": [...], "cause": ..., "confidence": ..., "rationale": ...}

Now the elimination steps are sampled first and sit in the context when the
cause token is chosen, so they genuinely condition it. Same idea as thinking
out loud before answering.

Deliberately identical everywhere else: the evidence block is taken verbatim
from GemmaAdvisor._prompt, so the two arms see exactly the same bullets and the
only variable is the output contract. Anything the comparison shows is
attributable to the reasoning step and nothing else.

Its own cache file, so it can never be confused with the baseline's responses.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ..types import Action, Cause, Outcome, Policy
from .base import bayes_action, nominal_belief
from .gemma import (
    GemmaAdvisor, _LETTER_TO_CAUSE, _belief_from, _parse, _REPO_ROOT,
)

# Reasoning needs room; the baseline's 200 would truncate mid-array and the
# JSON would fail to parse.
MAX_TOKENS = 700


_REASONING_INSTRUCTION = (
    'Work the causes one at a time before you answer. Each step must cite a '
    'specific piece of the evidence above and say which cause it supports or '
    'rules out. Do not name the final cause until the steps are done.\n\n'
    'Reply with ONLY a JSON object, no prose, with the keys in this order:\n'
    '{"steps": ["<one elimination step>", "<another>", "..."], '
    '"cause": "<one letter A-F>", "confidence": <0..1>, '
    '"rationale": "<one short sentence>"}'
)


class GemmaCotAdvisor(GemmaAdvisor):
    """Same evidence, same decision rule -- reasoning sampled before the cause."""

    name = "gemma_cot"

    def __init__(self, **kw) -> None:
        kw.setdefault("cache_path", str(_REPO_ROOT / "out" / "gemma_cot_cache.json"))
        super().__init__(**kw)
        # (node, t) -> what was sent and what came back, for the trace.
        self.records: dict[tuple[str, int], dict] = {}

    # -- prompt -------------------------------------------------------------

    def _prompt(self, obs) -> str:
        """Parent's prompt with only the output contract swapped.

        Cutting at the parent's reply instruction keeps the framing, the cause
        menu and every evidence bullet byte-identical to the baseline arm. If
        the marker ever moves, fail loudly rather than silently sending a
        prompt that is no longer comparable.
        """
        base = super()._prompt(obs)
        marker = "Reply with ONLY"
        i = base.find(marker)
        if i == -1:
            raise RuntimeError(
                "gemma_cot: could not find the reply instruction in the base "
                "prompt. GemmaAdvisor._prompt changed; update this splice "
                "rather than shipping a prompt that is no longer comparable.")
        return base[:i] + _REASONING_INSTRUCTION

    # -- call ---------------------------------------------------------------

    def _call_raw(self, prompt: str) -> tuple[dict | None, str]:
        """Like GemmaAdvisor._call but returns the raw reply text too, and
        allows enough tokens for the reasoning array."""
        ck = hashlib.sha256(f"cot\n{self.model}\n{prompt}".encode()).hexdigest()
        hit = self._cache.get(ck)
        if hit is not None:
            return hit.get("parsed"), hit.get("raw", "")

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
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
            parsed = _parse(content)
            if parsed is None:
                raise ValueError(f"could not parse cause from: {content!r}")
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
            if self.raise_on_error:
                raise
            if not self._warned:
                print(f"[gemma_cot] call failed ({e}); falling back to NOMINAL. "
                      f"Fix .env / endpoint and rerun.", file=sys.stderr)
                self._warned = True
            return None, ""

        self._cache[ck] = {"parsed": parsed, "raw": content}
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except Exception:
            pass
        return parsed, content

    # -- output -------------------------------------------------------------

    def decide(self, node: str, t: int) -> Policy:
        adverse = [(p, o) for (n, p), o in self._last_obs.items()
                   if n == node and o.outcome is not Outcome.OK]
        if not adverse:
            return Policy(nominal_belief(), Action.NONE, "", t, 1.0, "")

        peer, obs = max(adverse, key=lambda kv: kv[1].t)
        prompt = self._prompt(obs)
        parsed, raw = self._call_raw(prompt)

        if parsed is None:
            b = nominal_belief()
            return Policy(b, bayes_action(b)[0], peer, t, 0.0, "[gemma_cot unavailable]")

        cause = _LETTER_TO_CAUSE.get(
            str(parsed.get("cause", "")).strip()[:1].upper(), Cause.NOMINAL)
        conf = min(0.98, max(0.2, float(parsed.get("confidence", 0.6))))
        belief = _belief_from(cause, conf)
        action, _ = bayes_action(belief)

        steps = parsed.get("steps") or []
        if isinstance(steps, str):
            steps = [steps]
        steps = [str(s)[:400] for s in steps][:12]

        self.records[(node, t)] = {
            "peer": peer, "prompt": prompt, "raw": raw, "steps": steps,
        }

        return Policy(
            belief=belief, action=action, target=peer,
            until=t + 30 * 60_000, confidence=conf,
            rationale=str(parsed.get("rationale", ""))[:160],
        )
