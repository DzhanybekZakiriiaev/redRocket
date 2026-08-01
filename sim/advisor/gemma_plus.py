"""Gemma-PLUS advisor -- the LLM arm with two upgrades over GemmaAdvisor.

Same swap point, same interface (observe/receive/decide -> Policy), same
evidence, same co-failure/beacon/queue features, and the SAME shared
`bayes_action` decision rule as GemmaAdvisor. The only two differences:

  (a) FEW-SHOT: 5 worked examples are prepended to the prompt, one per
      discriminating case, so the model has concrete templates for how the
      tells map to a cause letter (A-F, consistent with `_CAUSE_MENU`).

  (b) SELF-CONSISTENCY: instead of one greedy (temperature 0) call, the model
      is sampled N=5 times at temperature ~0.7 and the cause is decided by
      MAJORITY VOTE. Confidence = votes_for_winner / N (the winning-vote
      fraction), which is a far better-calibrated signal than the single
      number a greedy call self-reports.

Everything else -- the belief simplex construction, the nominal fallback on an
all-fail, the "don't spend a call when nothing is adverse" short-circuit -- is
inherited unchanged from GemmaAdvisor.

Each of the N samples is cached by (prompt, sample_index), so a rerun replays
the exact same votes and the whole arm stays deterministic. Cache lives in
out/gemma_plus_cache.json (separate from the greedy arm's out/gemma_cache.json).
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections import Counter

from ..types import Cause
from .gemma import (
    GemmaAdvisor, LETTERS, _CAUSE_MENU, _parse, _REPO_ROOT,
)


# --------------------------------------------------------------------------
# Few-shot exemplars. Written in the SAME evidence-bullet format the live
# prompt uses, each ending in the exact JSON shape the model must emit. One
# example per discriminating tell (letters match _CAUSE_MENU):
#   C node_down   -> beacon dead + long silence
#   D pointing    -> beacon OK + no peer co-failure
#   B weather     -> peer co-failure at the same site
#   E buffer      -> deep local queue
#   F stale_sched -> carrier seen outside the window (time shift)
# --------------------------------------------------------------------------

_FEW_SHOT = """\
Here are worked examples. Study how each TELL selects exactly one letter.

EXAMPLE 1 (dead spacecraft bus):
- observed outcome: SILENT (silent)
- elapsed silence on this link: long (>30 min)
- other links of THIS orbiter failing now (self co-failure): unknown
- OTHER orbiters failing at this SAME peer (gossiped): no
- low-gain beacon healthy: no
- local send queue deep: no
- carrier seen OUTSIDE the scheduled window (time shift): no
- concurrent contacts: this orbiter 1, peer 1
ANSWER: {"cause": "C", "confidence": 0.9, "rationale": "Beacon is also silent and the outage is long, so the whole bus is down, not just the payload link."}

EXAMPLE 2 (antenna mispoint on this orbiter):
- observed outcome: DEGRADED (12000/100000 bps)
- elapsed silence on this link: short (<5 min)
- other links of THIS orbiter failing now (self co-failure): no
- OTHER orbiters failing at this SAME peer (gossiped): no
- low-gain beacon healthy: yes
- local send queue deep: no
- carrier seen OUTSIDE the scheduled window (time shift): no
- concurrent contacts: this orbiter 1, peer 2
ANSWER: {"cause": "D", "confidence": 0.85, "rationale": "Beacon is healthy and peers at the same site are fine, so only this orbiter's high-gain pointing is off."}

EXAMPLE 3 (dust storm at the surface site):
- observed outcome: SILENT (silent)
- elapsed silence on this link: medium (5-30 min)
- other links of THIS orbiter failing now (self co-failure): no
- OTHER orbiters failing at this SAME peer (gossiped): yes
- low-gain beacon healthy: unknown
- local send queue deep: no
- carrier seen OUTSIDE the scheduled window (time shift): no
- concurrent contacts: this orbiter 1, peer 3
ANSWER: {"cause": "B", "confidence": 0.85, "rationale": "Other orbiters over the same site are failing at once, which points to weather/dust at the surface, not a per-orbiter fault."}

EXAMPLE 4 (onboard recorder full):
- observed outcome: DEGRADED (30000/100000 bps)
- elapsed silence on this link: short (<5 min)
- other links of THIS orbiter failing now (self co-failure): no
- OTHER orbiters failing at this SAME peer (gossiped): no
- low-gain beacon healthy: yes
- local send queue deep: yes
- carrier seen OUTSIDE the scheduled window (time shift): no
- concurrent contacts: this orbiter 2, peer 1
ANSWER: {"cause": "E", "confidence": 0.8, "rationale": "The link is up but the local send queue is deep, so onboard storage is saturated."}

EXAMPLE 5 (stale sequence / ephemeris drift):
- observed outcome: LATE (0/100000 bps)
- elapsed silence on this link: medium (5-30 min)
- other links of THIS orbiter failing now (self co-failure): no
- OTHER orbiters failing at this SAME peer (gossiped): no
- low-gain beacon healthy: yes
- local send queue deep: no
- carrier seen OUTSIDE the scheduled window (time shift): yes
- concurrent contacts: this orbiter 1, peer 1
ANSWER: {"cause": "F", "confidence": 0.9, "rationale": "The carrier appears shifted outside the scheduled window, so the contact timing/ephemeris is stale, not absent."}

Now diagnose the NEW case below using the same rule."""


class GemmaPlusAdvisor(GemmaAdvisor):
    """GemmaAdvisor + few-shot prompting + self-consistency majority vote."""

    name = "gemma_plus"

    def __init__(self, *, cache_path: str | None = None, timeout_s: float = 30.0,
                 raise_on_error: bool = False, n_samples: int = 5,
                 temperature: float = 0.7):
        # Separate cache file from the greedy arm; absolute so cwd never matters.
        cache_path = cache_path or str(_REPO_ROOT / "out" / "gemma_plus_cache.json")
        super().__init__(cache_path=cache_path, timeout_s=timeout_s,
                         raise_on_error=raise_on_error)
        self.n_samples = n_samples
        self.temperature = temperature

    # -- prompt: inject the few-shot block before the live EVIDENCE ---------

    def _prompt(self, obs) -> str:
        base = super()._prompt(obs)
        marker = "EVIDENCE for orbiter"
        i = base.find(marker)
        if i == -1:                       # defensive: menu format changed
            return base
        return base[:i] + _FEW_SHOT + "\n\n" + base[i:]

    # -- one temperature>0 sample, cached by (prompt, sample_index) --------

    def _sample(self, prompt: str, index: int) -> dict | None:
        ck = hashlib.sha256(f"{self.model}\n{index}\n{prompt}".encode()).hexdigest()
        if ck in self._cache:
            return self._cache[ck]

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
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
            parsed = _parse(content)
            if parsed is None:
                raise ValueError(f"could not parse cause from: {content!r}")
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
            if self.raise_on_error:
                raise
            if not self._warned:
                print(f"[gemma_plus] call failed ({e}); failed samples are "
                      f"dropped from the vote. Fix .env / endpoint and rerun.",
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

    # -- replace the single greedy _call with an N-way majority vote -------
    #
    # Returns the same {"cause","confidence","rationale"} shape GemmaAdvisor's
    # `decide` expects, so the whole belief/bayes_action path is inherited
    # verbatim. Returns None (-> nominal fallback in decide) only when EVERY
    # sample failed, exactly matching GemmaAdvisor's all-fail behaviour.

    def _call(self, prompt: str) -> dict | None:
        votes: list[str] = []
        rationales: dict[str, str] = {}
        for i in range(self.n_samples):
            parsed = self._sample(prompt, i)
            if parsed is None:
                continue                              # dropped from the vote
            letter = str(parsed.get("cause", "")).strip()[:1].upper()
            if letter not in LETTERS:
                continue
            votes.append(letter)
            rationales.setdefault(letter, str(parsed.get("rationale", "")))

        if not votes:                                 # all-fail -> nominal
            return None

        counts = Counter(votes)
        # Deterministic tie-break: most votes, then alphabetical letter.
        winner = min(counts, key=lambda l: (-counts[l], l))
        confidence = counts[winner] / self.n_samples  # winning-vote fraction
        return {
            "cause": winner,
            "confidence": confidence,
            "rationale": rationales.get(winner, ""),
        }
