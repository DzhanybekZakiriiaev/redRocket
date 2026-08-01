"""Head-to-head diagnosis accuracy: single-shot 'gemma' vs 'gemma_plus'.

    python -m tools.eval_gemma_plus

Runs the SAME default scenario through the SAME Engine for each advisor and
scores diagnosis accuracy on FAULTED samples only:

  over the engine's belief records, take every sample whose ground-truth cause
  is != 'nominal'; it is correct when the belief argmax == truth.

Prints one line, e.g.:

    gemma 64.8%  gemma_plus 71.2%

Only the advisor differs between the two runs; contacts, faults, decision grid
and the shared bayes_action rule are identical, so the gap is a belief-quality
gap and nothing else. No skyfield positions are computed here -- we only need
eng.beliefs, so this is much faster than a full trace build.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sim.config import DEFAULT, ScenarioConfig      # noqa: E402
from sim import contacts as C                        # noqa: E402
from sim import faults as F                          # noqa: E402
from sim.engine import Engine                        # noqa: E402


def _run(advisor, cfg: ScenarioConfig) -> list[dict]:
    cs = C.read(cfg)
    fs = F.generate(cs, cfg)
    eng = Engine(cs, fs, advisor, cfg)
    eng.run()
    return eng.beliefs


def _accuracy(beliefs: list[dict]) -> tuple[float, int, int]:
    faulted = [b for b in beliefs if b["truth"] != "nominal"]
    if not faulted:
        return 0.0, 0, 0
    correct = sum(1 for b in faulted if b["argmax"] == b["truth"])
    return correct / len(faulted), correct, len(faulted)


def main() -> int:
    cfg = DEFAULT

    from sim.advisor.gemma import GemmaAdvisor
    from sim.advisor.gemma_plus import GemmaPlusAdvisor

    results: dict[str, tuple[float, int, int]] = {}
    for name, advisor in (("gemma", GemmaAdvisor()),
                          ("gemma_plus", GemmaPlusAdvisor())):
        beliefs = _run(advisor, cfg)
        results[name] = _accuracy(beliefs)

    g, gp = results["gemma"], results["gemma_plus"]
    print(f"gemma {g[0]*100:.1f}%  gemma_plus {gp[0]*100:.1f}%")
    print(f"  (gemma {g[1]}/{g[2]} faulted samples correct; "
          f"gemma_plus {gp[1]}/{gp[2]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
