"""Smoke-test the Gemma endpoint on Spur's compute BEFORE wiring it into a run.

    python -m tools.gemma_smoke

Checks, in order:
  1. .env has the three vars (key redacted in output).
  2. The OpenAI-compatible endpoint answers a trivial prompt.
  3. A real diagnosis-style prompt returns a parseable cause.

If step 2 fails it prints the HTTP error so you can fix BASE-URL / MODEL / key.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sim.advisor.gemma import (  # noqa: E402
    ENV_KEY, ENV_BASE_URL, ENV_MODEL, _load_env, _parse,
)


def _chat(base_url: str, key: str, model: str, prompt: str, timeout: float = 30.0):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 200,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    env = _load_env(_ROOT / ".env")
    import os
    key = os.environ.get(ENV_KEY) or env.get(ENV_KEY, "")
    base = os.environ.get(ENV_BASE_URL) or env.get(ENV_BASE_URL, "")
    model = os.environ.get(ENV_MODEL) or env.get(ENV_MODEL, "")

    print("== config (.env) ==")
    print(f"  {ENV_KEY:22s} = {'set (' + str(len(key)) + ' chars)' if key else 'MISSING'}")
    print(f"  {ENV_BASE_URL:22s} = {base or 'MISSING'}")
    print(f"  {ENV_MODEL:22s} = {model or 'MISSING'}")
    if not (key and base and model):
        print("\nFIX: add the MISSING line(s) to .env, then rerun.")
        return 1

    print("\n== 1. trivial ping ==")
    try:
        resp = _chat(base, key, model, "Reply with exactly the word: OK")
        content = resp["choices"][0]["message"]["content"]
        print(f"  endpoint answered: {content!r}")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:300]}")
        print("  -> check BASE-URL, MODEL id, and that the key is valid.")
        return 2
    except Exception as e:
        print(f"  FAILED: {e}\n  -> is BASE-URL reachable and OpenAI-compatible?")
        return 2

    print("\n== 2. diagnosis-style prompt ==")
    prompt = (
        "Reply with ONLY JSON: {\"cause\": \"<A-F>\", \"confidence\": <0..1>, "
        "\"rationale\": \"<one sentence>\"}.\n"
        "A=nominal B=dust-storm C=safe-mode D=antenna-mispoint E=recorder-full "
        "F=ephemeris-drift.\n"
        "Evidence: contact SILENT; beacon healthy; no other orbiter failing at "
        "this site; queue shallow. Most likely cause?")
    resp = _chat(base, key, model, prompt)
    content = resp["choices"][0]["message"]["content"]
    parsed = _parse(content)
    print(f"  raw: {content!r}")
    print(f"  parsed: {parsed}")
    if parsed is None:
        print("  -> model replied but not parseably; the advisor's fallback will "
              "handle it, but check the model follows the JSON instruction.")
        return 3

    print("\nAll good. Run the Gemma arm with:\n"
          "  python -c \"from sim.trace import write; write(name='trace_gemma.json')\" "
          "  # after setting advisor_name='gemma'\n"
          "or add a --advisor flag (see the note printed by the assistant).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
