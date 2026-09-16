"""Measure the effect of the thinking controls on real extraction-shaped calls.

Compares, through the new OpenAIBackend:
  * default (thinking on)
  * thinking: {type: disabled}
  * reasoning_effort: low

Reporting wall-clock latency, output size, JSON validity, and total tokens.
"""

from __future__ import annotations

import json
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

with open(os.path.join(_PROJECT_ROOT, ".env"), encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from asem.backends import build_backend  # noqa: E402

DIALOGUE = "\n".join([
    "[Caroline] I went to the LGBTQ support group yesterday.",
    "[Melanie] That sounds great. How was it?",
    "[Caroline] Really helpful. I am planning to go camping next month with my sister.",
    "[Melanie] Nice. I have been reading a lot about adoption lately.",
    "[Caroline] Me too. I adopted a puppy last week, she is a golden retriever.",
    "[Melanie] Congratulations! What did you name her?",
    "[Caroline] Luna. My sister named her actually.",
    "[Melanie] I am thinking of running a marathon in the autumn.",
    "[Caroline] That is a big goal. I ran one in 2022.",
    "[Melanie] I know, you told me. I get married next June.",
])

PROMPT = (
    "Extract every distinct atomic fact from this dialogue. Resolve relative dates against "
    "the session date 8 May 2023. Reply with ONLY a JSON array of objects with keys "
    '"fact", "keywords", "tags".\n\nSession date: 8 May 2023\n\n' + DIALOGUE
)

BASE = {
    "backend": "openai",
    "openai": {
        "model": "deepseek-v4-flash",
        "temperature": 0.0,
        "max_tokens": 4096,
        "base_url": os.environ["OPENAI_BASE_URL"],
        "api_key": os.environ["OPENAI_API_KEY"],
        "embedder_provider": "huggingface",
        "embedder_name": "sentence-transformers/all-MiniLM-L6-v2",
    },
}


def _valid_json_array(text: str) -> bool:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return False
    try:
        return isinstance(json.loads(text[start:end + 1]), list)
    except json.JSONDecodeError:
        return False


def run(label: str, extra: dict) -> None:
    cfg = json.loads(json.dumps(BASE))
    cfg["openai"].update(extra)
    backend = build_backend(cfg)
    t0 = time.perf_counter()
    out = (backend.generate(PROMPT) or "").strip()
    elapsed = time.perf_counter() - t0
    print(f"{label:<24}{elapsed:>7.1f}s  chars={len(out):>6}  "
          f"valid_json={_valid_json_array(out)}  total_tokens={backend.token_count}")


print(f"prompt chars: {len(PROMPT)}\n")
print(f"{'variant':<24}{'sec':>8}{'chars':>14}  {'':<10} tokens")
print("-" * 78)
run("thinking: default", {})
run("thinking: disabled", {"thinking": {"type": "disabled"}})
run("reasoning_effort=low", {"reasoning_effort": "low"})
