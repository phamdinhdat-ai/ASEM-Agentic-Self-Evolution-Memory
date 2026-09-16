"""Probe which model names the endpoint in .env will actually serve.

The note-extraction prompts require strict JSON output, so this asks each
candidate for a JSON array and reports the raw content.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_env = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_env):
    with open(_env, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from openai import OpenAI  # noqa: E402

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

PROMPT = (
    "Extract facts as JSON. Reply with ONLY this exact array and nothing else:\n"
    '[{"content": "[Alice] Alice adopted a dog.", "keywords": ["alice", "dog"]}]'
)

print(f"base_url: {os.environ['OPENAI_BASE_URL']}")
try:
    print("served  :", [m.id for m in client.models.list()])
except Exception as exc:  # noqa: BLE001
    print("served  : models.list failed:", exc)

for name in ("deepseek-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-v4-flash"):
    try:
        resp = client.chat.completions.create(
            model=name,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=256,
            temperature=0.0,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        print(f"\n{name}: finish={choice.finish_reason} chars={len(content)}")
        print(f"  content={content[:200]!r}")
        print(f"  usage={resp.usage}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n{name}: FAIL -> {type(exc).__name__}: {str(exc)[:200]}")
