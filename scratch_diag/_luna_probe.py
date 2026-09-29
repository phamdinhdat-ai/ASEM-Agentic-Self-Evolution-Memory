"""Verify the endpoint + model from .env before wiring a new config.

Checks:
  1. models.list() — does the pinned model exist?
  2. A trivial completion — does it answer at all?
  3. An ingest-sized extraction — the case that matters, with the same
     body variants that fixed the deepseek truncation.
"""

from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from openai import OpenAI  # noqa: E402


def clean(v: str | None) -> str | None:
    if v is None:
        return None
    return v.strip().strip('"').strip("'")


BASE_URL = clean(os.environ.get("OPENAI_BASE_URL"))
API_KEY = clean(os.environ.get("OPENAI_API_KEY"))
MODEL = sys.argv[1] if len(sys.argv) > 1 else clean(os.environ.get("LLM_MODEL"))

print(f"base_url = {BASE_URL}")
print(f"model    = {MODEL!r}\n")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)

# --- 1. Model availability ---
try:
    ids = [m.id for m in client.models.list().data]
    print(f"[models] {len(ids)} models available")
    hits = [i for i in ids if re.search(r"luna|gpt-5\.6|5\.6", i, re.I)]
    if hits:
        print(f"[models] 5.6/luna-ish candidates: {hits[:20]}")
    if MODEL not in ids:
        print(f"[models] !! pinned model NOT in list")
        print(f"[models] vgpt/*: {[i for i in ids if i.lower().startswith('vgpt')][:20]}")
except Exception as exc:  # noqa: BLE001
    print(f"[models] ERROR: {type(exc).__name__}: {exc}")

# --- 2. Trivial call ---
def probe(label: str, prompt: str, **body) -> None:
    params = {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}
    params.update(body)
    t0 = time.time()
    try:
        r = client.chat.completions.create(**params)
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] ERROR after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
        return
    dt = time.time() - t0
    ch = (r.choices or [None])[0]
    if ch is None:
        print(f"[{label}] no choices ({dt:.1f}s)")
        return
    msg = ch.message
    content = msg.content
    reasoning = getattr(msg, "reasoning_content", None)
    usage = getattr(r, "usage", None)
    print(f"[{label}] {dt:5.1f}s finish={ch.finish_reason!r} "
          f"content_len={len(content) if isinstance(content, str) else repr(content)[:50]} "
          f"reasoning_len={len(reasoning) if isinstance(reasoning, str) else None} "
          f"completion_tok={getattr(usage, 'completion_tokens', None)}")
    if isinstance(content, str) and content.strip():
        print(f"          head: {content.strip()[:100]!r}")


probe("triv", "Reply with exactly: ok", max_tokens=32)

# --- 3. Ingest-sized extraction (the real workload) ---
TURNS = [
    f"[Caroline] On 2023-05-0{(i % 9) + 1}, I "
    + ("went to the beach, walked along the pier and bought a hat from a street vendor"
       if i % 3 == 0 else
       "had coffee with Melanie at the corner cafe and talked about her grandmother in Sweden for an hour"
       if i % 3 == 1 else
       "called my brother and we planned a trip to the mountains next month")
    for i in range(1, 19)
]
PROMPT = (
    "Convert ONE dialogue session into standalone atomic factual notes.\n"
    "Resolve pronouns to names and relative time to absolute dates.\n"
    "ONE FACT PER NOTE - emit a separate object for EVERY distinct event.\n"
    "Return a JSON array of objects with keys: fact, subject, predicate, object, "
    "entities, keywords, tags, speaker.\n"
    "Output ONLY the JSON array.\n\n"
    "SESSION DATE: 2023-05-20\n\nDIALOGUE:\n" + "\n".join(TURNS)
)

probe("ingest 4096 plain", PROMPT, temperature=0.3, max_tokens=4096)
probe("ingest 4096 no-think", PROMPT, temperature=0.3, max_tokens=4096,
      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
