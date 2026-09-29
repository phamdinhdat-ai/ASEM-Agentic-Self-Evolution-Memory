"""Why does deepseek-v4-flash return raw_len=0 + finish_reason=length?

Truncated JSON would give raw_len > 0 (a partial array). raw_len == 0 means
the completion budget was spent BEFORE any visible content was written --
i.e. hidden reasoning tokens consumed the whole max_tokens allowance.

This probes the endpoint directly and prints the usage breakdown, whether
reasoning_content is present, and whether `thinking: {type: disabled}` is
actually honoured.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from openai import OpenAI  # noqa: E402

MODEL = os.environ.get("PROBE_MODEL", "deepseek-v4-flash")
BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)

# A session transcript big enough to trigger the real failure.
TURNS = [
    f"[Caroline] On 2023-05-0{(i % 9) + 1}, I {'went to the beach, walked along the pier and bought a hat from a street vendor' if i % 3 == 0 else ('had coffee with Melanie at the corner cafe and talked about her grandmother in Sweden for an hour' if i % 3 == 1 else 'called my brother and we planned a trip to the mountains next month')}"
    for i in range(1, 19)
]
DIALOGUE = "\n".join(TURNS)

PROMPT = (
    "Convert ONE dialogue session into standalone atomic factual notes.\n"
    "Resolve pronouns and relative time to absolute dates.\n"
    "ONE FACT PER NOTE. Emit a separate object for EVERY distinct event.\n"
    "Output ONLY a JSON array of objects with keys: fact, subject, predicate, "
    "object, entities, keywords, tags, speaker.\n\n"
    f"SESSION DATE: 2023-05-20\n\nDIALOGUE:\n{DIALOGUE}"
)


def probe(label: str, **body) -> None:
    params = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.3,
    }
    params.update(body)
    t0 = time.time()
    try:
        r = client.chat.completions.create(**params)
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] ERROR after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
        return

    dt = time.time() - t0
    choice = (r.choices or [None])[0]
    if choice is None:
        print(f"[{label}] no choices ({dt:.1f}s)")
        return
    msg = choice.message
    content = msg.content
    reasoning = getattr(msg, "reasoning_content", None)
    usage = getattr(r, "usage", None)
    comp = getattr(usage, "completion_tokens", None)
    prompt_tok = getattr(usage, "prompt_tokens", None)
    det = getattr(usage, "completion_tokens_details", None)
    reasoning_tok = getattr(det, "reasoning_tokens", None)

    print(f"[{label}] {dt:5.1f}s finish={choice.finish_reason!r} "
          f"content_len={len(content) if isinstance(content, str) else repr(content)[:60]} "
          f"reasoning_content_len={len(reasoning) if isinstance(reasoning, str) else None} "
          f"usage(completion={comp}, reasoning_tokens={reasoning_tok}, prompt={prompt_tok})")
    if isinstance(content, str) and content.strip():
        print(f"           head: {content.strip()[:110]!r}")


print(f"model={MODEL}  base_url={BASE_URL}\n")

# 1. The config as-is: max_tokens 4096 (what the killed run used), thinking disabled.
probe("4096 + thinking:disabled", max_tokens=4096, extra_body={"thinking": {"type": "disabled"}})

# 2. The bumped value, same body.
probe("8192 + thinking:disabled", max_tokens=8192, extra_body={"thinking": {"type": "disabled"}})

# 3. Is `thinking` the honoured knob at all? Try the chat_template_kwargs form.
probe("4096 + enable_thinking=False",
      max_tokens=4096, extra_body={"chat_template_kwargs": {"enable_thinking": False}})

# 4. No thinking controls at all -- the server default.
probe("4096 + no thinking control", max_tokens=4096)

# 5. Cheap sanity call: does a trivial prompt return content at all?
t0 = time.time()
r = client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": "Reply with exactly: ok"}], max_tokens=32
)
c0 = (r.choices or [None])[0]
print(f"\n[triv] {time.time() - t0:.1f}s finish={getattr(c0, 'finish_reason', None)!r} "
      f"content={getattr(c0.message, 'content', None)!r} "
      f"reasoning={getattr(c0.message, 'reasoning_content', None)!r}")
