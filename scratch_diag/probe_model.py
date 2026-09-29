# -*- coding: utf-8 -*-
"""Scratch: probe whether the endpoint accepts a given model name."""
import os, sys

ROOT = r"C:\Users\Dat Pham\Documents\datpd\master-phenikaa\thesis_master\ASEM-Masters\ASEM-Agentic-Self-Evolution-Memory"
os.chdir(ROOT)
dotenv = os.path.join(ROOT, ".env")
if os.path.exists(dotenv):
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv, override=False)
    except ImportError:
        for line in open(dotenv, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from openai import OpenAI
client = OpenAI()

for model in [sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-flash"]:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=8,
        )
        print(f"OK   {model!r} -> {r.choices[0].message.content!r}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {model!r} -> {type(e).__name__}: {str(e)[:220]}")
