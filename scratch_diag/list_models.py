# -*- coding: utf-8 -*-
"""Scratch: list model ids the configured endpoint serves (no secrets printed)."""
import os, sys

ROOT = r"C:\Users\Dat Pham\Documents\datpd\master-phenikaa\thesis_master\ASEM-Masters\ASEM-Agentic-Self-Evolution-Memory"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# Load .env the same way the runners do.
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

base = os.environ.get("OPENAI_BASE_URL", "")
print("base_url set:", bool(base))
print("api_key set:", bool(os.environ.get("OPENAI_API_KEY")))

from openai import OpenAI
client = OpenAI()
try:
    ids = [m.id for m in client.models.list()]
    print("models:", ids)
except Exception as e:  # noqa: BLE001
    print("models.list failed:", type(e).__name__, str(e)[:300])
