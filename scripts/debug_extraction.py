"""Debug: reproduce the batch note-extraction LLM call and inspect raw output.

Uses the exact config + P4 prompt + first session of the default v2 dataset,
then prints the raw model response and diagnostics. One API call.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass  # keys already in env

import yaml

from asem.backends import build_backend
from asem.batch_ingestion import _extract_json
from asem.llm_validator import validate_batch_notes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Load config + backend (LLM only, no embedder download needed) ---
cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/locomo_openai.yaml"), encoding="utf-8"))
infer_cfg = cfg["inference"]
backend = build_backend(infer_cfg)
print(f"backend={infer_cfg['backend']} provider={infer_cfg['langchain'].get('provider')} "
      f"model={infer_cfg['langchain'].get('model')} "
      f"base_url={infer_cfg['langchain'].get('base_url') or os.environ.get('OPENAI_BASE_URL', '(unset)')}")
print(f"OPENAI_API_KEY set: {bool(os.environ.get('OPENAI_API_KEY'))}")

# --- Load the first session of the default dataset (same parser as run_asem_v2) ---
from scripts.run_asem_v2 import _parse_sessions

data_path = os.path.join(ROOT, "datasets/locomo/locomo10.json")
with open(data_path, encoding="utf-8") as fh:
    dataset = json.load(fh)
record = dataset[0]
conversation = record["conversation"]
sessions = _parse_sessions(conversation)
print(f"\ndataset: {os.path.basename(data_path)} | conversation 0 | "
      f"sessions: {len(sessions)}")

# Reconstruct dialogue exactly as run_ingest_phase does
sess_num, date_str, turns = sessions[0]
header = f"[Session {sess_num}"
if date_str:
    header += f" — {date_str}"
header += "]"
dialogue = [header] + turns

# --- Render P4 exactly as BatchIngestor does ---
prompt_path = os.path.join(ROOT, "data/prompts/P4_batch_note_extraction.txt")
template = open(prompt_path, encoding="utf-8").read()
prompt = template.format(dialogue="\n".join(dialogue))
print(f"prompt chars: {len(prompt)} | dialogue turns: {len(dialogue)}")

# --- Call the model ---
print("\n>>> calling model ...")
raw = backend.generate(prompt)
print("=== RAW OUTPUT (first 2000 chars) ===")
print(raw[:2000])
print(f"\n(raw length: {len(raw)})")

# --- Diagnose ---
data = _extract_json(raw, expect_array=True)
print(f"\n=== PARSE ===")
print(f"type: {type(data).__name__}")
if isinstance(data, list):
    print(f"items: {len(data)}")
    for i, item in enumerate(data[:8]):
        if isinstance(item, dict):
            print(f"  [{i}] keys={list(item.keys())}")
            print(f"       content={str(item.get('content', '<<MISSING>>'))[:80]!r}")
        else:
            print(f"  [{i}] NOT A DICT: {type(item).__name__} = {str(item)[:80]!r}")
    print("  ..." if len(data) > 8 else "")
elif isinstance(data, dict):
    print(f"keys={list(data.keys())}")
    for key in ("notes", "facts", "results", "data", "items"):
        if isinstance(data.get(key), list):
            print(f"  wrapped array under '{key}': {len(data[key])} items")

result = validate_batch_notes(data) if isinstance(data, list) else None
print(f"\nvalidate_batch_notes: valid={result.valid if result else False}")
for e in (result.errors if result else ["not a list"])[:5]:
    print(f"  - {e}")

# And with the extraction flag (require_content) the retry path would use:
result2 = validate_batch_notes(data, require_content=True) if isinstance(data, list) else None
print(f"validate_batch_notes(require_content=True): "
      f"valid={result2.valid if result2 else False}")
if result2 and not result2.valid:
    for e in result2.errors[:3]:
        print(f"  - {e}")
