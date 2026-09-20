"""Field-population audit of the frozen ASEM banks.

The answer context is built from c + session_date + entities + K + X, and the
retriever's entity channel needs note.entities. If those are empty, both the
retrieval signal and the relational/attribution signal are degraded.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM = sys.argv[1] if len(sys.argv) > 1 else "ASEM"
BANK_FILE = sys.argv[2] if len(sys.argv) > 2 else "asem.sqlite"
TAG = sys.argv[3] if len(sys.argv) > 3 else "ds_nothink"
BANK_ROOT = os.path.join(ROOT, "static", "memory_banks", "locomo10", TAG, SYSTEM)

tot = Counter()
per_conv = {}
for conv in sorted(os.listdir(BANK_ROOT)):
    path = os.path.join(BANK_ROOT, conv, BANK_FILE)
    if not os.path.exists(path):
        continue
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    c = Counter()
    for r in conn.execute("SELECT c, K, G, X, entities, speaker, session_date FROM notes"):
        c["n"] += 1
        try:
            ents = json.loads(r["entities"] or "[]") or []
        except Exception:
            ents = []
        try:
            kw = json.loads(r["K"] or "[]") or []
        except Exception:
            kw = []
        if not ents:
            c["empty_entities"] += 1
        if not kw:
            c["empty_keywords"] += 1
        if not (r["X"] or "").strip() or r["X"] == r["c"]:
            c["empty_or_dup_X"] += 1
        if r["speaker"]:
            c["has_speaker"] += 1
        if r["session_date"]:
            c["has_session_date"] += 1
        if (r["c"] or "").lstrip().startswith("["):
            c["c_has_speaker_prefix"] += 1
    conn.close()
    per_conv[conv] = c
    tot.update(c)

n = tot["n"]
print(f"{SYSTEM} banks audited: {len(per_conv)} conversations, {n} notes")
print(f"  notes with EMPTY entities      : {tot['empty_entities']} ({100*tot['empty_entities']/n:.1f}%)")
print(f"  notes with EMPTY keywords (K)  : {tot['empty_keywords']} ({100*tot['empty_keywords']/n:.1f}%)")
print(f"  notes with empty/dup description: {tot['empty_or_dup_X']} ({100*tot['empty_or_dup_X']/n:.1f}%)")
print(f"  notes with speaker field set   : {tot['has_speaker']} ({100*tot['has_speaker']/n:.1f}%)")
print(f"  notes with session_date set    : {tot['has_session_date']} ({100*tot['has_session_date']/n:.1f}%)")
print(f"  notes whose c starts with [X]  : {tot['c_has_speaker_prefix']} ({100*tot['c_has_speaker_prefix']/n:.1f}%)")
print("\nper conversation:")
for conv, c in per_conv.items():
    print(f"  {conv}: n={c['n']:<4} empty_entities={100*c['empty_entities']/c['n']:5.1f}%  "
          f"empty_K={100*c['empty_keywords']/c['n']:5.1f}%  speaker={100*c['has_speaker']/c['n']:5.1f}%")
