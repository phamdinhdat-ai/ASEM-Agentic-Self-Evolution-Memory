"""Grep a bank's notes for terms — used to tell a prompt regression from a
smoke-bank coverage gap.

Usage:
  python scratch_diag/check_bank_terms.py <db_file> term1 term2 ...
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

db = sys.argv[1]
terms = [t.lower() for t in sys.argv[2:]] or ["trans", "married"]

conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT c, K, X, entities, speaker, session_date FROM notes").fetchall()
conn.close()

print(f"bank={db}  notes={len(rows)}")
for term in terms:
    hits = []
    for r in rows:
        blob = " ".join(str(x) for x in [r["c"], r["K"], r["X"], r["entities"]] if x).lower()
        if term in blob:
            hits.append(r)
    print(f"\n=== term {term!r}: {len(hits)} hit(s) ===")
    for r in hits[:6]:
        print(f"  speaker={r['speaker']} date={r['session_date']}")
        print(f"    X: {(r['X'] or '')[:160]}")
