# -*- coding: utf-8 -*-
"""Scratch: inspect an existing ASEM-THG static bank schema + sample."""
import json, os, sqlite3

ROOT = r"C:\Users\Dat Pham\Documents\datpd\master-phenikaa\thesis_master\ASEM-Masters\ASEM-Agentic-Self-Evolution-Memory"
db = os.path.join(ROOT, "static", "memory_banks", "locomo10", "ds_thg",
                  "ASEM-THG", "locomo_0000", "asem_thg.sqlite")
print("exists:", os.path.exists(db), db)
con = sqlite3.connect(db)
cur = con.cursor()
print("tables:", cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
cols = [r[1] for r in cur.execute("PRAGMA table_info(notes)")]
print("notes columns:", cols)
n = cur.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
print("notes:", n)
row = cur.execute("SELECT * FROM notes LIMIT 1").fetchone()
if row:
    d = dict(zip(cols, row))
    for k, v in d.items():
        s = str(v)
        print(f"  {k}: {s[:160]}")
con.close()
