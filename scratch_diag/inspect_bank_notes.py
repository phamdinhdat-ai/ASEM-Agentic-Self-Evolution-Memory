"""Dump a few notes from a real ingested bank to sanity-check extraction quality.

`_fallback_extract` (used when the LLM returns unparseable/empty JSON) produces
raw dialogue lines with a " (Date: ...)" suffix and the tag ["dialogue"]. Real
batch extraction produces standalone facts with keywords/entities/tags. This
distinguishes the two so a silently-failing extraction cannot hide behind a
healthy-looking note count.
"""

from __future__ import annotations

import os
import sqlite3
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BANK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _PROJECT_ROOT, "static", "memory_banks", "locomo10", "deepseek",
    "FastASEM", "locomo_0000", "fast_asem.sqlite",
)
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 12

if not os.path.exists(BANK):
    raise SystemExit(f"Bank not found: {BANK}")

conn = sqlite3.connect(f"file:{BANK}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT id, c, X, G, K, session_id, session_date, timestamp_iso, speaker "
    "FROM notes LIMIT ?",
    (LIMIT,),
).fetchall()
total = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

# How much of the bank looks like fallback output?
conn2 = sqlite3.connect(f"file:{BANK}?mode=ro", uri=True)
all_tags = conn2.execute("SELECT G FROM notes").fetchall()
contents = [r[0] for r in conn2.execute("SELECT c FROM notes").fetchall()]
conn2.close()

dialogue_tagged = sum(1 for r in all_tags if r[0] and "dialogue" in str(r[0]))
date_suffix = sum(1 for c in contents if c and "(Date:" in c)

print(f"bank      : {BANK}")
print(f"notes     : {total}")
print(f"tagged 'dialogue' (fallback marker) : {dialogue_tagged}")
print(f"content ending in '(Date: ...)'     : {date_suffix}")
print(f"mean content length                 : "
      f"{sum(len(c or '') for c in contents) / max(1, len(contents)):.0f} chars")
print()
for i, row in enumerate(rows):
    print(f"[{i}] session={row['session_id']} date={row['session_date']} "
          f"speaker={row['speaker']}")
    print(f"     tags={row['G']}")
    print(f"     keywords={row['K']}")
    print(f"     content={str(row['c'])[:150]!r}")
