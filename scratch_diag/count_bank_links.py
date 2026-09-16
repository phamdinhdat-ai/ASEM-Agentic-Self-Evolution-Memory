"""Report how many typed link edges each ingested bank actually contains.

`banks.json` / `manifest.json` record `link_edges: 0` for every non-ASEM system,
because `eval.phase_runner.finalize_system` only calls `finalize_conversation()`
when the system exposes it (ASEM v1 does; FastASEM does not). A 0 there therefore
means **"not measured"**, not "no links". This reads the `L` column directly so
the two cases can be told apart.

Usage:
    python scratch_diag/count_bank_links.py [glob]

Default glob covers every bank under `static/memory_banks/<dataset>/<tag>/`.
"""

from __future__ import annotations

import glob
import json
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATTERN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _ROOT, "static", "memory_banks", "*", "*", "*", "*", "*.sqlite"
)

files = sorted(glob.glob(_PATTERN))
if not files:
    raise SystemExit(f"No banks matched: {_PATTERN}")

print(f"{'notes':>7} {'nonzero':>8} {'edges':>7}  {'relations':<28} bank")
print("-" * 100)

grand_edges = 0
for path in files:
    rel_path = os.path.relpath(path, _ROOT)
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        rows = conn.execute(
            "SELECT L FROM notes WHERE L IS NOT NULL AND L != '[]'"
        ).fetchall()
        conn.close()
    except sqlite3.Error as exc:
        print(f"{'?':>7} {'?':>8} {'?':>7}  {'-':<28} {rel_path}  ({exc})")
        continue

    edges = 0
    relations: set[str] = set()
    for (raw,) in rows:
        try:
            items = json.loads(raw) or []
        except (TypeError, ValueError):
            continue
        if not isinstance(items, list):
            continue
        edges += len(items)
        for item in items:
            if isinstance(item, dict):
                relations.add(str(item.get("relation", "?")))

    grand_edges += edges
    rel = ",".join(sorted(relations)) if relations else "-"
    print(f"{notes:>7} {len(rows):>8} {edges:>7}  {rel:<28} {rel_path}")

print("-" * 100)
print(f"{len(files)} banks, {grand_edges} typed link edges total")
