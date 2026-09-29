"""Inspect a built static memory bank (SQLite) without loading FAISS.

    conda activate memory-r1 && python scratch_diag/_ckey_inspect_bank.py <bank.sqlite> [--samples N]

Reports file size, note count, per-session breakdown, and a few sample notes so
a partial/failed ingest can be told apart from a complete one.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import Counter


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    db = sys.argv[1].rstrip('"')
    samples = 5
    if "--samples" in sys.argv:
        samples = int(sys.argv[sys.argv.index("--samples") + 1])

    print(f"bank      : {db}")
    print(f"exists    : {os.path.exists(db)}")
    if not os.path.exists(db):
        return 1
    print(f"bytes     : {os.path.getsize(db):,}")

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]
    print(f"tables    : {tables}")

    cols = [r[1] for r in cur.execute("PRAGMA table_info(notes)")]
    print(f"note cols : {cols}")

    total = cur.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    print(f"notes     : {total}")

    if total == 0:
        print("\n-> EMPTY bank: nothing was persisted. The ingest failed before/while "
              "writing the first session, so this file carries no usable memory.")
        conn.close()
        return 0

    # Per-session coverage reveals WHERE the run stopped.
    if "session_id" in cols:
        rows = cur.execute(
            "SELECT COALESCE(session_id,'<none>') AS sid, COUNT(*) AS n "
            "FROM notes GROUP BY sid ORDER BY n DESC"
        ).fetchall()
        print(f"\nsessions represented: {len(rows)}")
        for r in rows[:12]:
            print(f"  {r['sid']:<16} {r['n']:>5} notes")

    # Link density tells you whether the hyper-graph linking pass ran at all.
    edge_total = 0
    nodes_with_links = 0
    for (raw,) in cur.execute("SELECT L FROM notes WHERE L IS NOT NULL AND L != '[]'"):
        try:
            items = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(items, list):
            if items:
                nodes_with_links += 1
            edge_total += len(items)
    print(f"\nlink edges: {edge_total} (on {nodes_with_links}/{total} notes)")

    print(f"\n--- {samples} sample notes ---")
    for row in cur.execute(
        "SELECT id, c, session_id, session_date FROM notes ORDER BY t LIMIT ?",
        (samples,),
    ):
        fact = (row["c"] or "").replace("\n", " ")
        print(f"[{row['id'][:8]}] {row['session_date'] or '?'} "
              f"({row['session_id'] or '-'})\n    {fact[:200]}")

    if "entities" in cols:
        tag_counts: Counter = Counter()
        for (raw,) in cur.execute("SELECT G FROM notes WHERE G IS NOT NULL"):
            try:
                tag_counts.update(json.loads(raw))
            except (TypeError, ValueError):
                pass
        print(f"\ntop tags  : {tag_counts.most_common(10)}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
