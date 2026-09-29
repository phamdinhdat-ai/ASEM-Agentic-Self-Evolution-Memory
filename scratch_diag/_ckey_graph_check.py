"""Inspect the ASEM-THG graph structure in a static bank.

Usage:
    conda activate memory-r1
    $env:PYTHONPATH="."
    python scratch_diag/_ckey_graph_check.py [path_to_asem_thg.sqlite]

If no path is given, defaults to:
  static/memory_banks/locomo10/ckey_deepseek/ASEM-THG/locomo_0000/asem_thg.sqlite
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import Counter

DEFAULT_DB = os.path.join(
    "static", "memory_banks", "locomo10", "ckey_deepseek",
    "ASEM-THG", "locomo_0000", "asem_thg.sqlite",
)


def main(db_path: str) -> None:
    if not os.path.exists(db_path):
        print(f"ERROR: {db_path} does not exist")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # --- Schema ---
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    print(f"Tables: {tables}")

    # --- Notes ---
    note_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    print(f"\n=== Notes: {note_count} ===")

    # Speaker distribution
    speakers = Counter(
        r[0] for r in conn.execute(
            "SELECT speaker FROM notes WHERE speaker IS NOT NULL AND speaker != ''"
        ).fetchall()
    )
    print(f"Speakers: {dict(speakers)}")

    # Tag distribution (G field is JSON array)
    tag_counter: Counter = Counter()
    for row in conn.execute("SELECT G FROM notes WHERE G IS NOT NULL AND G != ''").fetchall():
        try:
            tags = json.loads(row[0])
            if isinstance(tags, list):
                tag_counter.update(tags)
        except (json.JSONDecodeError, TypeError):
            pass
    print(f"Top tags: {tag_counter.most_common(10)}")

    # --- Links (from L field in notes) ---
    link_counter: Counter = Counter()  # relation type -> count
    total_links = 0
    linked_notes = 0
    unlinked_notes = 0
    link_targets: set = set()

    for row in conn.execute("SELECT id, L FROM notes").fetchall():
        note_id, l_json = row[0], row[1]
        if not l_json:
            unlinked_notes += 1
            continue
        try:
            links = json.loads(l_json)
        except (json.JSONDecodeError, TypeError):
            unlinked_notes += 1
            continue
        if not links:
            unlinked_notes += 1
            continue
        linked_notes += 1
        for link in links:
            if isinstance(link, dict):
                rel = link.get("relation", "linked")
                target = link.get("target_id", "?")
            else:
                rel = "linked"
                target = str(link)
            link_counter[rel] += 1
            total_links += 1
            link_targets.add(target)

    print(f"\n=== Links ===")
    print(f"Total link records: {total_links}")
    print(f"Notes with links: {linked_notes}")
    print(f"Notes without links: {unlinked_notes}")
    print(f"Unique link targets: {len(link_targets)}")
    print(f"Relation types: {dict(link_counter.most_common())}")

    # --- Temporal coverage ---
    dates = Counter()
    for row in conn.execute(
        "SELECT session_date FROM notes WHERE session_date IS NOT NULL AND session_date != ''"
    ).fetchall():
        dates[row[0]] += 1
    print(f"\n=== Temporal ===")
    print(f"Unique session dates: {len(dates)}")
    for d, c in sorted(dates.items())[:10]:
        print(f"  {d}: {c} notes")
    if len(dates) > 10:
        print(f"  ... and {len(dates) - 10} more")

    # --- Q-value distribution ---
    q_vals = [r[0] for r in conn.execute("SELECT q FROM notes").fetchall() if r[0] is not None]
    if q_vals:
        print(f"\n=== Q-values ===")
        print(f"  min={min(q_vals):.3f}  max={max(q_vals):.3f}  "
              f"mean={sum(q_vals)/len(q_vals):.3f}  "
              f"median={sorted(q_vals)[len(q_vals)//2]:.3f}")

    # --- Meta ---
    meta = {r[0]: r[1] for r in conn.execute("SELECT key, value FROM meta").fetchall()}
    print(f"\n=== Meta ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    conn.close()
    print(f"\nDone. DB: {db_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    main(path)
