"""Fast smoke test of the FIXED P4 extraction prompt.

Ingests a couple of LoCoMo sessions through the REAL production path
(`build_eval_system("ASEM", ...)` -> `ingest_system`) so the wiring matches the
static-bank build exactly, then audits the notes for the fields the retriever
and answer agent depend on:

  * entities  (was 100% empty -> killed the entity retrieval channel)
  * speaker
  * K / G / X
  * session_date

Usage:
  python scratch_diag/smoke_ingest_p4.py [conv_index] [n_sessions]
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_env(path: str = ".env") -> None:
    """Minimal .env loader (a bare `python -c` does not load it)."""
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return
    with open(full, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_env()

from eval.phase_runner import (  # noqa: E402
    build_eval_system,
    close_system,
    extract_sessions,
    finalize_system,
    ingest_system,
    load_raw_dataset,
    system_bank_size,
)

DATASET = os.path.join(ROOT, "datasets", "locomo", "locomo10.json")
CONFIG = os.path.join(ROOT, "configs", "models", "deepseek_openai.yaml")


def main() -> None:
    conv_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_sessions = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    raw = load_raw_dataset(DATASET)
    record = raw[conv_idx]
    sessions = extract_sessions(record.get("conversation", {}))
    subset = sessions[:n_sessions]
    print(f"conv={conv_idx}  sessions available={len(sessions)}  ingesting={len(subset)}")
    for s in subset:
        print(f"  session_{s.num}  date={s.date!r}  turns={len(s.turns)}")

    db_dir = tempfile.mkdtemp(prefix="smoke_p4_")
    print(f"db_dir={db_dir}")

    system = build_eval_system("ASEM", CONFIG, db_dir)
    try:
        ingest_system(system, "ASEM", subset)
        edges = finalize_system(system)
        size = system_bank_size(system)
    finally:
        close_system(system)

    print(f"\ningest done | notes={size} | finalize_edges={edges}")

    # ---- Audit the persisted note fields -----------------------------
    db_file = os.path.join(db_dir, "asem.sqlite")
    import sqlite3
    from collections import Counter

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, c, K, G, X, entities, speaker, session_date FROM notes"
    ).fetchall()

    c = Counter()
    samples = []
    for r in rows:
        c["n"] += 1
        ents = json.loads(r["entities"] or "[]") or []
        kw = json.loads(r["K"] or "[]") or []
        if ents:
            c["has_entities"] += 1
        if kw:
            c["has_keywords"] += 1
        if r["speaker"]:
            c["has_speaker"] += 1
        if r["session_date"]:
            c["has_session_date"] += 1
        if len(samples) < 6 and ents:
            samples.append(r)
    conn.close()

    n = c["n"]
    print("\n" + "=" * 78)
    print("FIELD AUDIT (target: entities > 0%)")
    print("=" * 78)
    print(f"  notes                 : {n}")
    print(f"  notes with entities   : {c['has_entities']} ({100*c['has_entities']/max(n,1):.1f}%)   <-- was 0%")
    print(f"  notes with keywords   : {c['has_keywords']} ({100*c['has_keywords']/max(n,1):.1f}%)")
    print(f"  notes with speaker    : {c['has_speaker']} ({100*c['has_speaker']/max(n,1):.1f}%)")
    print(f"  notes with session_date: {c['has_session_date']} ({100*c['has_session_date']/max(n,1):.1f}%)")

    print("\n--- sample notes (format check) ---")
    for r in samples:
        ents = json.loads(r["entities"] or "[]")
        kw = json.loads(r["K"] or "[]")
        print(f"\n  speaker={r['speaker']!r}  session_date={r['session_date']!r}")
        print(f"  entities = {ents}")
        print(f"  keywords = {kw}")
        print(f"  X        = {(r['X'] or '')[:150]!r}")
        print(f"  c        = {(r['c'] or '')[:150]!r}")

    print(f"\nBANK={db_file}")


if __name__ == "__main__":
    main()
