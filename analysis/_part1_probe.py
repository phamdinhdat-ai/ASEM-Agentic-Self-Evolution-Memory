"""Part-1 smoke probe: per-note facts, session ids, edges, dates (READ-ONLY)."""

import json
import sqlite3
import sys
from collections import Counter

from analysis.graph_metrics import load_notes, compute_metrics

bank = sys.argv[1] if len(sys.argv) > 1 else "data/benchmarks/longmemeval_banks/asem_thg.sqlite"
notes = load_notes(bank)
print("NOTES:", len(notes))

# --- session_id distribution -------------------------------------------
sids = Counter(str(n.get("session_id")) for n in notes)
print("\n=== session_id distinct values ===")
for k, v in sorted(sids.items(), key=lambda kv: str(kv[0])):
    print(f"  {k!r}: {v}")
print("distinct session_id count:", len(sids))
print("null session_id:", sum(1 for n in notes if not n.get("session_id")))

# --- session dates -----------------------------------------------------
print("\n=== session_date / t / timestamp_iso ===")
for n in notes[:6]:
    print(f"  {n['id'][:8]} | sid={n.get('session_id')!r} | session_date={n.get('session_date')!r}"
          f" | t={n.get('t')!r} | ts_iso={n.get('timestamp_iso')!r}")

# --- relation histogram -------------------------------------------------
rel = Counter()
for n in notes:
    for l in (n.get("L") or []):
        rel[str(l.get("relation"))] += 1
print("\n=== relation histogram (direct) ===", dict(rel))

# --- entities -----------------------------------------------------------
print("\n=== entities ===")
print("notes with empty entities:", sum(1 for n in notes if not (n.get("entities") or [])))
ent_counts = Counter()
for n in notes:
    for e in (n.get("entities") or []):
        ent_counts[str(e).lower()] += 1
print("distinct entities:", len(ent_counts))
print("top entities:", ent_counts.most_common(12))

# --- fan-out ------------------------------------------------------------
m = compute_metrics(notes)
print("\n=== fanout ===", json.dumps(m["entity_fanout"], indent=2))
print("implied same-entity pairs:", m["implied_same_entity_pairs"])
print("relation histogram:", m["relation_histogram"])
print("superseded notes:", m["n_notes_with_superseded_by"], "pairs:", m["n_supersession_pairs"])
print("date unknown:", m["n_notes_date_unknown"], "ts_iso null:", m["n_notes_timestamp_iso_null"])
print("session_id kinds:", m["session_id_kinds"], m["session_id_samples"])
