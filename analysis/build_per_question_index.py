"""Build a per-question LongMemEval graph-quality index from bank snapshots.

Owned by WS2 (`bank-runner`). LongMemEval's `ASEM-THG` bank is CLEARED per
question (`ASEMTHGSystem.reset()` -> `memory_bank.clear()`), so the graph for a
question only exists while that question is being processed. This script turns
captured per-question bank snapshots into the per-question table.

Two inputs are supported:
  * `--snapshot-dir`  : directory of `q_<idx>_<question_id>.sqlite` snapshots
  * `--from-banks`    : the live bank, measured once (single-row fallback)

Writes `<out>` = `data/benchmarks/longmemeval_banks/gb_full_per_question.json`
(or whatever `--out` says).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from typing import Any, Dict, List

from analysis.graph_metrics import _maybe_json, compute_metrics, load_notes


def top_entities(notes: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    """Top-k entities by document frequency, with df and df/N."""
    df: Counter = Counter()
    for n in notes:
        for e in {str(x).strip().lower() for x in (n.get("entities") or []) if str(x).strip()}:
            df[e] += 1
    total = len(notes) or 1
    return [
        {"entity": ent, "df": c, "df_over_n": round(c / total, 4)}
        for ent, c in df.most_common(k)
    ]


def snapshot_metrics(path: str) -> Dict[str, Any]:
    notes = load_notes(path)
    m = compute_metrics(notes, n_examples=0)
    return {
        "n_notes": m["n_notes"],
        "n_edges": m["n_link_records"],
        "edge_histogram": m["relation_histogram"],
        "entity_fanout": m["entity_fanout"],
        "n_distinct_entities": m["n_distinct_entities"],
        "implied_same_entity_pairs": m["implied_same_entity_pairs"],
        "top10_entities_by_df": top_entities(notes, 10),
        "n_notes_with_empty_entities": m["n_notes_entities_empty"],
        "n_notes_with_supersession": m["n_notes_with_superseded_by"],
        "share_notes_with_supersession": m["share_notes_with_superseded_by"],
        "n_supersession_pairs": m["n_supersession_pairs"],
        "n_notes_date_unknown": m["n_notes_date_unknown"],
        "n_notes_timestamp_iso_null": m["n_notes_timestamp_iso_null"],
        "session_ids": sorted({str(n.get("session_id")) for n in notes if n.get("session_id")}),
    }


_QNAME = re.compile(r"^q_(\d+)_(.+)\.sqlite$")


def from_snapshot_dir(d: str, out_path: str) -> Dict[str, Any]:
    files = sorted(glob.glob(os.path.join(d, "*.sqlite")))
    per_q: List[Dict[str, Any]] = []
    for path in files:
        base = os.path.basename(path)
        m = _QNAME.match(base)
        if not m:
            continue
        idx, qid = int(m.group(1)), m.group(2)
        row = {"idx": idx, "question_id": qid}
        row.update(snapshot_metrics(path))
        row["bank_path"] = path
        per_q.append(row)
    per_q.sort(key=lambda r: r["idx"])
    payload = {"dataset": "longmemeval_oracle", "n_questions": len(per_q), "per_question": per_q}
    _write(out_path, payload)
    return payload


def _write(out_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, out_path)
    print(f"wrote {out_path} ({payload.get('n_questions')} questions)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-dir", help="dir of q_<idx>_<qid>.sqlite snapshots")
    p.add_argument("--bank", help="single bank to measure as one row")
    p.add_argument("--question-id", default="unknown")
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="gb_full")
    args = p.parse_args()

    if args.snapshot_dir:
        payload = from_snapshot_dir(args.snapshot_dir, args.out)
        print(json.dumps({"n": payload["n_questions"]}, indent=2))
        return 0
    if args.bank:
        row = {"idx": 0, "question_id": args.question_id}
        row.update(snapshot_metrics(args.bank))
        row["bank_path"] = args.bank
        _write(args.out, {"dataset": "longmemeval_oracle", "label": args.label,
                          "n_questions": 1, "per_question": [row]})
        return 0
    p.error("need --snapshot-dir or --bank")
    return 2


if __name__ == "__main__":
    sys.exit(main())
