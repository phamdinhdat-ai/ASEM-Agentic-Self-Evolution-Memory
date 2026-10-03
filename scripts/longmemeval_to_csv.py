"""Export LongMemEval records to a flat table (CSV and/or columns/rows JSON).

The official LongMemEval ``.json`` is a list of records with 9 fields, four of
which are list-valued (``haystack_dates``, ``haystack_session_ids``,
``haystack_sessions``, ``answer_session_ids``).  This script flattens each
record into one table row, JSON-encoding the list cells so the result opens
cleanly in a spreadsheet (one record per row).

Usage
-----
    python scripts/longmemeval_to_csv.py \
        --input datasets/longmemeval/longmemeval_oracle.json \
        --output datasets/longmemeval/longmemeval_oracle.csv

    # columns/rows JSON mirror instead of (or in addition to) a CSV
    python scripts/longmemeval_to_csv.py \
        --input datasets/longmemeval/longmemeval_oracle.json \
        --json-output datasets/longmemeval/longmemeval_oracle_table.json

The ``--json-output`` form writes the plan's two-key table shape::

    {"columns": [9 names], "rows": [[9 values], ...]}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List

COLUMNS: List[str] = [
    "question_id",
    "question_type",
    "question",
    "answer",
    "question_date",
    "haystack_dates",
    "haystack_session_ids",
    "haystack_sessions",
    "answer_session_ids",
]

# Cells stored as JSON strings inside the flat table.
_LIST_CELLS = (
    "haystack_dates",
    "haystack_session_ids",
    "haystack_sessions",
    "answer_session_ids",
)


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path!r} is not a JSON list of records")
    return data


def record_to_row(record: Dict[str, Any]) -> List[str]:
    """Flatten one record into 9 string cells (list cells JSON-encoded)."""
    row: List[str] = []
    for col in COLUMNS:
        value = record.get(col, [] if col in _LIST_CELLS else "")
        if col in _LIST_CELLS:
            row.append(json.dumps(value if value is not None else [], ensure_ascii=False))
        else:
            row.append("" if value is None else str(value))
    return row


def write_csv(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for record in records:
            writer.writerow(record_to_row(record))


def write_table_json(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"columns": list(COLUMNS), "rows": [record_to_row(r) for r in records]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="LongMemEval .json (list of records)")
    parser.add_argument("--output", default=None, help="Destination .csv path")
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional columns/rows .json mirror path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Export only the first N records")
    args = parser.parse_args(argv)

    if not args.output and not args.json_output:
        parser.error("provide --output and/or --json-output")

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    if args.output:
        write_csv(records, args.output)
        print(f"wrote {len(records)} rows -> {args.output}")
    if args.json_output:
        write_table_json(records, args.json_output)
        print(f"wrote {len(records)} rows -> {args.json_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
