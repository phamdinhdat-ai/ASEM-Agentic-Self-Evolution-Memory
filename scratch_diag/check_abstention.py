"""Validate the abstention detector against the real ASEM predictions.

The recovery pass only fires when `is_abstention(pred)` is True, so the
detector must agree with how the model actually phrases refusals in this
dataset (e.g. "I don't know the exact date; the notes only mention ...",
"I don't know — the notes only say ..."). Any miss = a lost recovery attempt.

Usage:
  python scratch_diag/check_abstention.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from asem.answer_agent import is_abstention  # noqa: E402

PRED_DIR = os.path.join(ROOT, "data", "benchmarks", "results", "static", "locomo10", "preds")

# The regex the analysis scripts used (independent reference implementation).
REF_RE = re.compile(
    r"^\s*(i\s*(don'?t|do not) know|i'?m not sure|not mentioned|no information|"
    r"cannot determine|can'?t determine|unknown|n/?a)\b",
    re.I,
)


def main() -> None:
    rows = []
    path = os.path.join(PRED_DIR, "ds_nothink__deepseek_v4_flash__ASEM.jsonl")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    both = missed = newly = 0
    missed_examples = []
    newly_examples = []
    by_cat = Counter()

    for r in rows:
        pred = r["pred"] or ""
        a = bool(REF_RE.match(pred))
        b = is_abstention(pred)
        if a and b:
            both += 1
            by_cat[r["category_name"]] += 1
        elif a and not b:
            missed += 1
            if len(missed_examples) < 10:
                missed_examples.append((r, pred))
        elif b and not a:
            newly += 1
            if len(newly_examples) < 10:
                newly_examples.append((r, pred))

    print(f"preds                 : {len(rows)}")
    print(f"flagged by BOTH       : {both} ({100*both/len(rows):.1f}%)   <- the 23.5%-ish abstention rate")
    print(f"reference-only (MISSED): {missed}")
    print(f"detector-only (extra)  : {newly}")
    print("\nabstentions by category:")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:<14} {n}")

    if missed_examples:
        print("\n--- MISSED (reference matched, detector did not) ---")
        for r, p in missed_examples:
            print(f"  [{r['category_name']}] {p[:110]!r}")
    if newly_examples:
        print("\n--- EXTRA (detector matched, reference did not) ---")
        for r, p in newly_examples:
            print(f"  [{r['category_name']}] {p[:110]!r}")


if __name__ == "__main__":
    main()
