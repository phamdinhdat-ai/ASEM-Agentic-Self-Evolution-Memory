"""Corpus-level supersession audit across every finished bank in a tag.

Aggregates `supersession_audit.py` over all per-conversation banks and reports
the same-session / cross-session split corpus-wide, plus the per-conversation
rate (so "uniformly broken" vs "broken only where the allow-list fires" can be
distinguished). Read-only.

Usage:
    python analysis/supersession_corpus.py --root static/memory_banks/locomo10/thg_v2/ASEM-THG
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Any, Dict, List


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="analysis/_supersession_corpus.json")
    args = ap.parse_args()

    banks = sorted(glob.glob(os.path.join(args.root, "*", "asem_thg.sqlite")))
    print(f"banks found: {len(banks)}")
    rows: List[Dict[str, Any]] = []
    for bank in banks:
        conv = os.path.basename(os.path.dirname(bank))
        tmp = os.path.join(os.path.dirname(args.out), f"_sup_{conv}.json")
        rc = subprocess.call([
            sys.executable, "analysis/supersession_audit.py",
            "--bank", bank, "--examples", "0", "--json", tmp,
        ])
        if rc != 0 or not os.path.exists(tmp):
            print(f"  !! {conv}: audit failed rc={rc}")
            continue
        with open(tmp, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        d["conversation"] = conv
        rows.append(d)
        print(f"  {conv}: notes={d['n_notes']} records={d['n_superseded_by_link_records']} "
              f"pairs={d['n_unique_pairs']} same={d['n_same_session']} "
              f"cross={d['n_cross_session']}")

    tot_notes = sum(r["n_notes"] for r in rows)
    tot_pairs = sum(r["n_unique_pairs"] for r in rows)
    tot_same = sum(r["n_same_session"] for r in rows)
    tot_cross = sum(r["n_cross_session"] for r in rows)
    tot_notes_sup = sum(r.get("n_notes_with_supersession", 0) for r in rows)

    print("\n=== CORPUS TOTAL ===")
    print(f"  conversations        : {len(rows)}")
    print(f"  notes                : {tot_notes}")
    print(f"  supersession pairs    : {tot_pairs}")
    print(f"    same-session        : {tot_same}")
    print(f"    cross-session       : {tot_cross}")
    print(f"  pairs per 1k notes    : {(tot_pairs / tot_notes * 1000) if tot_notes else 0:.2f}")

    payload = {
        "root": args.root,
        "n_conversations": len(rows),
        "notes_total": tot_notes,
        "pairs_total": tot_pairs,
        "pairs_same_session": tot_same,
        "pairs_cross_session": tot_cross,
        "per_conversation": rows,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"json -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
