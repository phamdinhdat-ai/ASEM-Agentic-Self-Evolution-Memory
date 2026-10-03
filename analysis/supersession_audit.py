"""Supersession audit: same-session restatement vs cross-session replacement.

Owned by WS2 (`bank-runner`). For every `superseded_by` link in a bank, prints
both notes' session identity, dates, entities, keywords and fact sentence
verbatim, and classifies the pair. Read-only.

Usage:
    python analysis/supersession_audit.py \
        --bank static/memory_banks/locomo10/thg_v2/ASEM-THG/locomo_0000/asem_thg.sqlite
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _load(path: str) -> Dict[str, Dict[str, Any]]:
    uri = "file:" + os.path.abspath(path).replace("?", "%3f").replace("#", "%23")
    conn = sqlite3.connect(uri, uri=True)
    try:
        cols = [r[1] for r in conn.execute('PRAGMA table_info("notes")')]
        rows = list(conn.execute(f'SELECT {", ".join(chr(34)+c+chr(34) for c in cols)} FROM notes'))
    finally:
        conn.close()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rec = dict(zip(cols, row))
        for col in ("K", "G", "L", "entities", "e", "z"):
            if col in rec and isinstance(rec[col], str):
                try:
                    rec[col] = json.loads(rec[col])
                except (json.JSONDecodeError, TypeError):
                    pass
        out[str(rec["id"])] = rec
    return out


def _dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--examples", type=int, default=10)
    ap.add_argument("--json", help="write the full pair table here")
    args = ap.parse_args()

    notes = _load(args.bank)
    print(f"bank: {args.bank}")
    print(f"notes: {len(notes)}")

    # Collect unique superseded_by pairs.
    pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    n_link_records = 0
    for nid, rec in notes.items():
        for link in rec.get("L") or []:
            if isinstance(link, dict):
                rel = str(link.get("relation") or "linked")
                tgt = str(link.get("target_id") or "")
                vf, vt = link.get("valid_from"), link.get("valid_to")
            else:
                rel, tgt, vf, vt = "linked", str(link), None, None
            if rel != "superseded_by" or not tgt:
                continue
            n_link_records += 1
            key = _pair_key(nid, tgt)
            entry = pairs.setdefault(key, {"a": key[0], "b": key[1], "intervals": []})
            entry["intervals"].append({"from": nid, "to": tgt, "valid_from": vf, "valid_to": vt})

    print(f"superseded_by link records: {n_link_records}")
    print(f"unique undirected supersession pairs: {len(pairs)}")
    if not pairs:
        print("\nNo superseded_by links in this bank.")
        return 0

    rows: List[Dict[str, Any]] = []
    for key, entry in pairs.items():
        a, b = entry["a"], entry["b"]
        na, nb = notes.get(a, {}), notes.get(b, {})
        sa, sb = na.get("session_id"), nb.get("session_id")
        da, db = na.get("session_date"), nb.get("session_date")
        ta, tb = _dt(na.get("t")), _dt(nb.get("t"))
        same_session = bool(sa) and sa == sb
        # Which is older, by timestamp then by date string.
        if ta and tb:
            newer, older = (na, nb) if ta >= tb else (nb, na)
        else:
            newer, older = na, nb
        rows.append({
            "pair": [a, b],
            "same_session": same_session,
            "a_session_id": sa, "b_session_id": sb,
            "a_session_date": da, "b_session_date": db,
            "a_t": na.get("t"), "b_t": nb.get("t"),
            "older_id": older.get("id"), "newer_id": newer.get("id"),
            "older_session": older.get("session_id"),
            "newer_session": newer.get("session_id"),
            "older_date": older.get("session_date"),
            "newer_date": newer.get("session_date"),
            "older_sentence": older.get("c"),
            "newer_sentence": newer.get("c"),
            "older_entities": older.get("entities"),
            "newer_entities": newer.get("entities"),
            "older_K": older.get("K"),
            "newer_K": newer.get("K"),
            "older_X": older.get("X"),
            "newer_X": newer.get("X"),
            "intervals": entry["intervals"],
        })

    n_same = sum(1 for r in rows if r["same_session"])
    n_cross = len(rows) - n_same
    print(f"\n=== SPLIT ===")
    print(f"  same-session restatements : {n_same}")
    print(f"  cross-session replacements: {n_cross}")

    print(f"\n=== VERBATIM PAIRS (all {min(args.examples, len(rows))}) ===")
    for i, r in enumerate(rows[: args.examples], 1):
        kind = "SAME-SESSION" if r["same_session"] else "CROSS-SESSION"
        print(f"\n--- pair {i}/{len(rows)} [{kind}] ---")
        print(f"  session ids : {r['a_session_id']} / {r['b_session_id']}")
        print(f"  session date: {r['a_session_date']} / {r['b_session_date']}")
        print(f"  older ({r['older_id'][:8]} @ {r['older_session']} {r['older_date']}):")
        print(f"    {r['older_sentence']}")
        print(f"    entities={r['older_entities']}  K={r['older_K']}")
        print(f"  newer ({r['newer_id'][:8]} @ {r['newer_session']} {r['newer_date']}):")
        print(f"    {r['newer_sentence']}")
        print(f"    entities={r['newer_entities']}  K={r['newer_K']}")
        iv = r["intervals"]
        print(f"  intervals: {json.dumps(iv[:3])}")

    # Entity overlap tells us whether the pair shares a subject.
    def _ents(x: Any) -> set:
        return {str(e).strip().lower() for e in (x or []) if str(e).strip()}

    shared = [_ents(r["older_entities"]) & _ents(r["newer_entities"]) for r in rows]
    n_shared = sum(1 for s in shared if s)
    print(f"\n  pairs whose two notes share >=1 entity: {n_shared}/{len(rows)}")
    print(f"  most common shared entity among pairs: "
          f"{Counter(e for s in shared for e in s).most_common(5)}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "bank": args.bank,
                "n_notes": len(notes),
                "n_superseded_by_link_records": n_link_records,
                "n_unique_pairs": len(pairs),
                "n_same_session": n_same,
                "n_cross_session": n_cross,
                "n_pairs_sharing_entity": n_shared,
                "pairs": rows,
            }, fh, indent=2)
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
