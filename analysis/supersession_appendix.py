"""Full supersession appendix: every pair, with script-classifiable columns.

Owned by WS2. For every `superseded_by` pair in a bank tag, emits:

  * identity: conversation, older/newer note ids, session ids, session dates,
    same-session vs cross-session
  * both fact sentences verbatim
  * subject/predicate/object of each *as available* (the triplet is NOT stored
    on the note - see `hyper_graph.hydrate`'s docstring - so this is derived from
    keywords/entities and is marked `derived`, never presented as the stored
    triplet)
  * interval sanity, from the `LinkRecord.valid_from`/`valid_to` payload:
      older_closed_newer_open  <- the intended shape
      zero_width               <- valid_from == valid_to
      inverted / missing / other
  * shared head verb of the two predicate-ish keyword sets
  * `semantic_verdict` = "unread" for every row (a human/LLM must judge it;
    a script must not)

Usage:
    python analysis/supersession_appendix.py \
        --root static/memory_banks/locomo10/thg_v2/ASEM-THG \
        --out analysis/supersession_appendix.json --md analysis/supersession_appendix.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _dt(v: Any) -> Optional[datetime]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def load(bank: str) -> Dict[str, Dict[str, Any]]:
    uri = "file:" + os.path.abspath(bank).replace("?", "%3f").replace("#", "%23")
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = list(conn.execute(
            "SELECT id, c, K, X, entities, L, t, session_id, session_date FROM notes"
        ))
    finally:
        conn.close()
    out: Dict[str, Dict[str, Any]] = {}
    for nid, c, K, X, ents, L, t, sid, sdate in rows:
        def _j(v, d):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    return d
            return v if v is not None else d
        out[str(nid)] = {
            "id": str(nid), "c": c or "", "K": _j(K, []), "X": X or "",
            "entities": _j(ents, []), "L": _j(L, []), "t": t,
            "session_id": sid, "session_date": sdate,
        }
    return out


def _head_verb(note: Dict[str, Any]) -> Optional[str]:
    """Best-effort predicate proxy: first keyword that looks verb-like."""
    import re
    text = " ".join([str(k) for k in (note.get("K") or [])] + [note.get("c") or ""])
    words = re.findall(r"[a-z]+", text.lower())
    verbs = [w for w in words if w.endswith(("ed", "es", "s")) and len(w) > 3]
    return verbs[0] if verbs else (words[0] if words else None)


def classify_interval(older: Dict[str, Any], newer: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Interval sanity from the stored link payloads (both endpoints)."""
    def links(n):
        return [l for l in (n.get("L") or []) if isinstance(l, dict)
                and str(l.get("relation")) == "superseded_by"]

    older_links, newer_links = links(older), links(newer)
    detail = {
        "older_links": older_links,
        "newer_links": newer_links,
    }
    if not older_links or not newer_links:
        return "missing", detail

    # The link that points from the older note AT the newer one should be closed.
    to_newer = [l for l in older_links if str(l.get("target_id")) == newer["id"]]
    from_newer = [l for l in newer_links if str(l.get("target_id")) == older["id"]]
    if not to_newer or not from_newer:
        return "missing", detail

    older_vf = to_newer[0].get("valid_from")
    older_vt = to_newer[0].get("valid_to")
    newer_vt = from_newer[0].get("valid_to")

    if older_vf and older_vt and str(older_vf) == str(older_vt):
        return "zero_width", detail
    if newer_vt:
        return "inverted", detail
    if older_vt and not newer_vt:
        return "older_closed_newer_open", detail
    if not older_vt and not newer_vt:
        return "open_both", detail
    return "other", detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="analysis/supersession_appendix.json")
    ap.add_argument("--md", default="analysis/supersession_appendix.md")
    args = ap.parse_args()

    banks = sorted(glob.glob(os.path.join(args.root, "*", "asem_thg.sqlite")))
    rows: List[Dict[str, Any]] = []
    for bank in banks:
        conv = os.path.basename(os.path.dirname(bank))
        notes = load(bank)
        seen = set()
        for nid, rec in notes.items():
            for link in rec["L"]:
                if not isinstance(link, dict):
                    continue
                if str(link.get("relation")) != "superseded_by":
                    continue
                tgt = str(link.get("target_id") or "")
                if not tgt or tgt not in notes:
                    continue
                key = tuple(sorted((nid, tgt)))
                if key in seen:
                    continue
                seen.add(key)
                a, b = notes[key[0]], notes[key[1]]
                ta, tb = _dt(a.get("t")), _dt(b.get("t"))
                if ta and tb:
                    older, newer = (a, b) if ta <= tb else (b, a)
                else:
                    older, newer = a, b
                same = bool(older.get("session_id")) and older.get("session_id") == newer.get("session_id")
                cls, detail = classify_interval(older, newer)
                hv_o, hv_n = _head_verb(older), _head_verb(newer)
                rows.append({
                    "conversation": conv,
                    "older_id": older["id"], "newer_id": newer["id"],
                    "older_session": older.get("session_id"),
                    "newer_session": newer.get("session_id"),
                    "older_date": older.get("session_date"),
                    "newer_date": newer.get("session_date"),
                    "older_t": older.get("t"), "newer_t": newer.get("t"),
                    "same_or_cross_session": "same" if same else "cross",
                    "older_sentence": older.get("c"),
                    "newer_sentence": newer.get("c"),
                    "older_keywords": older.get("K"),
                    "newer_keywords": newer.get("K"),
                    "older_entities": older.get("entities"),
                    "newer_entities": newer.get("entities"),
                    "older_head_verb": hv_o, "newer_head_verb": hv_n,
                    "shared_head_verb": (hv_o == hv_n and hv_o is not None),
                    "triplet_source": "derived (triplet is not persisted on Note)",
                    "interval_class": cls,
                    "interval_detail": detail,
                    "semantic_verdict": "unread",
                })

    rows.sort(key=lambda r: (r["conversation"], str(r["older_date"])))
    cls_counts = Counter(r["interval_class"] for r in rows)
    same_counts = Counter(r["same_or_cross_session"] for r in rows)
    shared_counts = Counter(r["shared_head_verb"] for r in rows)

    print(f"pairs: {len(rows)}")
    print(f"  same/cross        : {dict(same_counts)}")
    print(f"  interval class    : {dict(cls_counts)}")
    print(f"  shared head verb  : {dict(shared_counts)}")
    print(f"  semantic verdict  : unread = {sum(1 for r in rows if r['semantic_verdict']=='unread')}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "root": args.root,
            "n_pairs": len(rows),
            "interval_class_counts": dict(cls_counts),
            "same_or_cross_counts": dict(same_counts),
            "shared_head_verb_counts": {str(k): v for k, v in shared_counts.items()},
            "note": ("semantic_verdict is 'unread' for every row on purpose: judging "
                     "whether two facts are genuine versions of one another requires "
                     "semantic judgement, not a script. Script-classifiable columns "
                     "(session split, interval class, shared head verb) ARE filled."),
            "pairs": rows,
        }, fh, indent=2)
    print(f"json -> {args.out}")

    lines = [
        "# Supersession appendix — every `superseded_by` pair",
        "",
        f"Source: `{args.root}`  |  pairs: **{len(rows)}**",
        "",
        "Counting rule: one row per **unique undirected pair**. Each pair is stored",
        "on BOTH endpoint notes, so the stored `superseded_by` link-record count is",
        "2x this number (166 pairs = 332 records across the 10 conversations).",
        "",
        "`semantic_verdict` is `unread` for every row by design — deciding whether two",
        "facts are genuine versions of one another needs semantic judgement, not a",
        "script. The script-classifiable columns ARE filled.",
        "",
        f"- same/cross-session: {dict(same_counts)}",
        f"- interval class: {dict(cls_counts)}",
        f"- shared head verb: {dict(shared_counts)}",
        "",
    ]
    for r in rows:
        lines.append(f"## {r['conversation']} — {r['older_id'][:8]} → {r['newer_id'][:8]}")
        lines.append("")
        lines.append(f"- **{r['same_or_cross_session']}-session**; older `{r['older_session']}` "
                     f"({r['older_date']}) → newer `{r['newer_session']}` ({r['newer_date']})")
        lines.append(f"- interval class: `{r['interval_class']}`")
        lines.append(f"- head verbs: older=`{r['older_head_verb']}` newer=`{r['newer_head_verb']}` "
                     f"shared={r['shared_head_verb']}")
        lines.append(f"- OLDER: {r['older_sentence']}")
        lines.append(f"  - K={r['older_keywords']} entities={r['older_entities']}")
        lines.append(f"- NEWER: {r['newer_sentence']}")
        lines.append(f"  - K={r['newer_keywords']} entities={r['newer_entities']}")
        lines.append(f"- semantic verdict: **{r['semantic_verdict']}**")
        lines.append("")
    with open(args.md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"md   -> {args.md}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
