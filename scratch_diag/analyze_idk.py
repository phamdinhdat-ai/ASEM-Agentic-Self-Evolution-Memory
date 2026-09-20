"""Diagnose the 'I don't know' (IDK) rate in the static LoCoMo eval.

Stage 1 (stdlib only): count IDK answers per system / category, and check
whether the gold answer's content is at least *present in the frozen bank*
(lexical overlap on content words). That splits a missing-answer failure into:
  (a) answer NOT in bank   -> ingestion / extraction loss
  (b) answer IN bank       -> retrieval / context / prompting loss

Usage:
  python scratch_diag/analyze_idk.py
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(ROOT, "data", "benchmarks", "results", "static", "locomo10", "preds")
BANK_ROOT = os.path.join(ROOT, "static", "memory_banks", "locomo10", "ds_nothink")

SYSTEMS = ["ASEM", "FastASEM", "FullContext", "NoMemory"]

IDK_RE = re.compile(
    r"^\s*(i\s*(don'?t|do not) know|i'?m not sure|not mentioned|no information|"
    r"cannot determine|can'?t determine|unknown|n/?a)\b",
    re.I,
)

STOP = set(
    """a an the of to in on at for with and or is are was were be been being it its this that
    his her their they he she them i you we my your our as by from into about over under
    had has have did do does not no but if then than so such very more most much many
    """.split()
)


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2}


def load_preds(system: str):
    path = os.path.join(PRED_DIR, f"ds_nothink__deepseek_v4_flash__{system}.jsonl")
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_bank_notes(conv: str):
    """Return list of (id, blob_text) for the ASEM bank of one conversation."""
    path = os.path.join(BANK_ROOT, "ASEM", conv, "asem.sqlite")
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    out = []
    for r in conn.execute("SELECT id, c, K, G, X, entities, session_date FROM notes"):
        blob = " ".join(
            str(x) for x in [r["c"], r["K"], r["G"], r["X"], r["entities"], r["session_date"]]
            if x is not None
        )
        out.append((r["id"], blob))
    conn.close()
    return out


def main() -> None:
    print("=" * 78)
    print("STAGE 1 — IDK rate by system / category")
    print("=" * 78)
    banks_cache: dict[str, list] = {}
    for sysname in SYSTEMS:
        rows = load_preds(sysname)
        if not rows:
            continue
        idk = [r for r in rows if IDK_RE.match(r["pred"] or "")]
        print(f"\n{sysname}: n={len(rows)}  IDK={len(idk)} ({100*len(idk)/len(rows):.1f}%)")
        bycat = Counter((r["category_name"], IDK_RE.match(r["pred"] or "") is not None) for r in rows)
        cats = sorted({r["category_name"] for r in rows})
        for cat in cats:
            tot = sum(v for (c, _), v in bycat.items() if c == cat)
            n_idk = bycat.get((cat, True), 0)
            print(f"    {cat:<14} n={tot:<4} idk={n_idk:<4} ({100*n_idk/max(tot,1):.1f}%)")
        print(f"    avg answer len = {sum(len(r['pred'] or '') for r in rows)/len(rows):.0f} chars")

    print("\n" + "=" * 78)
    print("STAGE 2 — for ASEM IDK answers: is the gold answer present in the bank?")
    print("=" * 78)
    rows = load_preds("ASEM")
    idk_rows = [r for r in rows if IDK_RE.match(r["pred"] or "")]
    present = 0
    absent = 0
    examples_present = []
    examples_absent = []
    for r in idk_rows:
        conv = r["conversation_id"]
        if conv not in banks_cache:
            banks_cache[conv] = load_bank_notes(conv)
        notes = banks_cache[conv]
        ref_toks = tokens(r["ref"] or "")
        if not ref_toks:
            continue
        hit = None
        for nid, blob in notes:
            bt = tokens(blob)
            if ref_toks and len(ref_toks & bt) / len(ref_toks) >= 0.6:
                hit = nid
                break
        if hit:
            present += 1
            if len(examples_present) < 12:
                examples_present.append((r, hit))
        else:
            absent += 1
            if len(examples_absent) < 12:
                examples_absent.append(r)

    total = present + absent
    print(f"\nASEM IDK rows with a usable ref: {total}")
    print(f"  answer tokens >=60% found in SOME bank note : {present} ({100*present/max(total,1):.1f}%)")
    print(f"  answer tokens NOT found in any bank note    : {absent} ({100*absent/max(total,1):.1f}%)")

    print("\n--- Examples: gold IS in the bank but model said IDK (retrieval/context loss) ---")
    for r, nid in examples_present:
        print(f"  [{r['conversation_id']} #{r['idx']} {r['category_name']}] Q: {r['question']}")
        print(f"      ref={r['ref']!r}  bank_note={nid}  pred={(r['pred'] or '')[:90]!r}")

    print("\n--- Examples: gold NOT in the bank (ingestion/extraction loss) ---")
    for r in examples_absent:
        print(f"  [{r['conversation_id']} #{r['idx']} {r['category_name']}] Q: {r['question']}")
        print(f"      ref={r['ref']!r}  pred={(r['pred'] or '')[:90]!r}")


if __name__ == "__main__":
    main()
