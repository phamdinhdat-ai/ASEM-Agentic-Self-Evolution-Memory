"""Evidence-recall instrument for ASEM-THG (G7).

Answers one question the repo could not answer before: **did retrieval put the
gold evidence into the context the answer model actually saw?**

Why this exists
---------------
Every ASEM-THG number published before this file was an end-to-end score. When
a category scores badly there was no way to tell whether retrieval missed the
evidence or the answer model had it and still failed. That ambiguity is what
made the multi-hop cell (LoCoMo EM 0.0625, judge 0.344) impossible to act on.

What it measures
----------------
Notes are facts extracted from a session, so the join is at SESSION granularity:

    note.session_id  ("s1", "s2", ...)  <->  gold evidence  ("D1:3" -> session 1)

Two recall numbers per category:

  session_recall  fraction of questions with >=1 retrieved note from a gold
                  session. This is the ceiling on what the answer model can do.
  all_sessions    fraction of questions where EVERY gold session is represented.
                  Multi-hop questions name two sessions; missing one is why the
                  second premise is unavailable.

Recall is reported over the notes the answer model actually received
(``used_notes``), so it measures context recall, not candidate recall. That is
deliberate: a gold note that retrieval found but the context builder dropped is
just as unavailable to the model as one never retrieved.

Usage
-----
    python analysis/evidence_recall.py \
        --preds data/benchmarks/results/.../preds/<tag>__<model>__ASEM-THG.jsonl \
        --dataset datasets/locomo/locomo10.json \
        --bank-root static/memory_banks/locomo10/<tag>/ASEM-THG

    # compare several arms side by side
    python analysis/evidence_recall.py --arm legacy=<preds-a.jsonl> --arm thgv1=<preds-b.jsonl> ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# LoCoMo writes gold evidence as "D<session>:<turn>", e.g. "D1:3" or "D2:8".
_DIA_RE = re.compile(r"D(\d+):(\d+)")


def gold_sessions(evidence: Iterable[str]) -> Set[str]:
    """Map LoCoMo evidence ids to the bank's session_id vocabulary.

    ``"D1:3"`` -> ``"s1"``. The bank stores ``session_id`` as ``s<N>`` and the
    dataset writes ``D<N>``; the two are the same ordinal (verified: conv0's
    first question's gold is ``D1:3`` and its top retrieved note carries
    ``session_id='s1'``).
    """
    out: Set[str] = set()
    for item in evidence or []:
        m = _DIA_RE.search(str(item))
        if m:
            out.add(f"s{int(m.group(1))}")
    return out


def load_note_sessions(bank_path: str) -> Dict[str, str]:
    """note_id -> session_id, read straight from the frozen bank."""
    conn = sqlite3.connect(bank_path)
    try:
        rows = conn.execute(
            "SELECT id, session_id FROM notes WHERE session_id IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return {str(r[0]): str(r[1]) for r in rows}


def find_bank(bank_root: str, conversation_id: str) -> Optional[str]:
    """Locate ``<bank_root>/<conv>/<something>.sqlite``."""
    conv_dir = os.path.join(bank_root, conversation_id)
    if not os.path.isdir(conv_dir):
        return None
    for name in sorted(os.listdir(conv_dir)):
        if name.endswith(".sqlite"):
            return os.path.join(conv_dir, name)
    return None


def evaluate_arm(
    preds_path: str,
    dataset_path: str,
    bank_root: str,
) -> Dict[str, object]:
    """Compute per-category session recall for one prediction file."""
    with open(preds_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    with open(dataset_path, encoding="utf-8") as fh:
        dataset = json.load(fh)

    # QA arrays are per conversation, indexed the same way as ``_idx``
    # (verified: 199/199 exact question matches on conv0). The harness names
    # conversations positionally (``locomo_0000``) while the dataset's own
    # ``sample_id`` is ``conv-26``; the ordinal is the join key.
    qa_by_conv: Dict[str, List[dict]] = {
        f"locomo_{i:04d}": conv.get("qa", []) for i, conv in enumerate(dataset)
    }

    bank_cache: Dict[str, Dict[str, str]] = {}

    # category -> [n, hits_any, hits_all, n_used]
    agg: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0])
    missing_bank = 0
    missing_used = 0
    used_legacy = 0

    for row in rows:
        conv = str(row.get("conversation_id", ""))
        idx = int(row.get("idx", -1))
        # Prefer the post-fit context set: notes the answer prompt actually
        # carried. Fall back to the retriever's candidate list for preds files
        # written before `rendered_notes` existed.
        used = row.get("rendered_notes")
        source = "rendered"
        if used is None:
            used = row.get("used_notes")
            source = "used"

        qa = qa_by_conv.get(conv)
        if qa is None or not (0 <= idx < len(qa)):
            continue

        gold = gold_sessions(qa[idx].get("evidence") or [])
        if not gold:
            # Questions with no evidence (some adversarial rows) cannot be
            # scored for recall; excluding them keeps the denominator honest.
            continue

        if used is None:
            missing_used += 1
            continue
        if source == "used":
            used_legacy += 1

        if conv not in bank_cache:
            path = find_bank(bank_root, conv)
            bank_cache[conv] = load_note_sessions(path) if path else {}
            if not path:
                missing_bank += 1
        note_sessions = bank_cache[conv]

        got = {note_sessions[nid] for nid in used if nid in note_sessions}

        cat = row.get("category_name") or f"cat{row.get('category', 0)}"
        bucket = agg[cat]
        bucket[0] += 1
        bucket[3] += len(used)
        if got & gold:
            bucket[1] += 1
        if gold <= got:
            bucket[2] += 1

    return {
        "preds": preds_path,
        "n_rows": len(rows),
        "missing_bank": missing_bank,
        "missing_used": missing_used,
        "used_legacy_fallback": used_legacy,
        "per_category": {
            cat: {
                "n": v[0],
                "session_recall": (v[1] / v[0]) if v[0] else 0.0,
                "all_sessions": (v[2] / v[0]) if v[0] else 0.0,
                "mean_used_notes": (v[3] / v[0]) if v[0] else 0.0,
            }
            for cat, v in sorted(agg.items())
        },
    }


def _totals(per_cat: Dict[str, dict]) -> dict:
    n = sum(c["n"] for c in per_cat.values())
    if not n:
        return {"n": 0, "session_recall": 0.0, "all_sessions": 0.0}
    return {
        "n": n,
        "session_recall": sum(c["session_recall"] * c["n"] for c in per_cat.values()) / n,
        "all_sessions": sum(c["all_sessions"] * c["n"] for c in per_cat.values()) / n,
    }


def _print_arm(name: str, res: Dict[str, object]) -> None:
    per_cat = res["per_category"]
    print(f"\n=== {name} ===")
    print(f"    {res['preds']}")
    if res["missing_bank"]:
        print(f"    WARNING: {res['missing_bank']} conversation(s) had no bank")
    if res["missing_used"]:
        print(f"    WARNING: {res['missing_used']} row(s) carried no note ids")
    if res["used_legacy_fallback"]:
        print(
            f"    NOTE: {res['used_legacy_fallback']} row(s) had no rendered_notes; "
            "scored on the candidate list (pre-G7 preds file)"
        )
    print(f"    {'category':<14} {'n':>4} {'sess_recall':>12} {'all_sessions':>13} {'used':>6}")
    for cat, c in per_cat.items():
        print(
            f"    {cat:<14} {c['n']:>4} {c['session_recall']:>12.3f} "
            f"{c['all_sessions']:>13.3f} {c['mean_used_notes']:>6.1f}"
        )
    t = _totals(per_cat)
    print(
        f"    {'OVERALL':<14} {t['n']:>4} {t['session_recall']:>12.3f} "
        f"{t['all_sessions']:>13.3f}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=PREDS_JSONL",
        help="A prediction file to score, labelled. Repeatable.",
    )
    ap.add_argument("--preds", help="Single prediction file (shorthand for --arm).")
    ap.add_argument("--dataset", default="datasets/locomo/locomo10.json")
    ap.add_argument("--bank-root", default="static/memory_banks/locomo10/thg_v2_postfix/ASEM-THG")
    ap.add_argument("--json-out", help="Write the full result set here.")
    args = ap.parse_args(argv)

    arms: List[Tuple[str, str]] = []
    for spec in args.arm:
        if "=" not in spec:
            print(f"--arm needs NAME=PATH, got {spec!r}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        arms.append((name, path))
    if args.preds:
        arms.append((os.path.basename(args.preds), args.preds))

    if not arms:
        print("nothing to score: pass --preds or --arm NAME=PATH", file=sys.stderr)
        return 2

    results = {}
    for name, path in arms:
        if not os.path.exists(path):
            print(f"missing preds file: {path}", file=sys.stderr)
            return 2
        res = evaluate_arm(path, args.dataset, args.bank_root)
        results[name] = res
        _print_arm(name, res)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
