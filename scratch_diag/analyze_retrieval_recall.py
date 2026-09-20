"""Stage 3 — measure retrieval recall on the frozen ASEM banks.

For every ASEM QA row, find which bank notes contain the gold answer (>=60%
content-word overlap), then run the real retriever twice:
  (a) with the eval query  "Conversation between X and Y. Question: ..."
  (b) with the bare question only
and record whether a gold note made it into the S4 context.

This separates:
  * ingestion loss   (no note contains the answer at all)
  * retrieval loss   (answer is in the bank but not retrieved)
  * context/prompt loss (answer retrieved, model still said IDK)

Usage:
  python scratch_diag/analyze_retrieval_recall.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from asem.memory_bank import MemoryBank
from asem.retriever import HybridRetriever
from asem.backends.langchain_backend import _build_embedder

PRED_DIR = os.path.join(ROOT, "data", "benchmarks", "results", "static", "locomo10", "preds")
BANK_ROOT = os.path.join(ROOT, "static", "memory_banks", "locomo10", "ds_nothink")
EMBED_CFG = {"embedder_provider": "huggingface",
             "embedder_name": "sentence-transformers/all-MiniLM-L6-v2"}
IDK_RE = re.compile(r"^\s*(i\s*(don'?t|do not) know|i'?m not sure|not mentioned|no information|"
                    r"cannot determine|can'?t determine|unknown|n/?a)\b", re.I)
STOP = set("""a an the of to in on at for with and or is are was were be been being it its this that
his her their they he she them i you we my your our as by from into about over under had has have did
do does not no but if then than so such very more most much many""".split())


def toks(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2}


class EmbedOnly:
    def __init__(self):
        self._e = _build_embedder(EMBED_CFG)

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self._e.embed_query(text), dtype="float32")

    def generate(self, *a, **k):
        raise RuntimeError("no generate")


def main() -> None:
    rows = []
    with open(os.path.join(PRED_DIR, "ds_nothink__deepseek_v4_flash__ASEM.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    by_conv = defaultdict(list)
    for r in rows:
        by_conv[r["conversation_id"]].append(r)

    embedder = EmbedOnly()
    retr = HybridRetriever(
        backend=embedder, k1=20, k2=5, delta=0.30, lambda_weight=0.40,
        use_rrf=True, use_bm25=True, use_entity_filter=True, use_temporal_boost=True,
        dense_weight=1.0, bm25_weight=0.8, entity_weight=0.6, temporal_weight=0.5,
        rrf_k=60, max_link_hops=2, enable_link_traversal=True,
    )

    agg = Counter()
    by_cat = defaultdict(Counter)
    by_idk = defaultdict(Counter)
    prefix_cost_examples = []

    for conv, crows in sorted(by_conv.items()):
        src = os.path.join(BANK_ROOT, "ASEM", conv, "asem.sqlite")
        if not os.path.exists(src):
            continue
        tmp = os.path.join(tempfile.mkdtemp(), "bank.sqlite")
        shutil.copy2(src, tmp)
        bank = MemoryBank(tmp)
        blob_cache = []
        for n in bank.list_notes():
            blob = " ".join(str(x) for x in [n.c, n.K, n.G, n.X, n.entities, n.session_date] if x)
            blob_cache.append((n.id, toks(blob)))

        for r in crows:
            rt = toks(r["ref"] or "")
            if not rt:
                continue
            gold = {nid for nid, bt in blob_cache if len(rt & bt) / len(rt) >= 0.6}
            agg["ref_usable"] += 1
            if gold:
                agg["gold_in_bank"] += 1
            else:
                agg["gold_missing_bank"] += 1
                continue

            is_idk = bool(IDK_RE.match(r["pred"] or ""))
            hit_prefix = bool({n.id for n in retr.retrieve(r["query"], bank)} & gold)
            hit_bare = bool({n.id for n in retr.retrieve(r["question"], bank)} & gold)

            bucket = "idk" if is_idk else "answered"
            for key in (bucket, "all"):
                by_idk[key]["n"] += 1
                by_idk[key]["prefix"] += hit_prefix
                by_idk[key]["bare"] += hit_bare
            by_cat[r["category_name"]]["n"] += 1
            by_cat[r["category_name"]]["prefix"] += hit_prefix
            by_cat[r["category_name"]]["bare"] += hit_bare

            if hit_bare and not hit_prefix and len(prefix_cost_examples) < 15:
                prefix_cost_examples.append((conv, r["idx"], r["category_name"], r["question"]))
        bank.close()

    def pct(a, b):
        return f"{100*a/b:.1f}%" if b else "n/a"

    print("=" * 78)
    print("RETRIEVAL RECALL — gold note present in bank, then retrieved into S4 ctx")
    print("=" * 78)
    print(f"rows with usable ref        : {agg['ref_usable']}")
    print(f"  gold answer IN bank       : {agg['gold_in_bank']} ({pct(agg['gold_in_bank'], agg['ref_usable'])})")
    print(f"  gold answer NOT in bank   : {agg['gold_missing_bank']} ({pct(agg['gold_missing_bank'], agg['ref_usable'])})")

    print("\n--- recall of the gold note, among rows where it IS in the bank ---")
    for key in ("all", "answered", "idk"):
        c = by_idk[key]
        print(f"  {key:<10} n={c['n']:<5} prefixed_query={pct(c['prefix'], c['n'])}  "
              f"bare_question={pct(c['bare'], c['n'])}")

    print("\n--- by category (prefixed / bare) ---")
    for cat, c in sorted(by_cat.items()):
        print(f"  {cat:<14} n={c['n']:<5} {pct(c['prefix'], c['n'])} / {pct(c['bare'], c['n'])}")

    print("\n--- cases where the bare question retrieves the gold but the eval prefix does NOT ---")
    for conv, idx, cat, q in prefix_cost_examples:
        print(f"  [{conv} #{idx} {cat}] {q}")


if __name__ == "__main__":
    main()
