"""Trace retrieval for ASEM's 'I don't know' answers.

For a given conversation, re-run the real HybridRetriever on the frozen bank
and check whether the note that contains the gold answer is actually pulled
into the S4 context, and at what rank. Also compares the eval query
("Conversation between X and Y. Question: ...") against the bare question, to
measure how much the prefix hurts embedding/entity retrieval.

Usage:
  python scratch_diag/trace_retrieval.py locomo_0000 7 29 105 171
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from asem.memory_bank import MemoryBank
from asem.retriever import HybridRetriever
from asem.backends.langchain_backend import _build_embedder

PRED_DIR = os.path.join(ROOT, "data", "benchmarks", "results", "static", "locomo10", "preds")
BANK_ROOT = os.path.join(ROOT, "static", "memory_banks", "locomo10", "ds_nothink")
EMBED_CFG = {
    "embedder_provider": "huggingface",
    "embedder_name": "sentence-transformers/all-MiniLM-L6-v2",
}

STOP = set("""a an the of to in on at for with and or is are was were be been being it its this
that his her their they he she them i you we my your our as by from into about over under had has
have did do does not no but if then than so such very more most much many""".split())


def toks(text: str) -> set[str]:
    import re
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2}


class EmbedOnly:
    """Backend shim: retriever only needs .embed(); generation is never called."""

    def __init__(self):
        self._e = _build_embedder(EMBED_CFG)

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self._e.embed_query(text), dtype="float32")

    def generate(self, *a, **k):  # pragma: no cover
        raise RuntimeError("trace does not call generate()")


def load_conv_rows(conv: str) -> dict[int, dict]:
    path = os.path.join(PRED_DIR, "ds_nothink__deepseek_v4_flash__ASEM.jsonl")
    out = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["conversation_id"] == conv:
                out[r["idx"]] = r
    return out


def main() -> None:
    conv = sys.argv[1] if len(sys.argv) > 1 else "locomo_0000"
    idxs = [int(x) for x in sys.argv[2:]] or [7, 29, 105, 128, 135, 152, 171]

    src = os.path.join(BANK_ROOT, "ASEM", conv, "asem.sqlite")
    tmp = os.path.join(tempfile.mkdtemp(), "bank.sqlite")
    shutil.copy2(src, tmp)

    embedder = EmbedOnly()
    bank = MemoryBank(tmp)
    notes = bank.list_notes()
    print(f"conv={conv}  bank notes={len(notes)}")

    retr = HybridRetriever(
        backend=embedder, k1=20, k2=5, delta=0.30, lambda_weight=0.40,
        use_rrf=True, use_bm25=True, use_entity_filter=True, use_temporal_boost=True,
        dense_weight=1.0, bm25_weight=0.8, entity_weight=0.6, temporal_weight=0.5,
        rrf_k=60, max_link_hops=2, enable_link_traversal=True,
    )

    rows = load_conv_rows(conv)

    def gold_notes(ref: str):
        rt = toks(ref)
        hits = []
        for n in notes:
            blob = " ".join(str(x) for x in [n.c, n.K, n.G, n.X, n.entities, n.session_date] if x)
            bt = toks(blob)
            if rt and len(rt & bt) / len(rt) >= 0.6:
                hits.append(n.id)
        return hits

    for idx in idxs:
        r = rows.get(idx)
        if r is None:
            print(f"\n### idx {idx}: not found")
            continue
        full_q = r["query"]
        bare_q = r["question"]
        gold = gold_notes(r["ref"])
        print("\n" + "=" * 78)
        print(f"### idx {idx} [{r['category_name']}]  pred={(r['pred'] or '')[:70]!r}")
        print(f"    Q       : {bare_q}")
        print(f"    ref     : {r['ref']!r}")
        print(f"    gold in : {gold or 'NOT FOUND IN BANK'}")

        for label, q in (("eval query (with prefix)", full_q), ("bare question", bare_q)):
            retr.stats = {}
            got = retr.retrieve(q, bank)
            got_ids = [n.id for n in got]
            hit = [g for g in gold if g in got_ids]
            print(f"\n    --- {label} ---")
            print(f"        stats={retr.stats}")
            print(f"        retrieved={len(got)}  gold_retrieved={bool(hit)} {hit}")
            for k, n in enumerate(got):
                mark = " <== GOLD" if n.id in gold else ""
                print(f"          {k+1}. [{n.session_date}] {n.c[:95]!r}{mark}")
    bank.close()


if __name__ == "__main__":
    main()
