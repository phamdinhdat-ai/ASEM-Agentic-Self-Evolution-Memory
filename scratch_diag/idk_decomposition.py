"""Decompose ASEM's 'I don't know' answers into the THREE distinct failure modes.

  (1) INGESTION  — no note in the bank contains the gold answer
  (2) RETRIEVAL  — the answer IS in the bank, but retrieval never puts it in the
                   context the answer agent sees
  (3) ABSTENTION — the answer IS in the retrieved context, and the model still
                   answered "I don't know"

Each bucket needs a completely different fix, so measure before tuning.
Retrieval is run with the BARE question (what `strip_query_prefix` now feeds)
and the current relation-aware traversal.

Usage:
  python scratch_diag/idk_decomposition.py
"""
from __future__ import annotations

import json
import math
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
# Which frozen bank to measure. The IDK QUESTION SET always comes from
# ds_nothink (the only run with predictions); passing TAG=ds_fixed measures the
# rebuilt banks against that same question set — a bank-side, LLM-free
# projection of how the ingestion + retrieval buckets shifted.
TAG = sys.argv[1] if len(sys.argv) > 1 else "ds_nothink"
BANK_ROOT = os.path.join(ROOT, "static", "memory_banks", "locomo10", TAG)
EMBED_CFG = {"embedder_provider": "huggingface",
             "embedder_name": "sentence-transformers/all-MiniLM-L6-v2"}
IDK_RE = re.compile(r"^\s*(i\s*(don'?t|do not) know|i'?m not sure|not mentioned|no information|"
                    r"cannot determine|can'?t determine|unknown|n/?a)\b", re.I)
STOP = set("""a an the of to in on at for with and or is are was were be been being it its this that
his her their they he she them i you we my your our as by from into about over under had has have did
do does not no but if then than so such very more most much many""".split())


def toks(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2}


def cached_bm25(notes):
    docs = [re.findall(r"\w+", " ".join([n.c, " ".join(n.K), " ".join(n.G), n.X,
                                         " ".join(n.entities)]).lower()) for n in notes]
    tcs, df = [], {}
    for d in docs:
        tc = {}
        for t in d:
            tc[t] = tc.get(t, 0) + 1
        tcs.append(tc)
        for t in tc:
            df[t] = df.get(t, 0) + 1
    N = max(1, len(docs))
    avgdl = sum(len(d) for d in docs) / N
    k1, b = 1.5, 0.75

    def search(query, k=10):
        out = []
        for i, tc in enumerate(tcs):
            s = 0.0
            for tok in re.findall(r"\w+", query.lower()):
                if tok in tc:
                    f = tc[tok]
                    idf = math.log(1.0 + (N - df.get(tok, 0) + 0.5) / (df.get(tok, 0) + 0.5))
                    s += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * (len(docs[i]) / avgdl)))
            if s > 0:
                out.append((s, notes[i]))
        out.sort(key=lambda x: x[0], reverse=True)
        return out[:k]

    return search


def cached_entities(notes):
    idx = [({e.lower() for e in n.entities}, n) for n in notes]

    def search(entities, k=10):
        norm = {e.strip().lower() for e in entities if e.strip()}
        if not norm:
            return []
        m = [(len(norm & ne), n) for ne, n in idx if norm & ne]
        m.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in m[:k]]

    return search


class MemoBackend:
    def __init__(self, cache):
        self.cache = cache

    def embed(self, text):
        return self.cache[text]

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

    embedder = _build_embedder(EMBED_CFG)
    retr = HybridRetriever(
        backend=None, k1=20, k2=5, delta=0.30, lambda_weight=0.40,
        use_rrf=True, use_bm25=True, use_entity_filter=True, use_temporal_boost=True,
        dense_weight=1.0, bm25_weight=0.8, entity_weight=0.6, temporal_weight=0.5,
        rrf_k=60, max_link_hops=2, enable_link_traversal=True,
    )

    buckets = Counter()
    by_cat = defaultdict(Counter)
    examples = defaultdict(list)

    for conv, crows in sorted(by_conv.items()):
        idk_rows = [r for r in crows if IDK_RE.match(r["pred"] or "")]
        if not idk_rows:
            continue
        src = os.path.join(BANK_ROOT, "ASEM", conv, "asem.sqlite")
        if not os.path.exists(src):
            continue
        tmp = os.path.join(tempfile.mkdtemp(), "bank.sqlite")
        shutil.copy2(src, tmp)
        bank = MemoryBank(tmp)
        notes = bank.list_notes()
        bank.list_notes = lambda _n=notes: _n
        bank.bm25_search = cached_bm25(notes)
        bank.search_by_entities = cached_entities(notes)

        blobs = [(n.id, toks(" ".join(str(x) for x in [n.c, n.K, n.G, n.X, n.entities, n.session_date] if x)))
                 for n in notes]

        texts = {r["question"] for r in idk_rows}
        vecs = embedder.embed_documents(list(texts))
        retr.backend = MemoBackend({t: np.asarray(v, dtype="float32") for t, v in zip(texts, vecs)})

        for r in idk_rows:
            rt = toks(r["ref"] or "")
            cat = r["category_name"]
            if not rt:
                buckets["no_ref"] += 1
                continue
            gold = {nid for nid, bt in blobs if len(rt & bt) / len(rt) >= 0.6}
            if not gold:
                buckets["ingestion"] += 1
                by_cat[cat]["ingestion"] += 1
                if len(examples["ingestion"]) < 6:
                    examples["ingestion"].append(r)
                continue
            ctx = {n.id for n in retr.retrieve(r["question"], bank)}
            if gold & ctx:
                buckets["abstention"] += 1
                by_cat[cat]["abstention"] += 1
                if len(examples["abstention"]) < 6:
                    examples["abstention"].append(r)
            else:
                buckets["retrieval"] += 1
                by_cat[cat]["retrieval"] += 1
                if len(examples["retrieval"]) < 6:
                    examples["retrieval"].append(r)
        bank.close()

    total = buckets["ingestion"] + buckets["retrieval"] + buckets["abstention"]
    print("=" * 78)
    print(f"ASEM 'I don't know' decomposition  (n={total} IDK rows with a usable ref)")
    print(f"  IDK question set : ds_nothink predictions")
    print(f"  bank measured    : {TAG}")
    print("=" * 78)
    labels = {
        "ingestion": "INGESTION  — gold not in bank        -> re-ingest / extraction",
        "retrieval": "RETRIEVAL  — gold in bank, not in ctx -> widen/repair retrieval",
        "abstention": "ABSTENTION — gold in ctx, still IDK   -> prompt / decision",
    }
    for key in ("ingestion", "retrieval", "abstention"):
        print(f"  {labels[key]:<52} {buckets[key]:>4} ({100*buckets[key]/max(total,1):5.1f}%)")

    print("\nby category (ingestion / retrieval / abstention):")
    for cat, c in sorted(by_cat.items()):
        n = c["ingestion"] + c["retrieval"] + c["abstention"]
        print(f"  {cat:<14} n={n:<4} {c['ingestion']:>3} / {c['retrieval']:>3} / {c['abstention']:>3}")

    for key in ("ingestion", "retrieval", "abstention"):
        print(f"\n--- {key} examples ---")
        for r in examples[key]:
            print(f"  [{r['conversation_id']} #{r['idx']} {r['category_name']}] {r['question']}")
            print(f"      ref={r['ref']!r}  pred={(r['pred'] or '')[:60]!r}")


if __name__ == "__main__":
    main()
