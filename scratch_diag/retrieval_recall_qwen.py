"""Where do ASEM's Qwen3-4B losses come from: ingestion, retrieval, or answering?

Takes the slice of questions where the judge marks ASEM wrong (optionally also
requiring FullContext to be RIGHT, i.e. the questions a huge verbatim context
answers and the memory graph does not) and replays ASEM's REAL retriever against
the frozen ds_fixed bank to split each row into:

  INGESTION  — no note in the bank carries the gold answer
  RETRIEVAL  — the gold-bearing note exists but never enters the top-k context
  GENERATION — the gold-bearing note IS in the retrieved context, yet the answer
               was still wrong (paraphrase/abstraction/person-swap failure)

Retrieval uses the bare question, exactly what `strip_query_prefix` feeds the
pipeline. No LLM call: the embedder is real (all-MiniLM), the backend is a memo
cache, and BM25/entity search are cached in memory.

Usage:
  python scratch_diag/retrieval_recall_qwen.py [CATEGORY_FILTER] [--fc-right]
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

import numpy as np  # noqa: E402
from asem.backends.langchain_backend import _build_embedder  # noqa: E402
from asem.memory_bank import MemoryBank  # noqa: E402
from asem.retriever import HybridRetriever  # noqa: E402

TAG = "ds_fixed"
MODEL = "qwen_qwen3_4b_instruct_2507"
RES = os.path.join(ROOT, "data", "benchmarks", "results", "static", "locomo10")
BANK_ROOT = os.path.join(ROOT, "static", "memory_banks", "locomo10", TAG)
EMBED_CFG = {"embedder_provider": "huggingface",
             "embedder_name": "sentence-transformers/all-MiniLM-L6-v2"}

STOP = set("""a an the of to in on at for with and or is are was were be been being it its this that
his her their they he she them i you we my your our as by from into about over under had has have did
do does not no but if then than so such very more most much many""".split())


def toks(text):
    return {w for w in re.findall(r"[a-z0-9']+", (text or "").lower())
            if w not in STOP and len(w) > 2}


def cached_bm25(notes):
    docs = [re.findall(r"\w+", " ".join([n.c, " ".join(n.K), " ".join(n.G), n.X,
                                         " ".join(n.entities)]).lower()) for n in notes]
    tcs, df = [], {}
    for d in docs:
        tc = Counter(d)
        tcs.append(tc)
        for w in tc:
            df[w] = df.get(w, 0) + 1
    N = len(docs)

    def bm25(query, k=10):
        q = [w for w in re.findall(r"\w+", (query or "").lower()) if w in df]
        scores = []
        for tc, n in zip(tcs, notes):
            s = 0.0
            for w in q:
                f = tc.get(w, 0)
                if not f:
                    continue
                idf = np.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * (f * 2.5) / (f + 1.5 * (0.25 + 0.75 * len(tc) / max(1, sum(len(t) for t in tcs) / N)))
            if s:
                scores.append((s, n))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]

    return bm25


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


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    cat_filter = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    fc_right = "--fc-right" in sys.argv

    preds = {r["idx"]: r for r in load(os.path.join(RES, "preds", f"{TAG}__{MODEL}__ASEM.jsonl"))}
    sc_asem = {r["idx"]: r for r in load(os.path.join(RES, "scores", f"{TAG}__{MODEL}__ASEM.jsonl"))}
    sc_fc = {r["idx"]: r for r in load(os.path.join(RES, "scores", f"{TAG}__{MODEL}__FullContext.jsonl"))}

    slices = {
        "all ASEM-wrong": lambda i: not sc_asem[i]["judge_correct"],
        "cat4 single-hop ASEM-wrong": lambda i: (preds[i]["category_name"] == "conversational"
                                                 and not sc_asem[i]["judge_correct"]),
        "cat4 single-hop ASEM-wrong & FC-right": lambda i: (preds[i]["category_name"] == "conversational"
                                                            and not sc_asem[i]["judge_correct"]
                                                            and sc_fc[i]["judge_correct"]),
        "cat1 multi-hop ASEM-wrong": lambda i: (preds[i]["category_name"] == "single_hop"
                                                and not sc_asem[i]["judge_correct"]),
        "cat4 single-hop ASEM-RIGHT (contrast)": lambda i: (preds[i]["category_name"] == "conversational"
                                                            and sc_asem[i]["judge_correct"]),
    }
    if cat_filter:
        slices = {k: v for k, v in slices.items() if cat_filter in k}

    embedder = _build_embedder(EMBED_CFG)
    retr = HybridRetriever(
        backend=None, k1=20, k2=5, delta=0.30, lambda_weight=0.40,
        use_rrf=True, use_bm25=True, use_entity_filter=True, use_temporal_boost=True,
        dense_weight=1.0, bm25_weight=0.8, entity_weight=0.6, temporal_weight=0.5,
        rrf_k=60, max_link_hops=2, enable_link_traversal=True,
    )

    by_conv = defaultdict(list)
    for i, r in preds.items():
        by_conv[r["conversation_id"]].append(i)

    results = defaultdict(Counter)
    examples = defaultdict(list)

    for conv, idxs in sorted(by_conv.items()):
        src = os.path.join(BANK_ROOT, "ASEM", conv, "asem.sqlite")
        if not os.path.exists(src):
            print(f"  (no bank for {conv})")
            continue
        tmp = os.path.join(tempfile.mkdtemp(), "bank.sqlite")
        shutil.copy2(src, tmp)
        bank = MemoryBank(tmp)
        notes = bank.list_notes()
        bank.list_notes = lambda _n=notes: _n
        bank.bm25_search = cached_bm25(notes)
        bank.search_by_entities = cached_entities(notes)
        blobs = [(n.id, toks(" ".join(str(x) for x in
                                      [n.c, n.K, n.G, n.X, n.entities, n.session_date] if x)))
                 for n in notes]

        questions = {preds[i]["question"] for i in idxs}
        vecs = embedder.embed_documents(list(questions))
        retr.backend = MemoBackend({q: np.asarray(v, dtype="float32") for q, v in zip(questions, vecs)})

        for i in idxs:
            r = preds[i]
            rt = toks(r["ref"])
            if not rt:
                results["ALL"]["no_ref"] += 1
                continue
            for name, fn in slices.items():
                if not fn(i):
                    continue
                c = results[name]
                c["n"] += 1
                gold = {nid for nid, bt in blobs if len(rt & bt) / len(rt) >= 0.6}
                if not gold:
                    c["ingestion"] += 1
                    if len(examples[(name, "ingestion")]) < 4:
                        examples[(name, "ingestion")].append(r)
                    continue
                ctx = {n.id for n in retr.retrieve(r["question"], bank)}
                if gold & ctx:
                    c["generation"] += 1
                    if len(examples[(name, "generation")]) < 4:
                        examples[(name, "generation")].append(r)
                else:
                    c["retrieval"] += 1
                    if len(examples[(name, "retrieval")]) < 4:
                        examples[(name, "retrieval")].append(r)
        bank.close()

    print()
    print("=" * 92)
    print(f"ASEM failure decomposition — bank={TAG}, preds={MODEL}")
    print("  gold-bearing note = any note whose fields cover >=60% of the gold tokens")
    print("=" * 92)
    print(f"{'slice':<42}{'n':>6}{'INGESTION':>12}{'RETRIEVAL':>12}{'GENERATION':>12}")
    for name, c in results.items():
        if name == "ALL":
            continue
        n = c["n"] or 1
        ing = f"{c['ingestion']} ({100*c['ingestion']/n:.0f}%)"
        ret = f"{c['retrieval']} ({100*c['retrieval']/n:.0f}%)"
        gen = f"{c['generation']} ({100*c['generation']/n:.0f}%)"
        print(f"{name:<42}{c['n']:>6}{ing:>12}{ret:>12}{gen:>12}")

    for (name, bucket), rows in sorted(examples.items()):
        print(f"\n  examples [{name} / {bucket}]:")
        for r in rows:
            print(f"    Q: {r['question'][:80]}")
            print(f"       gold={r['ref'][:60]!r}  pred={r['pred'][:90]!r}")


if __name__ == "__main__":
    main()
