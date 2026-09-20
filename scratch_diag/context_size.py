"""Measure the ACTUAL retrieval context size, so the context window can be sized.

For every question, run the real retriever on a frozen bank and compute the size
of the fully rendered ANSWER prompt (the exact string sent to the model):

  * distil path  -> P_distil.txt + json.dumps(_note_payload(...) for each note)
  * recovery     -> same, but with the widened pool (k2/recovery_k2, lower delta)

Reports percentiles and how many prompts exceed a given window, both for the
configured per-call cap and for the client-wide `inference.*.max_tokens`.

Usage:
  python scratch_diag/context_size.py [TAG] [CONFIG]
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import yaml
from asem.answer_agent import AnswerAgent
from asem.backends.langchain_backend import _build_embedder
from asem.memory_bank import MemoryBank
from asem.retriever import HybridRetriever

PRED_DIR = os.path.join(ROOT, "data", "benchmarks", "results", "static", "locomo10", "preds")
EMBED_CFG = {"embedder_provider": "huggingface",
             "embedder_name": "sentence-transformers/all-MiniLM-L6-v2"}

TAG = sys.argv[1] if len(sys.argv) > 1 else "ds_fixed"
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "configs/models/qwen3_4b_openai.yaml"
BANK_ROOT = os.path.join(ROOT, "static", "memory_banks", "locomo10", TAG)


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


def pct(values, q):
    if not values:
        return 0
    return sorted(values)[min(len(values) - 1, int(len(values) * q))]


def main() -> None:
    with open(os.path.join(ROOT, CONFIG), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    inf = cfg.get("inference", {})
    block = inf.get(inf.get("backend")) or {}
    client_max = int(block.get("max_tokens") or 0)
    ans = cfg.get("answer", {}) or {}
    answer_cap = int(ans.get("max_tokens") or 0)
    window = int(ans.get("context_window") or 0)
    hp = cfg["hyperparameters"]

    distil = open(os.path.join(ROOT, "data/prompts/P_distil.txt"), encoding="utf-8").read()

    print(f"tag            : {TAG}")
    print(f"config         : {CONFIG}")
    print(f"k1={hp['k1']}  k2={hp['k2']}  delta={hp['delta']}")
    print(f"client max_tok : {client_max}")
    print(f"answer cap     : {answer_cap}   context_window: {window or 'unset'}")

    rows = []
    with open(os.path.join(PRED_DIR, "ds_nothink__deepseek_v4_flash__ASEM.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    by_conv = defaultdict(list)
    for r in rows:
        by_conv[r["conversation_id"]].append(r["question"])

    embedder = _build_embedder(EMBED_CFG)
    distil_sizes, recovery_sizes, cand_counts = [], [], []
    worst = []

    for conv, questions in sorted(by_conv.items()):
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

        uniq = sorted(set(questions))
        vecs = embedder.embed_documents(uniq)
        cache = {q: np.asarray(v, dtype="float32") for q, v in zip(uniq, vecs)}

        def render(q, notes_):
            payload = json.dumps([AnswerAgent._note_payload(n) for n in notes_])
            return distil.format(query=q, candidates=payload)

        base = HybridRetriever(
            backend=None, k1=hp["k1"], k2=hp["k2"], delta=hp["delta"],
            lambda_weight=hp["lambda"], use_rrf=True, use_bm25=True,
            use_entity_filter=True, use_temporal_boost=True,
            rrf_k=60, max_link_hops=2, enable_link_traversal=True,
        )
        wide = HybridRetriever(
            backend=None, k1=hp["k1"], k2=12, delta=0.15,
            lambda_weight=hp["lambda"], use_rrf=True, use_bm25=True,
            use_entity_filter=True, use_temporal_boost=True,
            rrf_k=60, max_link_hops=2, enable_link_traversal=True,
        )
        base.backend = MemoBackend(cache)
        wide.backend = MemoBackend(cache)

        for q in questions:
            first = base.retrieve(q, bank)
            pool = wide.retrieve(q, bank)
            seen = {n.id for n in first}
            merged = list(first) + [n for n in pool if n.id not in seen]

            t1 = len(render(q, first)) // 4
            t2 = len(render(q, merged)) // 4
            distil_sizes.append(t1)
            recovery_sizes.append(t2)
            cand_counts.append(len(first))
            worst.append((t2, conv, q, len(merged)))
        bank.close()

    def report(label, sizes):
        print(f"\n--- {label} ---")
        print(f"  n={len(sizes)}  min={min(sizes)}  p50={pct(sizes,0.50)}  p90={pct(sizes,0.90)}  "
              f"p99={pct(sizes,0.99)}  max={max(sizes)}  mean={statistics.mean(sizes):.0f}")
        for name, cap in (("answer cap", answer_cap), ("client max_tokens", client_max)):
            if not cap or not window:
                continue
            budget = window - cap
            over = sum(1 for s in sizes if s > budget)
            print(f"  exceed window({window}) - {name}({cap}) = {budget}: {over} "
                  f"({100*over/len(sizes):.1f}%)")

    print(f"\ncandidates per query: min={min(cand_counts)} p50={pct(cand_counts,0.5)} max={max(cand_counts)}")
    report("DISTIL prompt (first pass)", distil_sizes)
    report("DISTIL prompt (recovery: widened + merged pool)", recovery_sizes)

    print("\n--- 10 largest prompts (recovery pool) ---")
    for t, conv, q, n in sorted(worst, reverse=True)[:10]:
        print(f"  ~{t:>5} tok  notes={n:<3} [{conv}] {q[:70]}")

    out = os.path.join(ROOT, "scratch_diag", "_context_size.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"tag={TAG} config={CONFIG}\n")
        fh.write(f"window={window} answer_cap={answer_cap} client_max_tokens={client_max}\n")
        fh.write(f"distil  : {sorted(distil_sizes)}\n")
        fh.write(f"recovery: {sorted(recovery_sizes)}\n")
    print(f"\nfull distribution -> {out}")


if __name__ == "__main__":
    main()
