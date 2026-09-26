"""A/B the retriever algorithm changes on the frozen banks (LLM-free).

For every sampled question whose gold answer IS in the bank, run the real
`HybridRetriever` under several configurations and report whether a gold-bearing
note reaches the returned context, plus the context cost.

Variants
  legacy    masked=off, hops=1, dedupe=off   (≈ what the Qwen3-4B run used)
  +masked   person names masked out for a second retrieval channel
  +hops2    true 2-hop link traversal (hop_decay)
  +dedupe   drop notes that restate a higher-ranked one
  all       masked + hops2 + dedupe

Usage:
  python scratch_diag/retrieval_variants.py [--per-cat N] [--tag ds_fixed]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from asem.backends.langchain_backend import _build_embedder  # noqa: E402
from asem.memory_bank import MemoryBank  # noqa: E402
from asem.retriever import HybridRetriever, mask_person_names  # noqa: E402

MODEL = "qwen_qwen3_4b_instruct_2507"
RES = os.path.join(ROOT, "data", "benchmarks", "results", "static", "locomo10")
EMBED_CFG = {"embedder_provider": "huggingface",
             "embedder_name": "sentence-transformers/all-MiniLM-L6-v2"}
HARNESS_TO_REAL = {"adversarial": "adversarial", "temporal": "temporal",
                   "commonsense": "open_domain", "single_hop": "multi_hop",
                   "conversational": "single_hop"}
STOP = set("""a an the of to in on at for with and or is are was were be been being it its this that
his her their they he she them i you we my your our as by from into about over under had has have did
do does not no but if then than so such very more most much many""".split())

VARIANTS = {
    "legacy": dict(use_masked_query=False, max_link_hops=1, use_dedupe=False),
    "+masked": dict(use_masked_query=True, max_link_hops=1, use_dedupe=False),
    "+hops2": dict(use_masked_query=False, max_link_hops=2, use_dedupe=False),
    "+dedupe": dict(use_masked_query=False, max_link_hops=1, use_dedupe=True),
    "all": dict(use_masked_query=True, max_link_hops=2, use_dedupe=True),
}


def toks(text):
    return {w for w in __import__("re").findall(r"[a-z0-9']+", (text or "").lower())
            if w not in STOP and len(w) > 2}


class MemoBackend:
    def __init__(self):
        self.cache = {}

    def embed(self, text):
        return self.cache[text]

    def generate(self, *a, **k):
        raise RuntimeError("no generate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ds_fixed")
    ap.add_argument("--system", default="ASEM")
    ap.add_argument("--per-cat", type=int, default=300, help="max rows per category")
    args = ap.parse_args()

    bank_root = os.path.join(ROOT, "static", "memory_banks", "locomo10", args.tag, args.system)
    rows = [json.loads(l) for l in open(
        os.path.join(RES, "preds", f"{args.tag}__{MODEL}__{args.system}.jsonl"), encoding="utf-8")]
    by_conv = defaultdict(list)
    for r in rows:
        by_conv[r["conversation_id"]].append(r)

    embedder = _build_embedder(EMBED_CFG)
    retriever = HybridRetriever(
        backend=MemoBackend(), k1=20, k2=5, delta=0.30, lambda_weight=0.40,
        use_rrf=True, use_bm25=True, use_entity_filter=True, use_temporal_boost=True,
        dense_weight=1.0, bm25_weight=0.8, entity_weight=0.6, temporal_weight=0.5,
        rrf_k=60, link_traversal_topn=3, enable_link_traversal=True,
    )

    per_cat: Counter = Counter()
    recall = {name: Counter() for name in VARIANTS}
    ctx_chars = {name: [] for name in VARIANTS}
    notes_out = {name: [] for name in VARIANTS}
    masked_rows = 0

    for conv, crows in sorted(by_conv.items()):
        src = os.path.join(bank_root, conv, "asem.sqlite")
        if not os.path.exists(src):
            continue
        tmp = os.path.join(tempfile.mkdtemp(), "bank.sqlite")
        shutil.copy2(src, tmp)
        bank = MemoryBank(tmp)
        notes = bank.list_notes()
        bank.list_notes = lambda _n=notes: _n
        blobs = [(n.id, toks(" ".join(str(x) for x in
                                     [n.c, n.K, n.G, n.X, n.entities, n.session_date] if x)))
                 for n in notes]

        picked = []
        for r in crows:
            cat = HARNESS_TO_REAL.get(r.get("category_name", ""), r.get("category_name", ""))
            if per_cat[cat] >= args.per_cat:
                continue
            per_cat[cat] += 1
            picked.append(r)
        if not picked:
            bank.close()
            continue

        questions = {r["question"] for r in picked}
        vecs = embedder.embed_documents(list(questions))
        cache = {q: np.asarray(v, dtype="float32") for q, v in zip(questions, vecs)}
        # the retriever embeds the bare question AND (for the masked variant) the
        # masked question, so pre-compute both to keep the sweep fast
        for q in questions:
            m = mask_person_names(q)
            if m and m not in cache:
                cache[m] = np.asarray(embedder.embed_documents([m])[0], dtype="float32")
        retriever.backend = MemoBackend()
        retriever.backend.cache = cache

        for r in picked:
            rt = toks(r["ref"])
            if not rt:
                continue
            gold = {nid for nid, bt in blobs if len(rt & bt) / len(rt) >= 0.6}
            if not gold:
                continue  # ingestion loss: no retriever can help
            cat = HARNESS_TO_REAL.get(r.get("category_name", ""), r.get("category_name", ""))
            if mask_person_names(r["question"]):
                masked_rows += 1
            for name, knobs in VARIANTS.items():
                for k, v in knobs.items():
                    setattr(retriever, k, v)
                got = retriever.retrieve(r["question"], bank)
                titles = {n.id for n in got}
                recall[name][cat + "/n"] += 1
                if gold & titles:
                    recall[name][cat + "/hit"] += 1
                ctx_chars[name].append(sum(len(n.X or "") + len(n.c or "") for n in got))
                notes_out[name].append(len(got))
        bank.close()

    print(f"\ntag={args.tag} system={args.system}  rows with gold in bank: "
          f"{recall['legacy']['adversarial/n'] + recall['legacy']['single_hop/n'] + recall['legacy']['multi_hop/n'] + recall['legacy']['open_domain/n'] + recall['legacy']['temporal/n']}"
          f"  (questions with a maskable name: {masked_rows})")
    cats = ["adversarial", "single_hop", "multi_hop", "temporal", "open_domain"]
    print(f"\n{'variant':<10}" + "".join(f"{c[:11]:>13}" for c in cats) + f"{'ALL':>9}{'notes':>8}{'chars p50':>11}")
    for name in VARIANTS:
        cells = []
        tot_hit = tot_n = 0
        for c in cats:
            hit, n = recall[name][f"{c}/hit"], recall[name][f"{c}/n"]
            tot_hit += hit
            tot_n += n
            cells.append(f"{100 * hit / n:>12.1f}%" if n else f"{'-':>13}")
        print(f"{name:<10}" + "".join(cells) + f"{100 * tot_hit / max(1, tot_n):>8.1f}%"
              f"{statistics.mean(notes_out[name]):>8.1f}{statistics.median(ctx_chars[name]):>11.0f}")

    print("\nper-variant delta vs legacy (recall, ALL categories):")
    base = sum(recall['legacy'][f"{c}/hit"] for c in cats) / max(1, sum(recall['legacy'][f"{c}/n"] for c in cats))
    for name in VARIANTS:
        r = sum(recall[name][f"{c}/hit"] for c in cats) / max(1, sum(recall[name][f"{c}/n"] for c in cats))
        print(f"  {name:<10} {100 * r:>6.1f}%   ({100 * (r - base):+.1f} pp)")


if __name__ == "__main__":
    main()
