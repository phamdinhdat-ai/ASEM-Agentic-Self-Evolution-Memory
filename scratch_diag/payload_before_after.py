"""Before/after answer-prompt size: the FULL bank row vs the LEAN payload.

Renders the real retrieval context for every LoCoMo question twice — once with
the legacy payload (every field, every graph edge) and once with the current
`AnswerAgent._note_payload` — and reports token percentiles plus the per-field
breakdown, so the savings are measured rather than asserted.

Usage:
  python scratch_diag/payload_before_after.py [TAG] [CONFIG] [SYSTEM]
"""
from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict

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
SYSTEM = sys.argv[3] if len(sys.argv) > 3 else "ASEM"
BANK_ROOT = os.path.join(ROOT, "static", "memory_banks", "locomo10", TAG, SYSTEM)


def legacy_payload(note) -> dict:
    """The payload as it was BEFORE the budget work: nothing pruned."""
    return {
        "id": note.id,
        "keywords": note.K,
        "tags": note.G,
        "description": note.X,
        "content": note.c,
        "utility": note.q,
        "session_date": note.session_date,
        "timestamp_iso": note.timestamp_iso or (note.t.isoformat() if note.t else None),
        "entities": note.entities,
        "speaker": note.speaker,
        "relations": [
            {"relation": link.relation or "linked", "target_id": link.target_id}
            for link in note.L
        ],
    }


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
                    idf = np.log(1.0 + (N - df.get(tok, 0) + 0.5) / (df.get(tok, 0) + 0.5))
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
    return sorted(values)[min(len(values) - 1, int(len(values) * q))]


def main() -> None:
    with open(os.path.join(ROOT, CONFIG), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    hp = cfg["hyperparameters"]
    ans = cfg.get("answer", {}) or {}
    window = int(ans.get("context_window") or 0)
    cap = int(ans.get("max_tokens") or 0)
    limit = int(ans.get("content_char_limit", 200))
    max_notes = int(ans.get("max_context_notes") or 0) or None
    distil = open(os.path.join(ROOT, "data/prompts/P_distil.txt"), encoding="utf-8").read()

    print(f"tag={TAG} system={SYSTEM} config={CONFIG}")
    print(f"window={window or 'unset'} answer_cap={cap} "
          f"max_context_notes={max_notes} content_char_limit={limit}")

    rows = []
    with open(os.path.join(PRED_DIR, f"ds_nothink__deepseek_v4_flash__{SYSTEM}.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    by_conv = defaultdict(list)
    for r in rows:
        by_conv[r["conversation_id"]].append(r["question"])

    embedder = _build_embedder(EMBED_CFG)
    before, after = [], []
    counts = []

    for conv, questions in sorted(by_conv.items()):
        src = os.path.join(BANK_ROOT, conv, "asem.sqlite")
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
        cache = {q: np.asarray(v, dtype="float32")
                 for q, v in zip(uniq, embedder.embed_documents(uniq))}
        retriever = HybridRetriever(
            backend=MemoBackend(cache), k1=hp["k1"], k2=hp["k2"], delta=hp["delta"],
            lambda_weight=hp["lambda"], use_rrf=True, use_bm25=True,
            use_entity_filter=True, use_temporal_boost=True, rrf_k=60,
            max_link_hops=2, enable_link_traversal=True,
        )

        for q in questions:
            got = retriever.retrieve(q, bank)
            if not got:
                continue
            selected = got[: max_notes] if max_notes else got
            counts.append(len(selected))

            legacy = distil.format(
                query=q, candidates=json.dumps([legacy_payload(n) for n in got]))
            lean = distil.format(
                query=q,
                candidates=json.dumps([
                    AnswerAgent._note_payload(
                        n, content_chars=limit, in_context={m.id for m in selected})
                    for n in selected
                ]),
            )
            before.append(len(legacy) // 4)
            after.append(len(lean) // 4)

        bank.close()

    def report(label, sizes):
        print(f"\n--- {label} ---")
        print(f"  n={len(sizes)}  min={min(sizes)}  p50={pct(sizes,0.50)}  "
              f"p90={pct(sizes,0.90)}  p99={pct(sizes,0.99)}  max={max(sizes)}  "
              f"mean={statistics.mean(sizes):.0f}")
        if window and cap:
            budget = window - cap
            over = sum(1 for s in sizes if s > budget)
            print(f"  exceed window({window}) - cap({cap}) = {budget}: "
                  f"{over} ({100*over/len(sizes):.1f}%)")

    print(f"\nnotes per query: min={min(counts)} p50={pct(counts,0.5)} max={max(counts)}")
    report("BEFORE (legacy payload, all notes)", before)
    report("AFTER  (lean payload, capped)", after)
    cut = 100 * (1 - statistics.mean(after) / statistics.mean(before))
    print(f"\ntoken reduction: mean {statistics.mean(before):.0f} -> "
          f"{statistics.mean(after):.0f} tok  (-{cut:.1f}%)")


if __name__ == "__main__":
    main()
