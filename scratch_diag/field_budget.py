"""Where do the answer-prompt tokens actually go?

Renders the real retrieval context for every question, then accounts for the
prompt size FIELD BY FIELD (`content`, `keywords`, `tags`, `description`,
`entities`, `speaker`, `relations`, `session_date`, `timestamp_iso`, `utility`)
plus the JSON scaffolding, so the payload can be slimmed on evidence instead of
by guesswork.

Also prints a `content` vs `description` sample so the "can we drop the raw turn?"
question can be answered from the bank itself.

Usage:
  python scratch_diag/field_budget.py [TAG] [CONFIG] [SYSTEM]
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

FIELDS = ("content", "description", "keywords", "tags", "entities", "speaker",
          "relations", "session_date", "timestamp_iso", "utility")


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


def main() -> None:
    with open(os.path.join(ROOT, CONFIG), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    hp = cfg["hyperparameters"]

    print(f"tag={TAG}  system={SYSTEM}  config={CONFIG}")
    print(f"k1={hp['k1']} k2={hp['k2']} delta={hp['delta']}")

    pred_file = os.path.join(PRED_DIR, f"ds_nothink__deepseek_v4_flash__{SYSTEM}.jsonl")
    rows = []
    with open(pred_file, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    by_conv = defaultdict(list)
    for r in rows:
        by_conv[r["conversation_id"]].append(r["question"])

    embedder = _build_embedder(EMBED_CFG)
    totals = {f: 0 for f in FIELDS}
    per_note_chars, scaffold_chars, n_notes_total = [], 0, 0
    content_ratio = []          # len(X) / len(c)
    samples = []

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
            payload = [AnswerAgent._note_payload(n) for n in got]
            full = json.dumps(payload)
            n_notes_total += len(payload)
            scaffold_chars += len(full) - sum(
                len(json.dumps(v)) - 2 for p in payload for v in p.values()
            )
            for p in payload:
                for f in FIELDS:
                    totals[f] += len(json.dumps(p[f]) if f in p else 0) - 1
                c_len, x_len = len(p["content"]), len(p.get("description") or "")
                if c_len:
                    content_ratio.append(x_len / c_len)
                    per_note_chars.append(len(json.dumps(p)))
                if len(samples) < 4 and c_len > 200 and x_len > 0:
                    samples.append((conv, q, p["content"][:300], p["description"][:300]))

        bank.close()

    grand = sum(totals.values()) + scaffold_chars
    print(f"\nretrieved notes: {n_notes_total}   "
          f"avg notes/query: {n_notes_total / max(1, len(by_conv)):.1f}   "
          f"avg chars/note: {statistics.mean(per_note_chars):.0f} "
          f"(~{statistics.mean(per_note_chars)/4:.0f} tok)")
    print(f"\n--- chars per field, over ALL retrieved notes ({grand} total) ---")
    for f in sorted(FIELDS, key=lambda x: -totals[x]):
        share = 100 * totals[f] / grand
        per_note = totals[f] / max(1, n_notes_total)
        print(f"  {f:<15} {totals[f]:>12,}  {share:>5.1f}%   {per_note:>7.1f} chars/note")
    print(f"  {'JSON scaffold':<15} {scaffold_chars:>12,}  "
          f"{100*scaffold_chars/grand:>5.1f}%   "
          f"{scaffold_chars/max(1, n_notes_total):>7.1f} chars/note")
    print(f"\ndescription/content length ratio: p50={statistics.median(content_ratio):.2f} "
          f"mean={statistics.mean(content_ratio):.2f}")

    print("\n--- content vs description samples ---")
    for conv, q, c, x in samples:
        print(f"\n[{conv}] {q[:80]}")
        print(f"  content    ({len(c):>4}+ chars): {c}")
        print(f"  description(        ): {x}")
        print(f"  first sentence of content: {c.split('.')[0][:200]}")


if __name__ == "__main__":
    main()
