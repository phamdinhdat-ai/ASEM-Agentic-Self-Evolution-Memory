"""Stage 3+4 — efficient recall measurement AND link-graph evidence.

PART A: For every ASEM QA row, find bank notes containing the gold answer and
        measure whether the real retriever returns one of them into the S4
        context, comparing the eval query ("Conversation between X and Y.
        Question: ...") with the bare question.

PART B: Show the relational graph that the answer agent NEVER sees:
        - relation-label distribution across the frozen ASEM banks
        - for adversarial questions whose gold answer IS in the bank, print the
          gold note's speaker/entities and its typed links (the info that would
          let the model resolve "who did this").

Perf note: the production `bm25_search`/`search_by_entities` call `list_notes()`
(re-deserialising every 384-d embedding) on EVERY query. Here we build those
indexes once per conversation, and batch-embed all queries up front.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
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


def _doc_text(n) -> str:
    return " ".join([n.c, " ".join(n.K), " ".join(n.G), n.X, " ".join(n.entities)]).lower()


def make_cached_bm25(notes):
    docs = [re.findall(r"\w+", _doc_text(n)) for n in notes]
    tcs = []
    df = {}
    for d in docs:
        tc = {}
        for t in d:
            tc[t] = tc.get(t, 0) + 1
        tcs.append(tc)
        for t in tc:
            df[t] = df.get(t, 0) + 1
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, N)
    k1, b = 1.5, 0.75

    def bm25_search(query, k=10):
        tokens = re.findall(r"\w+", query.lower())
        out = []
        for i, tc in enumerate(tcs):
            score = 0.0
            for tok in tokens:
                if tok in tc:
                    f = tc[tok]
                    idf = math.log(1.0 + (N - df.get(tok, 0) + 0.5) / (df.get(tok, 0) + 0.5))
                    score += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * (len(docs[i]) / avgdl)))
            if score > 0:
                out.append((score, notes[i]))
        out.sort(key=lambda x: x[0], reverse=True)
        return out[:k]

    return bm25_search


def make_cached_entities(notes):
    idx = [({e.lower() for e in n.entities}, n) for n in notes]

    def search_by_entities(entities, k=10):
        norm = {e.strip().lower() for e in entities if e.strip()}
        if not norm:
            return []
        m = [(len(norm & ne), n) for ne, n in idx if norm & ne]
        m.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in m[:k]]

    return search_by_entities


class MemoBackend:
    def __init__(self, cache):
        self.cache = cache

    def embed(self, text):
        return self.cache[text]

    def generate(self, *a, **k):
        raise RuntimeError("no generate")


def relation_stats(conv: str) -> Counter:
    path = os.path.join(BANK_ROOT, "ASEM", conv, "asem.sqlite")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    c = Counter()
    for r in conn.execute("SELECT L FROM notes"):
        try:
            for lr in json.loads(r["L"] or "[]"):
                rel = lr.get("relation") if isinstance(lr, dict) else "linked"
                c[rel] += 1
        except Exception:
            pass
    conn.close()
    return c


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

    agg = Counter()
    by_idk = defaultdict(Counter)
    by_cat = defaultdict(Counter)
    prefix_examples = []
    graph_examples = []

    for conv, crows in sorted(by_conv.items()):
        src = os.path.join(BANK_ROOT, "ASEM", conv, "asem.sqlite")
        if not os.path.exists(src):
            continue
        tmp = os.path.join(tempfile.mkdtemp(), "bank.sqlite")
        shutil.copy2(src, tmp)
        bank = MemoryBank(tmp)
        notes = bank.list_notes()
        nid_to_note = {n.id: n for n in notes}
        bank.list_notes = lambda _n=notes: _n
        bank.bm25_search = make_cached_bm25(notes)
        bank.search_by_entities = make_cached_entities(notes)

        blob_toks = [(n.id, toks(" ".join(str(x) for x in [n.c, n.K, n.G, n.X, n.entities, n.session_date] if x)))
                     for n in notes]

        # decide gold per row, collect texts to embed
        gold_map = {}
        texts = set()
        for r in crows:
            rt = toks(r["ref"] or "")
            if not rt:
                continue
            gold = {nid for nid, bt in blob_toks if len(rt & bt) / len(rt) >= 0.6}
            agg["ref_usable"] += 1
            if gold:
                agg["gold_in_bank"] += 1
                gold_map[r["idx"]] = gold
                texts.add(r["query"])
                texts.add(r["question"])
            else:
                agg["gold_missing_bank"] += 1

        if texts:
            vecs = embedder.embed_documents(list(texts))
            cache = {t: np.asarray(v, dtype="float32") for t, v in zip(texts, vecs)}
            retr.backend = MemoBackend(cache)

        for r in crows:
            gold = gold_map.get(r["idx"])
            if not gold:
                continue
            is_idk = bool(IDK_RE.match(r["pred"] or ""))
            got_p = {n.id for n in retr.retrieve(r["query"], bank)}
            got_b = {n.id for n in retr.retrieve(r["question"], bank)}
            hp, hb = bool(got_p & gold), bool(got_b & gold)
            bucket = "idk" if is_idk else "answered"
            for key in (bucket, "all"):
                by_idk[key]["n"] += 1
                by_idk[key]["prefix"] += hp
                by_idk[key]["bare"] += hb
            by_cat[r["category_name"]]["n"] += 1
            by_cat[r["category_name"]]["prefix"] += hp
            by_cat[r["category_name"]]["bare"] += hb
            if hb and not hp and len(prefix_examples) < 12:
                prefix_examples.append((conv, r["idx"], r["category_name"], r["question"]))
            if is_idk and r["category_name"] == "adversarial" and len(graph_examples) < 4:
                gid = sorted(gold)[0]
                gn = nid_to_note.get(gid)
                if gn is not None:
                    graph_examples.append((conv, r, gn, [l.relation for l in gn.L]))
        bank.close()

    def pct(a, b):
        return f"{100*a/b:.1f}%" if b else "n/a"

    print("=" * 78)
    print("PART A — retrieval recall of the gold note (ASEM frozen banks)")
    print("=" * 78)
    print(f"rows with usable ref      : {agg['ref_usable']}")
    print(f"  gold answer IN bank     : {agg['gold_in_bank']} ({pct(agg['gold_in_bank'], agg['ref_usable'])})")
    print(f"  gold answer NOT in bank : {agg['gold_missing_bank']} ({pct(agg['gold_missing_bank'], agg['ref_usable'])})")
    print("\nrecall among rows where the gold note IS in the bank:")
    for key in ("all", "answered", "idk"):
        c = by_idk[key]
        print(f"  {key:<9} n={c['n']:<5} prefixed={pct(c['prefix'], c['n']):>6}   bare={pct(c['bare'], c['n']):>6}")
    print("\nby category (prefixed / bare):")
    for cat, c in sorted(by_cat.items()):
        print(f"  {cat:<14} n={c['n']:<5} {pct(c['prefix'], c['n']):>6} / {pct(c['bare'], c['n']):>6}")
    print("\n--- bare retrieves gold but eval-prefix does NOT (prefix cost) ---")
    for conv, idx, cat, q in prefix_examples:
        print(f"  [{conv} #{idx} {cat}] {q}")

    print("\n" + "=" * 78)
    print("PART B — the relational graph the answer agent never sees")
    print("=" * 78)
    all_rel = Counter()
    for conv in sorted(by_conv):
        if os.path.exists(os.path.join(BANK_ROOT, "ASEM", conv, "asem.sqlite")):
            all_rel += relation_stats(conv)
    print("relation-label distribution across ASEM banks:", dict(all_rel))
    print("\nanswer context carries: c + session_date + entities + keywords + description.")
    print("answer context NEVER carries: L (typed links / relations). See answer_agent.direct_answer.")

    print("\n--- adversarial IDK cases: gold note IS in the bank, with its typed links ---")
    for conv, r, gn, rels in graph_examples:
        print(f"\n  [{conv} #{r['idx']}] Q: {r['question']}")
        print(f"      ref        : {r['ref']!r}")
        print(f"      pred       : {(r['pred'] or '')[:110]!r}")
        print(f"      gold note  : speaker={gn.speaker!r} entities={gn.entities}")
        print(f"      content    : {gn.c[:220]!r}")
        print(f"      relations  : {rels}")
    print("\n(Every gold note above has typed edges; none of those relation labels")
    print(" reach the prompt, so the model cannot resolve attribution/adversarial swaps.)")


if __name__ == "__main__":
    main()
