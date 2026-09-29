# -*- coding: utf-8 -*-
"""Side-by-side retrieval comparison: ASEM-THG (EnhancedHybridRetriever) vs
FastASEM (HybridRetriever-RRF) on the SAME conv-26 (locomo_0000) queries.

Both banks were ingested with the same config hash (deepseek, all-MiniLM-L6-v2),
so their embeddings are comparable. Embeddings only -> NO LLM calls.
Writes scratch_diag/retriever_comparison.md.
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import sys

ROOT = r"C:\Users\Dat Pham\Documents\datpd\master-phenikaa\thesis_master\ASEM-Masters\ASEM-Agentic-Self-Evolution-Memory"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

dotenv = os.path.join(ROOT, ".env")
if os.path.exists(dotenv):
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv, override=False)
    except ImportError:
        pass

import yaml

from asem.backends import build_backend
from asem.enhanced_retriever import EnhancedHybridRetriever
from asem.memory_bank import MemoryBank
from asem.retriever import HybridRetriever

THG_BANK = os.path.join(ROOT, "static/memory_banks/locomo10/ds_thg/ASEM-THG/locomo_0000/asem_thg.sqlite")
FA_BANK = os.path.join(ROOT, "static/memory_banks/locomo10/ds_fixed/FastASEM/locomo_0000/fast_asem.sqlite")
CFG = os.path.join(ROOT, "configs/presets/sota_benchmark.yaml")

out = io.StringIO()
def p(*a): print(*a, file=out)

cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
hp = cfg["hyperparameters"]
rt = cfg["retriever"]
backend = build_backend(cfg["inference"])

rrf_kwargs = dict(
    use_rrf=True,
    use_bm25=rt["use_bm25"], use_entity_filter=rt["use_entity_filter"],
    use_temporal_boost=rt["use_temporal_boost"],
    dense_weight=rt["dense_weight"], bm25_weight=rt["bm25_weight"],
    entity_weight=rt["entity_weight"], temporal_weight=rt["temporal_weight"],
    rrf_k=rt["rrf_k"],
    enable_link_traversal=True, max_link_hops=rt["max_hops"], hop_decay=rt["hop_decay"],
    link_traversal_topn=3,
)
thg = EnhancedHybridRetriever(
    backend=backend, k1=hp["k1"], k2=hp["k2"], delta=hp["delta"], lambda_weight=hp["lambda"],
    max_hops=rt["max_hops"], multi_hop_topn=5,
    alpha=0.35, beta=0.25, gamma=0.40,
    enable_global_semantics=True, enable_intent_q=True, **rrf_kwargs,
)
fast = HybridRetriever(
    backend=backend, k1=hp["k1"], k2=hp["k2"], delta=hp["delta"], lambda_weight=hp["lambda"],
    **rrf_kwargs,
)

def bank_stats(path):
    con = sqlite3.connect(path)
    n = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    edges = 0
    for (l,) in con.execute("SELECT L FROM notes"):
        edges += len(json.loads(l or "[]"))
    con.close()
    return n, edges // 2

def gold_rank(notes, gold, k=8):
    for i, n in enumerate(notes[:k], 1):
        if gold.lower() in (n.c or "").lower():
            return i
    return None

p("# Retriever comparison — ASEM-THG vs FastASEM (conv-26)\n")
p(f"- config: `{os.path.relpath(CFG, ROOT)}`  k1={hp['k1']} k2={hp['k2']} "
  f"delta={hp['delta']} lambda={hp['lambda']} rrf_k={rt['rrf_k']} max_hops={rt['max_hops']}\n")
for name, path in [("ASEM-THG", THG_BANK), ("FastASEM", FA_BANK)]:
    n, e = bank_stats(path)
    p(f"- **{name}** bank: {n} notes, {e} edges  (`{os.path.relpath(path, ROOT)}`)")
p("")

QUERIES = [
    ("Temporal", "When did Caroline go to the LGBTQ support group?", "7 May 2023"),
    ("Single-hop", "What did Caroline research?", "adoption"),
    ("Single-hop (term)", "Where did Caroline move from 4 years ago?", "Sweden"),
    ("Multi-hop", "Would Melanie be considered an ally to the transgender community?", "support"),
    ("Conversational", "What are Caroline's plans for the summer?", "adoption"),
    ("Bare term", "pottery", "pottery"),
]

summary = []
thg_bank = MemoryBank(THG_BANK)
fa_bank = MemoryBank(FA_BANK)
for label, q, gold in QUERIES:
    p(f"\n---\n\n## [{label}] `{q}`  (gold≈'{gold}')\n")
    thg_notes = thg.retrieve(q, thg_bank)
    fa_notes = fast.retrieve(q, fa_bank)
    tr = gold_rank(thg_notes, gold)
    fr = gold_rank(fa_notes, gold)
    summary.append((label, q, gold, tr, fr))
    p(f"- gold in top-8: **ASEM-THG rank {tr}** | **FastASEM rank {fr}**")
    p("")
    p("| rank | ASEM-THG (enhanced) | FastASEM (RRF) |")
    p("|--:|---|---|")
    for i in range(4):
        a = thg_notes[i].c[:95] if i < len(thg_notes) else ""
        b = fa_notes[i].c[:95] if i < len(fa_notes) else ""
        p(f"| {i+1} | {a} | {b} |")

p("\n---\n\n## Summary — gold rank (top-8, lower is better; `-` = miss)\n")
p("| query | ASEM-THG | FastASEM |")
p("|---|--:|--:|")
for label, q, gold, tr, fr in summary:
    tm = tr if tr else "–"
    fm = fr if fr else "–"
    p(f"| {label}: {q[:52]} | {tm} | {fm} |")

thg_bank.close()
fa_bank.close()

with open(os.path.join(ROOT, "scratch_diag", "retriever_comparison.md"), "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print("wrote scratch_diag/retriever_comparison.md")
rng = [f"{l}: THG={tr} FA={fr}" for l, q, g, tr, fr in summary]
print("\n".join(rng))
