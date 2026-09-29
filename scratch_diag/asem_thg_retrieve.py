# -*- coding: utf-8 -*-
"""ASEM-THG retrieval test: run sample queries through the THG retriever and
show the retrieved notes, scores, channels, and graph stats.

Uses the pre-built static bank (conv-26 / locomo_0000). Embeddings only, so
NO LLM calls. Writes scratch_diag/asem_thg_retrieve.md.
"""
from __future__ import annotations

import io
import json
import os
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
        for line in open(dotenv, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import numpy as np
import yaml

from asem.backends import build_backend
from asem.enhanced_retriever import EnhancedHybridRetriever
from asem.memory_bank import MemoryBank
from asem.retriever import HybridRetriever

BANK = os.path.join(ROOT, "static", "memory_banks", "locomo10", "ds_thg",
                    "ASEM-THG", "locomo_0000", "asem_thg.sqlite")
CFG = os.path.join(ROOT, "configs", "models", "deepseek_api.yaml")

out = io.StringIO()
def p(*a): print(*a, file=out)

# ---- build -----------------------------------------------------------------
cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
hp = cfg["hyperparameters"]
backend = build_backend(cfg["inference"])
bank = MemoryBank(BANK)

retriever = EnhancedHybridRetriever(
    backend=backend,
    k1=hp["k1"], k2=hp["k2"], delta=hp["delta"], lambda_weight=hp["lambda"],
    max_hops=2, hop_decay=0.7, multi_hop_topn=5,
    alpha=0.35, beta=0.25, gamma=0.40,
    enable_global_semantics=True, enable_intent_q=True,
)

p("# ASEM-THG retrieval test\n")
p(f"- bank: `{os.path.relpath(BANK, ROOT)}`  ({bank.size()} notes)")
p(f"- retriever: `EnhancedHybridRetriever`  k1={hp['k1']} k2={hp['k2']} "
  f"delta={hp['delta']} lambda={hp['lambda']} max_hops=2 alpha/beta/gamma=0.35/0.25/0.40")
p(f"- model config: `{os.path.relpath(CFG, ROOT)}`  (embedder only; no LLM calls)\n")

def cos(a, b):
    a = np.asarray(a, dtype="float32"); b = np.asarray(b, dtype="float32")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0

def show(notes, e_q, limit=5):
    p(f"| # | sim(e_q,e) | intent(e_q,z) | q | session_date | fact | entities |")
    p("|--:|--:|--:|--:|---|---|---|")
    for i, n in enumerate(notes[:limit], 1):
        ents = ", ".join(n.entities or [])
        fact = (n.c or "").replace("|", "/")[:120]
        p(f"| {i} | {cos(e_q, n.e):.3f} | "
          f"{cos(e_q, n.z):.3f} | {n.q:.2f} | "
          f"{n.session_date or ''} | {fact} | {ents} |")

# ---- raw channel demos -----------------------------------------------------
p("## Raw retrieval channels (what the RRF fuse consumes)\n")
try:
    bm = bank.bm25_search("Sweden", k=5)
    p("**BM25 channel** — query `Sweden` (top 5):")
    for i, (score, n) in enumerate(bm, 1):
        p(f"{i}. ({score:.2f}) {n.c[:110]}")
except Exception as e:  # noqa: BLE001
    p(f"BM25 channel error: {type(e).__name__}: {e}")

p("")
try:
    ent = bank.search_by_entities(["Melanie", "pottery"], k=5)
    p("**Entity channel** — entities `['Melanie','pottery']` (top 5):")
    for i, n in enumerate(ent, 1):
        p(f"{i}. {n.c[:110]}  (ents: {', '.join(n.entities or [])})")
except Exception as e:  # noqa: BLE001
    p(f"Entity channel error: {type(e).__name__}: {e}")

# ---- sample queries --------------------------------------------------------
QUERIES = [
    ("Temporal (when)", "When did Caroline go to the LGBTQ support group?"),
    ("Bare term", "Sweden"),
    ("Bare term", "pottery"),
    ("Single-hop", "What did Caroline research?"),
    ("Multi-hop", "Would Melanie be considered an ally to the transgender community?"),
    ("Conversational", "What are Caroline's plans for the summer?"),
]

for label, q in QUERIES:
    p(f"\n---\n\n## [{label}] `{q}`\n")
    try:
        e_q = backend.embed(q)
        base = HybridRetriever.retrieve(retriever, q, bank)
        base_stats = dict(retriever.stats)
        enh = retriever.retrieve(q, bank)
        enh_stats = dict(retriever.stats)
        p(f"- base RRF (Phase A+B): **{len(base)}** notes "
          f"(phase_a_hits={base_stats.get('phase_a_hits')})")
        p(f"- after multi-hop + global re-rank: **{len(enh)}** notes "
          f"(multi_hop_added={enh_stats.get('multi_hop_added')}, "
          f"max_hops={enh_stats.get('max_hops')})")
        p(f"\n**Top retrieved (enhanced):**\n")
        show(enh, e_q, limit=6)
    except Exception as e:  # noqa: BLE001
        p(f"ERROR: {type(e).__name__}: {e}")

bank.close()
with open(os.path.join(ROOT, "scratch_diag", "asem_thg_retrieve.md"), "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print("wrote scratch_diag/asem_thg_retrieve.md")
