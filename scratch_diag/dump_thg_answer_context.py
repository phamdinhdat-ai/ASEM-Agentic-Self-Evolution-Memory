# -*- coding: utf-8 -*-
"""Dump the ACTUAL context the ASEM-THG answer agent receives, for sample
questions where the relevant evidence IS in the bank.

Replays the frozen ds_thg bank with a **probe backend** that records the prompt
and returns a canned, non-abstention answer -> NO LLM calls.

Writes scratch_diag/asem_thg_answer_context.md.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = r"C:\Users\Dat Pham\Documents\datpd\master-phenikaa\thesis_master\ASEM-Masters\ASEM-Agentic-Self-Evolution-Memory"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from asem.backends.langchain_backend import _build_embedder
from eval.systems import build_system, strip_query_prefix

BANK = os.path.join(ROOT, "static/memory_banks/locomo10/ds_thg/ASEM-THG/locomo_0000/asem_thg.sqlite")
CFG = "configs/models/deepseek_openai.yaml"
PREDS = os.path.join(ROOT, "data/benchmarks/results/static/locomo10/preds/ds_thg__deepseek_v4_flash__ASEM-THG.jsonl")
EMBED_CFG = {"embedder_provider": "huggingface",
             "embedder_name": "sentence-transformers/all-MiniLM-L6-v2"}
PLACEHOLDER = ("PROBE ANSWER: placeholder so no recovery pass fires; never scored.")

TARGETS = sys.argv[1:] or [
    "What do sunflowers represent according to Caroline?",
    "Would Caroline pursue writing as a career option?",
    "Would Melanie be considered a member of the LGBTQ community?",
    "How many children does Melanie have?",
    "When did Caroline go to the adoption meeting?",
]


class ProbeBackend:
    default_max_tokens = 512

    def __init__(self, embedder):
        self.embedder = embedder
        self._cache = {}
        self.prompts = []

    def embed(self, text):
        if text not in self._cache:
            self._cache[text] = np.asarray(
                self.embedder.embed_documents([text])[0], dtype="float32")
        return self._cache[text]

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if "selected_ids" in prompt:
            return json.dumps({"selected_ids": [], "answer": PLACEHOLDER})
        return PLACEHOLDER


def load_preds():
    rows = {}
    for line in open(PREDS, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["question"]] = r
    return rows


def main() -> None:
    out = io.StringIO()
    def p(*a): print(*a, file=out)

    # ---- stage the frozen bank so the dump never writes to it ----
    tmp = tempfile.mkdtemp(prefix="thgctx_")
    shutil.copy2(BANK, os.path.join(tmp, "asem_thg.sqlite"))

    embedder = _build_embedder(EMBED_CFG)
    probe = ProbeBackend(embedder)
    system = build_system("ASEM-THG", CFG, tmp, backend=probe)
    bank = system.pipeline.memory_bank
    retriever = system.pipeline.retriever

    preds = load_preds()

    p("# ASEM-THG — answer-agent context dump\n")
    p(f"- bank: `{os.path.relpath(BANK, ROOT)}`  ({bank.size()} notes)")
    p(f"- retriever: `EnhancedHybridRetriever`  direct_mode={getattr(system.pipeline.answer_agent, 'direct_mode', '?')}")
    p(f"- probe backend: prompts recorded, no LLM calls\n")

    for q in TARGETS:
        row = preds.get(q)
        if row is None:
            p(f"\n---\n\n## `{q}`\n\n_(not found in preds)_\n")
            continue
        s = row["conversation_id"]
        gold = str(row["ref"])
        probe.prompts = []
        system.answer(q)                      # -> pipeline.read_path -> direct_answer
        prompt = probe.prompts[-1] if probe.prompts else ""

        # retrieved notes (same call path the pipeline used)
        notes = retriever.retrieve(strip_query_prefix(q), bank)

        p(f"\n---\n\n## [{row['category_name']}] `{q}`\n")
        p(f"- **gold**: `{gold}`")
        p(f"- **baseline pred** (distil mode, pre-P0): `{row['pred']}`  (em={row['em']}, em_loose={row['em_loose']})")
        p(f"- **retrieved**: {len(notes)} notes\n")
        p("### Retrieved notes (rank order)\n")
        p("| # | date | fact | entities | gold here? |")
        p("|--:|---|---|---|---|")
        gold_toks = {w for w in re.findall(r"[a-z0-9]+", gold.lower()) if len(w) > 2}
        rank_hit = None
        for i, n in enumerate(notes[:10], 1):
            ents = ", ".join(n.entities or [])
            fact = (n.c or "").replace("|", "/")[:110]
            toks = set(re.findall(r"[a-z0-9]+", (n.c or "").lower()))
            hit = bool(gold_toks and gold_toks <= toks) or (gold.lower() in (n.c or "").lower())
            if hit and rank_hit is None:
                rank_hit = i
            p(f"| {i} | {n.session_date or ''} | {fact} | {ents} | {'✔' if hit else ''} |")
        p(f"\n**diagnosis:** gold evidence in retrieved set at rank **{rank_hit}** "
          f"({'GENERATION-bound' if rank_hit else 'RETRIEVAL-bound'})\n")

        # the exact context block the answer agent received
        marker = "Memory Notes:"
        ctx = prompt.split(marker, 1)[1].strip() if marker in prompt else prompt
        p("### Exact context the answer agent received (`{context}`)\n")
        p("```")
        p(ctx if len(ctx) < 6000 else ctx[:6000] + "\n...[truncated]")
        p("```")

    bank.close()
    with open(os.path.join(ROOT, "scratch_diag", "asem_thg_answer_context.md"), "w", encoding="utf-8") as f:
        f.write(out.getvalue())
    print("wrote scratch_diag/asem_thg_answer_context.md")


if __name__ == "__main__":
    main()
