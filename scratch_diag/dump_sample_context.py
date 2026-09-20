"""Dump the COMPLETE answer prompt for one real question, end to end.

Shows exactly what the backbone receives: the frozen memory bank, the retriever's
output, and the fully rendered prompt — legacy payload vs lean payload, first
pass vs recovery pass.

Usage:
  python scratch_diag/dump_sample_context.py [TAG] [SYSTEM] [CONV] [QUESTION SUBSTRING]
  python scratch_diag/dump_sample_context.py ds_fixed ASEM locomo_0000 "support group"
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import yaml
from asem.answer_agent import AnswerAgent
from asem.backends.langchain_backend import _build_embedder
from asem.memory_bank import MemoryBank
from asem.retriever import HybridRetriever

CONFIG = "configs/models/qwen3_4b_openai.yaml"
EMBED_CFG = {"embedder_provider": "huggingface",
             "embedder_name": "sentence-transformers/all-MiniLM-L6-v2"}

TAG = sys.argv[1] if len(sys.argv) > 1 else "ds_fixed"
SYSTEM = sys.argv[2] if len(sys.argv) > 2 else "ASEM"
CONV = sys.argv[3] if len(sys.argv) > 3 else "locomo_0000"
NEEDLE = sys.argv[4] if len(sys.argv) > 4 else "support group"

BANK = os.path.join(ROOT, "static", "memory_banks", "locomo10", TAG, SYSTEM, CONV,
                    "fast_asem.sqlite" if SYSTEM == "FastASEM" else "asem.sqlite")
PREDS = os.path.join(ROOT, "data", "benchmarks", "results", "static", "locomo10",
                     "preds", f"ds_nothink__deepseek_v4_flash__{SYSTEM}.jsonl")

OUT = []


def emit(line: str = "") -> None:
    """Print best-effort (the Windows console may be cp1252 and reject '->' etc.)."""
    OUT.append(line)
    try:
        print(line)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(line.encode(enc, errors="replace").decode(enc, errors="replace"))


def legacy_payload(note) -> dict:
    return {
        "id": note.id, "keywords": note.K, "tags": note.G, "description": note.X,
        "content": note.c, "utility": note.q, "session_date": note.session_date,
        "timestamp_iso": note.timestamp_iso or (note.t.isoformat() if note.t else None),
        "entities": note.entities, "speaker": note.speaker,
        "relations": [{"relation": l.relation or "linked", "target_id": l.target_id}
                      for l in note.L],
    }


def main() -> None:
    with open(os.path.join(ROOT, CONFIG), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    hp = cfg["hyperparameters"]
    ans = cfg.get("answer", {}) or {}
    limit = int(ans.get("content_char_limit", 200))
    max_notes = int(ans.get("max_context_notes") or 0) or None
    window = int(ans.get("context_window") or 0)
    cap = int(ans.get("max_tokens") or 0)

    # The query as the evaluator sends it (with the conversation prefix).
    query, ref = None, None
    with open(PREDS, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("conversation_id") == CONV and NEEDLE.lower() in row["question"].lower():
                query, ref = row["question"], row.get("reference") or row.get("answer")
                break
    if query is None:
        raise SystemExit(f"no question matching {NEEDLE!r} in {CONV}")

    emit("=" * 100)
    emit(f"COMPLETE ANSWER CONTEXT  |  tag={TAG}  system={SYSTEM}  bank={os.path.relpath(BANK, ROOT)}")
    emit(f"config={CONFIG}  max_context_notes={max_notes}  content_char_limit={limit}  "
         f"window={window} answer_cap={cap}")
    emit("=" * 100)

    tmp = os.path.join(tempfile.mkdtemp(), "bank.sqlite")
    shutil.copy2(BANK, tmp)
    bank = MemoryBank(tmp)
    notes = bank.list_notes()

    backend_embed = _build_embedder(EMBED_CFG)
    e_q = np.asarray(backend_embed.embed_query(query), dtype="float32")

    class EmbedOnly:
        def embed(self, text):
            if text == query:
                return e_q
            return np.asarray(backend_embed.embed_query(text), dtype="float32")

        def generate(self, *a, **k):
            raise RuntimeError("no generate")

    retriever = HybridRetriever(
        backend=EmbedOnly(), k1=hp["k1"], k2=hp["k2"], delta=hp["delta"],
        lambda_weight=hp["lambda"], use_rrf=True, use_bm25=True,
        use_entity_filter=True, use_temporal_boost=True, rrf_k=60,
        max_link_hops=2, enable_link_traversal=True,
    )
    first = retriever.retrieve(query, bank)

    wide = HybridRetriever(
        backend=EmbedOnly(), k1=hp["k1"], k2=12, delta=0.15,
        lambda_weight=hp["lambda"], use_rrf=True, use_bm25=True,
        use_entity_filter=True, use_temporal_boost=True, rrf_k=60,
        max_link_hops=2, enable_link_traversal=True,
    )
    pool = wide.retrieve(query, bank)
    seen = {n.id for n in first}
    merged = list(first) + [n for n in pool if n.id not in seen]

    emit(f"\nBANK      : {len(notes)} notes")
    emit(f"QUESTION  : {query}")
    emit(f"GOLD      : {ref}")
    emit(f"RETRIEVER : {len(first)} candidates (k2={hp['k2']} + link traversal), "
         f"stats={retriever.stats}")
    emit(f"RECOVERY  : widened k2=12/delta=0.15 -> {len(pool)}, merged -> {len(merged)}")
    emit(f"SELECTED  : {len(first[:max_notes]) if max_notes else len(first)} notes reach the prompt")

    distil = open(os.path.join(ROOT, "data/prompts/P_distil.txt"), encoding="utf-8").read()

    def render(notes_, lean: bool, pretty: bool) -> str:
        ids = {n.id for n in notes_}
        payload = [AnswerAgent._note_payload(n, content_chars=limit, in_context=ids)
                   if lean else legacy_payload(n) for n in notes_]
        return distil.format(
            query=query,
            candidates=json.dumps(payload, indent=2 if pretty else None),
        )

    selected = first[:max_notes] if max_notes else first

    for label, notes_, lean in (
        ("BEFORE — legacy payload (every field, every edge)", first, False),
        ("AFTER — lean payload, capped to max_context_notes", selected, True),
    ):
        # Production sends COMPACT json; the pretty form below is for reading.
        compact = render(notes_, lean, pretty=False)
        text = render(notes_, lean, pretty=True)
        emit("\n" + "#" * 100)
        emit(f"# {label}")
        emit(f"# PRODUCTION (compact json): ~{len(compact) // 4} tokens "
             f"({len(compact)} chars), {len(notes_)} notes")
        emit(f"# pretty-printed below:      ~{len(text) // 4} tokens ({len(text)} chars)")
        emit("#" * 100)
        emit(text)

    # The graph-rendered context used by the direct (FastASEM / ASEMv2) path.
    agent = AnswerAgent(
        backend=None, prompt_template=distil,
        baseline_prompt_template=open(
            os.path.join(ROOT, "data/prompts/P_temporal_qa.txt"), encoding="utf-8").read(),
        direct_mode=True, content_char_limit=limit, max_context_notes=max_notes,
    )
    graph = agent._render_graph_context(selected)
    emit("\n" + "#" * 100)
    emit(f"# SAME NOTES via _render_graph_context (FastASEM / ASEMv2 direct path)")
    emit(f"# ~{len(graph) // 4} tokens  ({len(graph)} chars)")
    emit("#" * 100)
    emit(graph)

    out = os.path.join(ROOT, "scratch_diag", "_sample_context.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    emit(f"\nwritten -> {out}")
    bank.close()


if __name__ == "__main__":
    main()
