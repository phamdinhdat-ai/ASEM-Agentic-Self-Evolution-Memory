"""System builders for evaluation runners.

Each system (baseline or ASEM) gets its own isolated MemoryBank so that
evaluations are independent and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Dict, List

import yaml

from asem.answer_agent import AnswerAgent
from asem.backends import build_backend
from asem.link_evolver import LinkEvolver
from asem.memory_bank import MemoryBank
from asem.memory_manager import MemoryManager
from asem.note import NoteConstructor
from asem.pipeline import ASEMPipeline
from asem.retriever import HybridRetriever
from asem.utility_updater import UtilityUpdater
from eval.baselines import (
    AtomicLinking,
    FullContext,
    NoMemory,
    RLManagerOnly,
    SimRetrieval,
    ValueRetrievalOnly,
)


@dataclass
class ASEMSystem:
    """Wrapper that exposes the ASEM pipeline as a baseline-style interface."""

    pipeline: ASEMPipeline

    def answer(self, query: str, history: List[str]) -> str:
        for item in history:
            self.pipeline.write_path(item, datetime.utcnow())
        _, answer = self.pipeline.read_path(query)
        return answer

    def reset(self) -> None:
        """Clear the pipeline's memory bank between conversations."""
        self.pipeline.memory_bank.clear()


@dataclass
class ASEMSystemV2:
    """Two-phase ASEM system: pre-ingest once, then retrieval-only per QA.

    **Phase 1 (offline)**: Call ``ingest_conversation(turns)`` once to build
    the full knowledge graph from all dialogue turns.

    **Phase 2 (online)**: Call ``answer(query)`` for each QA pair — NO
    re-ingestion, retrieval-only.

    This eliminates the deduplication bug in ASEMSystem v1 where the same
    turns were re-ingested for every QA pair.
    """

    pipeline: ASEMPipeline
    batch_ingestor: object  # BatchIngestor (lazy import to avoid circular deps)

    # ---- private --------------------------------------------------------
    _ingested: bool = False

    def ingest_conversation(self, dialogue_turns: List[str]) -> int:
        """Pre-ingest all dialogue turns ONCE before any QA queries.

        Returns the number of notes created.
        """
        from asem.batch_ingestion import BatchIngestor
        notes = self.batch_ingestor.ingest_conversation(
            dialogue_turns, self.pipeline.memory_bank,
        )
        self._ingested = True
        return len(notes)

    def answer(self, query: str, history: List[str] = None) -> str:
        """Retrieve + answer from the pre-built knowledge graph.

        The ``history`` parameter is accepted for interface compatibility but
        **ignored** — all turns must be ingested via ``ingest_conversation()``
        before calling this method.
        """
        # If not yet ingested and history is provided, auto-ingest
        if not self._ingested and history:
            self.ingest_conversation(history)

        _, answer = self.pipeline.read_path(query)
        return answer

    def reset(self) -> None:
        """Clear the memory bank for the next conversation."""
        self.pipeline.memory_bank.clear()
        self._ingested = False


# ---------------------------------------------------------------------------
# Shared prompt templates (extracted once)
# ---------------------------------------------------------------------------

_NO_MEMORY_PROMPT = (
    "Answer the following question as concisely as possible. "
    "Give only the answer (a few words or one sentence). "
    "If you are not sure, provide your best guess rather than asking for more information.\n\n"
    "Question: {query}\n\nAnswer:"
)

_FULL_CONTEXT_PROMPT = (
    "Use the conversation excerpts below to answer the question. "
    "Reply with only the answer — a few words or one sentence, no explanation.\n\n"
    "Conversation:\n{context}\n\n"
    "Question: {query}\n\nAnswer:"
)

_RETRIEVAL_PROMPT = (
    "Use the retrieved memory notes below to answer the question. "
    "Reply with only the answer — a few words or one sentence, no explanation.\n\n"
    "Memory:\n{context}\n\n"
    "Question: {query}\n\nAnswer:"
)

_MEMORY_MANAGER_PROMPT = (
    "Decide memory write operation. Output JSON:\n"
    '{{"op": "ADD|UPDATE|DELETE|NOOP", "target_id": "<note_id or null>"}}\n'
    "Rules: ADD if new info. UPDATE if similar note exists. "
    "DELETE if contradicted. NOOP if irrelevant.\n"
    "Content: {content}\n"
    "Existing notes: {memory}"
)

_DISTIL_PROMPT = (
    "Select the memory notes needed to answer and provide the answer. "
    "Output JSON:\n"
    '{{"selected_ids": ["id1", ...], "answer": "concise answer"}}\n'
    "Query: {query}\n"
    "Memory notes: {candidates}"
)

_SUMMARY_PROMPT = (
    "Summarize this interaction as a memory note. "
    "Output 1-2 factual sentences capturing what was learned.\n"
    "Query: {query}\n"
    "Answer: {answer}\n"
    "Reward: {reward}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_bank(db_dir: str, name: str) -> MemoryBank:
    """Create a fresh MemoryBank for a single system.

    Removes stale WAL/SHM files from previous crashed runs to avoid
    SQLite locking issues.
    """
    path = os.path.join(db_dir, f"{name}.sqlite")
    for suffix in ["", "-wal", "-shm", "-journal"]:
        full = path + suffix
        if os.path.exists(full):
            try:
                os.remove(full)
            except PermissionError:
                import time
                time.sleep(0.5)
                try:
                    os.remove(full)
                except PermissionError:
                    pass
    return MemoryBank(path)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_asem_system(config_path: str, db_dir: str) -> ASEMSystem:
    """Build the full ASEM pipeline wrapped as an eval system."""
    cfg = _load_config(config_path)
    backend = build_backend(cfg["inference"])
    hp = cfg["hyperparameters"]

    note_prompt = _load_text("data/prompts/P1_note_construction.txt")
    link_prompt = _load_text("data/prompts/P2_link_generation.txt")
    evolve_prompt = _load_text("data/prompts/P3_memory_evolution.txt")

    note_constructor = NoteConstructor(
        backend=backend, prompt_template=note_prompt, q0=hp["q0"]
    )
    memory_manager = MemoryManager(
        backend=backend, prompt_template=_MEMORY_MANAGER_PROMPT,
    )
    link_evolver = LinkEvolver(
        backend=backend,
        link_prompt_template=link_prompt,
        evolve_prompt_template=evolve_prompt,
        k=hp["k"],
    )
    retriever = HybridRetriever(
        backend=backend,
        k1=hp["k1"], k2=hp["k2"],
        delta=hp["delta"], lambda_weight=hp["lambda"],
    )
    answer_agent = AnswerAgent(
        backend=backend,
        prompt_template=_DISTIL_PROMPT,
        baseline_prompt_template=_RETRIEVAL_PROMPT,
    )
    utility_updater = UtilityUpdater(
        backend=backend,
        alpha=hp["alpha"], q0=hp["q0"],
        summary_prompt_template=_SUMMARY_PROMPT,
        note_constructor=note_constructor,
    )

    _ensure_dir(db_dir)
    bank = _make_bank(db_dir, "asem")

    pipeline = ASEMPipeline(
        memory_bank=bank,
        note_constructor=note_constructor,
        memory_manager=memory_manager,
        link_evolver=link_evolver,
        retriever=retriever,
        answer_agent=answer_agent,
        utility_updater=utility_updater,
    )

    return ASEMSystem(pipeline=pipeline)


def build_asem_v2_system(config_path: str, db_dir: str) -> ASEMSystemV2:
    """Build the two-phase ASEM v2 pipeline with batch ingestion + enhanced retrieval."""
    cfg = _load_config(config_path)
    backend = build_backend(cfg["inference"])
    hp = cfg["hyperparameters"]

    note_prompt = _load_text("data/prompts/P1_note_construction.txt")
    link_prompt = _load_text("data/prompts/P2_link_generation.txt")
    evolve_prompt = _load_text("data/prompts/P3_memory_evolution.txt")
    extract_prompt = _load_text("data/prompts/P4_batch_note_extraction.txt")
    mem_ops_prompt = _load_text("data/prompts/P5_batch_memory_ops.txt")
    batch_link_prompt = _load_text("data/prompts/P6_batch_link_generation.txt")

    from asem.batch_ingestion import BatchIngestor
    from asem.enhanced_retriever import EnhancedHybridRetriever

    note_constructor = NoteConstructor(
        backend=backend, prompt_template=note_prompt, q0=hp["q0"],
    )
    memory_manager = MemoryManager(
        backend=backend, prompt_template=_MEMORY_MANAGER_PROMPT,
    )
    link_evolver = LinkEvolver(
        backend=backend,
        link_prompt_template=link_prompt,
        evolve_prompt_template=evolve_prompt,
        k=hp["k"],
    )
    retriever = EnhancedHybridRetriever(
        backend=backend,
        k1=hp["k1"], k2=hp["k2"],
        delta=hp["delta"], lambda_weight=hp["lambda"],
        max_hops=2, hop_decay=0.7, multi_hop_topn=5,
        alpha=0.35, beta=0.25, gamma=0.40,
        enable_global_semantics=True,
        enable_intent_q=True,
    )
    answer_agent = AnswerAgent(
        backend=backend,
        prompt_template=_DISTIL_PROMPT,
        baseline_prompt_template=_RETRIEVAL_PROMPT,
    )
    utility_updater = UtilityUpdater(
        backend=backend,
        alpha=hp["alpha"], q0=hp["q0"],
        summary_prompt_template=_SUMMARY_PROMPT,
        note_constructor=note_constructor,
    )
    batch_ingestor = BatchIngestor(
        backend=backend,
        extraction_prompt=extract_prompt,
        memory_ops_prompt=mem_ops_prompt,
        link_prompt=batch_link_prompt,
        q0=hp["q0"],
        top_k_neighbors=hp.get("k", 5),
    )

    _ensure_dir(db_dir)
    bank = _make_bank(db_dir, "asem_v2")

    pipeline = ASEMPipeline(
        memory_bank=bank,
        note_constructor=note_constructor,
        memory_manager=memory_manager,
        link_evolver=link_evolver,
        retriever=retriever,
        answer_agent=answer_agent,
        utility_updater=utility_updater,
    )

    return ASEMSystemV2(pipeline=pipeline, batch_ingestor=batch_ingestor)


def build_baselines(
    config_path: str,
    db_dir: str,
    max_history_turns: int = 150,
) -> Dict[str, object]:
    """Build all six baseline systems, each with its own isolated MemoryBank.

    Args:
        max_history_turns: Truncation limit for FullContext baseline.
            0 = no truncation. Default 150 for LoCoMo.
    """
    cfg = _load_config(config_path)
    backend = build_backend(cfg["inference"])
    hp = cfg["hyperparameters"]

    note_prompt = _load_text("data/prompts/P1_note_construction.txt")
    link_prompt = _load_text("data/prompts/P2_link_generation.txt")
    evolve_prompt = _load_text("data/prompts/P3_memory_evolution.txt")

    _ensure_dir(db_dir)

    # Shared components that don't hold mutable state
    note_constructor = NoteConstructor(
        backend=backend, prompt_template=note_prompt, q0=hp["q0"]
    )
    memory_manager = MemoryManager(
        backend=backend, prompt_template=_MEMORY_MANAGER_PROMPT,
    )
    link_evolver = LinkEvolver(
        backend=backend,
        link_prompt_template=link_prompt,
        evolve_prompt_template=evolve_prompt,
        k=hp["k"],
    )
    retriever = HybridRetriever(
        backend=backend,
        k1=hp["k1"], k2=hp["k2"],
        delta=hp["delta"], lambda_weight=hp["lambda"],
    )
    answer_agent = AnswerAgent(
        backend=backend,
        prompt_template=_DISTIL_PROMPT,
        baseline_prompt_template=_RETRIEVAL_PROMPT,
    )
    utility_updater = UtilityUpdater(
        backend=backend,
        alpha=hp["alpha"], q0=hp["q0"],
        summary_prompt_template=_SUMMARY_PROMPT,
        note_constructor=note_constructor,
    )

    return {
        "NoMemory": NoMemory(
            backend=backend,
            prompt_template=_NO_MEMORY_PROMPT,
        ),
        "FullContext": FullContext(
            backend=backend,
            prompt_template=_FULL_CONTEXT_PROMPT,
            max_history_turns=max_history_turns,
        ),
        "SimRetrieval": SimRetrieval(
            backend=backend,
            memory_bank=_make_bank(db_dir, "simretrieval"),
            note_constructor=note_constructor,
            top_k=hp["k2"],
            prompt_template=_RETRIEVAL_PROMPT,
        ),
        "AtomicLinking": AtomicLinking(
            backend=backend,
            memory_bank=_make_bank(db_dir, "atomiclinking"),
            note_constructor=note_constructor,
            link_evolver=link_evolver,
            top_k=hp["k2"],
            prompt_template=_RETRIEVAL_PROMPT,
        ),
        "RLManagerOnly": RLManagerOnly(
            backend=backend,
            memory_bank=_make_bank(db_dir, "rlmanageronly"),
            note_constructor=note_constructor,
            memory_manager=memory_manager,
            top_k=hp["k2"],
            prompt_template=_RETRIEVAL_PROMPT,
        ),
        "ValueRetrievalOnly": ValueRetrievalOnly(
            backend=backend,
            memory_bank=_make_bank(db_dir, "valueretrievalonly"),
            note_constructor=note_constructor,
            retriever=retriever,
            utility_updater=utility_updater,
            answer_agent=answer_agent,
        ),
    }


def get_systems(
    config_path: str = "configs/default.yaml",
    db_dir: str = "data/benchmarks/eval_banks",
) -> Dict[str, object]:
    """Build ASEM and baseline systems for evaluation runners.

    Each system gets its own isolated MemoryBank.
    """
    systems = build_baselines(config_path, db_dir)
    systems["ASEM"] = build_asem_system(config_path, db_dir)
    systems["ASEMv2"] = build_asem_v2_system(config_path, db_dir)
    return systems
