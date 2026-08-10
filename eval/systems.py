"""System builders for evaluation runners.

Each system (baseline or ASEM) gets its own isolated MemoryBank so that
evaluations are independent and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Dict, List, Tuple

import yaml

from asem.answer_agent import AnswerAgent
from asem.backends import build_backend
from asem.link_evolver import LinkEvolver
from asem.logging_utils import get_logger
from asem.memory_bank import MemoryBank
from asem.memory_manager import MemoryManager
from asem.note import Note, NoteConstructor
from asem.pipeline import ASEMPipeline
from asem.retriever import HybridRetriever
from asem.utility_updater import UtilityUpdater
from asem.write_gate import WriteGate
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
    _logger = get_logger(__name__)

    # Track whether this conversation has been pre-ingested
    _pre_ingested: bool = False

    def ingest(self, content: str) -> None:
        """Write a single turn into the memory bank without answering.

        Used for session-aware pre-ingestion: all conversation turns for a
        session are ingested first, then queries are answered.
        """
        self._logger.debug("ASEMSystem.ingest | content={!r}", content[:120])
        try:
            self.pipeline.write_path(content, datetime.utcnow())
        except Exception as exc:
            self._logger.opt(exception=exc).error(
                "ASEMSystem.ingest | write_path failed | content={!r}", content[:80])
            raise

    def ingest_session(self, turns: List[str], session_label: str) -> List[Note]:
        """Ingest all turns from one session as a batch through S1→S2→S3.

        Args:
            turns: List of dialogue turn texts (e.g., '[Caroline] Hey! ...')
            session_label: Session identifier with date (e.g.,
                'session_1 — 1:56 pm on 8 May, 2023')

        Returns:
            List of notes created or updated during ingestion.
        """
        self._logger.info("ASEMSystem.ingest_session | session={!r} | turns={} | bank_size={}",
                         session_label, len(turns), self.bank_size)
        try:
            notes = self.pipeline.write_batch(turns, session_label, datetime.utcnow())
        except Exception as exc:
            self._logger.opt(exception=exc).error(
                "ASEMSystem.ingest_session | write_batch failed | session={!r}", session_label)
            raise
        self._pre_ingested = True
        return notes

    def ingest_conversation(
        self, session_batches: List[Tuple[str, List[str]]]
    ) -> List[Note]:
        """Ingest a conversation session-by-session, each session fully batched.

        Each session gets 3 LLM calls (S1 batch + S2 batch + S3 global).
        Cross-chunk link evolution runs once at the end.

        Args:
            session_batches: List of (session_label, turns) for all sessions.

        Returns:
            List of all notes created or updated.
        """
        total_turns = sum(len(t) for _, t in session_batches)
        self._logger.info("ASEMSystem.ingest_conversation | sessions={} | total_turns={} | bank_size={}",
                         len(session_batches), total_turns, self.bank_size)

        all_notes: List[Note] = []
        for i, (label, turns) in enumerate(session_batches):
            self._logger.info("ingest_conversation | session {}/{}: {!r} ({} turns)",
                            i + 1, len(session_batches), label, len(turns))
            notes = self.pipeline.write_batch(turns, label, datetime.utcnow())
            all_notes.extend(notes)
            self._logger.info("ingest_conversation | session {}/{} done | bank_size={}",
                            i + 1, len(session_batches), self.bank_size)

        self._pre_ingested = True
        return all_notes

    def finalize_conversation(self) -> int:
        """Run cross-chunk link evolution after all sessions are ingested.

        Returns:
            Number of new edges created across chunks.
        """
        self._logger.info("ASEMSystem.finalize_conversation | bank_size={}", self.bank_size)
        try:
            new_edges = self.pipeline.cross_chunk_link_evolve()
        except Exception as exc:
            self._logger.opt(exception=exc).error(
                "ASEMSystem.finalize_conversation | cross_chunk_link_evolve failed")
            raise
        stats = self.pipeline.get_stats()
        self._logger.info("ASEMSystem.finalize_conversation | stats={}", stats)
        return new_edges

    @property
    def bank_size(self) -> int:
        """Return the number of notes currently in the memory bank (fast)."""
        return self.pipeline.memory_bank.size()

    def answer(self, query: str, history: List[str]) -> str:
        self._logger.debug("ASEMSystem.answer | query={!r} | history_turns={} | bank_size={} | pre_ingested={}",
                          query[:120], len(history), self.bank_size, self._pre_ingested)

        # If pre-ingested, skip history replay and go straight to retrieval
        if not self._pre_ingested:
            for i, item in enumerate(history):
                try:
                    self.pipeline.write_path(item, datetime.utcnow())
                except Exception as exc:
                    self._logger.opt(exception=exc).error(
                        "ASEMSystem.answer | write_path failed at turn {} | content={!r}",
                        i, item[:80])
                    raise

        try:
            used_notes, answer = self.pipeline.read_path(query)
        except Exception as exc:
            self._logger.opt(exception=exc).error(
                "ASEMSystem.answer | read_path failed for query={!r}", query[:120])
            raise

        self._logger.info("ASEMSystem.answer → answer={!r} | used_notes={} | bank_size={}",
                         answer[:150], [n.id for n in used_notes], self.bank_size)
        return answer

    def reset(self) -> None:
        """Clear the pipeline's memory bank between conversations."""
        self._pre_ingested = False
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
    mem_manager_prompt = _load_text("data/prompts/P_memory_manager.txt")
    distil_prompt = _load_text("data/prompts/P_distil.txt")
    summary_prompt = _load_text("data/prompts/P_summary.txt")
    batch_extract_prompt = _load_text("data/prompts/P1_batch_note_construction.txt")
    batch_evolve_prompt = _load_text("data/prompts/P3_batch_evolution.txt")

    retry_cfg = cfg.get("llm_retry", {}) or {}
    max_retries = int(retry_cfg.get("max_retries", 0))

    note_constructor = NoteConstructor(
        backend=backend, prompt_template=note_prompt, q0=hp["q0"],
        max_retries=max_retries, batch_prompt_template=batch_extract_prompt,
    )
    memory_manager = MemoryManager(
        backend=backend, prompt_template=mem_manager_prompt,
        max_retries=max_retries,
    )
    wg_cfg = cfg.get("write_gate", {}) or {}
    link_evolver = LinkEvolver(
        backend=backend,
        link_prompt_template=link_prompt,
        evolve_prompt_template=evolve_prompt,
        k=hp["k"],
        link_tau=float(cfg.get("link_tau", 0.35)),
        max_retries=max_retries,
        evolve_batch_template=batch_evolve_prompt,
    )
    retriever = HybridRetriever(
        backend=backend,
        k1=hp["k1"], k2=hp["k2"],
        delta=hp["delta"], lambda_weight=hp["lambda"],
    )
    answer_agent = AnswerAgent(
        backend=backend,
        prompt_template=distil_prompt,
        baseline_prompt_template=_RETRIEVAL_PROMPT,
        max_retries=max_retries,
    )
    utility_updater = UtilityUpdater(
        backend=backend,
        alpha=hp["alpha"], q0=hp["q0"],
        summary_prompt_template=summary_prompt,
        note_constructor=note_constructor,
    )
    write_gate = WriteGate(
        enabled=bool(wg_cfg.get("enabled", True)),
        tau_high=float(wg_cfg.get("tau_high", 0.45)),
        tau_redund=float(wg_cfg.get("tau_redund", 0.92)),
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
        write_gate=write_gate,
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
    mem_manager_prompt = _load_text("data/prompts/P_memory_manager.txt")
    distil_prompt = _load_text("data/prompts/P_distil.txt")
    summary_prompt = _load_text("data/prompts/P_summary.txt")
    batch_extract_prompt = _load_text("data/prompts/P1_batch_note_construction.txt")
    batch_evolve_prompt = _load_text("data/prompts/P3_batch_evolution.txt")

    retry_cfg = cfg.get("llm_retry", {}) or {}
    max_retries = int(retry_cfg.get("max_retries", 0))

    from asem.batch_ingestion import BatchIngestor
    from asem.enhanced_retriever import EnhancedHybridRetriever

    note_constructor = NoteConstructor(
        backend=backend, prompt_template=note_prompt, q0=hp["q0"],
        max_retries=max_retries, batch_prompt_template=batch_extract_prompt,
    )
    memory_manager = MemoryManager(
        backend=backend, prompt_template=mem_manager_prompt,
        max_retries=max_retries,
    )
    link_evolver = LinkEvolver(
        backend=backend,
        link_prompt_template=link_prompt,
        evolve_prompt_template=evolve_prompt,
        k=hp["k"],
        link_tau=float(cfg.get("link_tau", 0.35)),
        max_retries=max_retries,
        evolve_batch_template=batch_evolve_prompt,
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
        prompt_template=distil_prompt,
        baseline_prompt_template=_RETRIEVAL_PROMPT,
        max_retries=max_retries,
    )
    utility_updater = UtilityUpdater(
        backend=backend,
        alpha=hp["alpha"], q0=hp["q0"],
        summary_prompt_template=summary_prompt,
        note_constructor=note_constructor,
    )
    batch_ingestor = BatchIngestor(
        backend=backend,
        extraction_prompt=extract_prompt,
        memory_ops_prompt=mem_ops_prompt,
        link_prompt=batch_link_prompt,
        q0=hp["q0"],
        top_k_neighbors=hp.get("k", 5),
        max_retries=max_retries,
    )
    wg_cfg = cfg.get("write_gate", {}) or {}
    write_gate = WriteGate(
        enabled=bool(wg_cfg.get("enabled", True)),
        tau_high=float(wg_cfg.get("tau_high", 0.45)),
        tau_redund=float(wg_cfg.get("tau_redund", 0.92)),
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
        write_gate=write_gate,
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
    mem_manager_prompt = _load_text("data/prompts/P_memory_manager.txt")
    distil_prompt = _load_text("data/prompts/P_distil.txt")
    summary_prompt = _load_text("data/prompts/P_summary.txt")
    batch_extract_prompt = _load_text("data/prompts/P1_batch_note_construction.txt")
    batch_evolve_prompt = _load_text("data/prompts/P3_batch_evolution.txt")

    retry_cfg = cfg.get("llm_retry", {}) or {}
    max_retries = int(retry_cfg.get("max_retries", 0))

    _ensure_dir(db_dir)

    # Shared components that don't hold mutable state
    note_constructor = NoteConstructor(
        backend=backend, prompt_template=note_prompt, q0=hp["q0"],
        max_retries=max_retries, batch_prompt_template=batch_extract_prompt,
    )
    memory_manager = MemoryManager(
        backend=backend, prompt_template=mem_manager_prompt,
        max_retries=max_retries,
    )
    link_evolver = LinkEvolver(
        backend=backend,
        link_prompt_template=link_prompt,
        evolve_prompt_template=evolve_prompt,
        k=hp["k"],
        max_retries=max_retries,
        evolve_batch_template=batch_evolve_prompt,
    )
    retriever = HybridRetriever(
        backend=backend,
        k1=hp["k1"], k2=hp["k2"],
        delta=hp["delta"], lambda_weight=hp["lambda"],
    )
    answer_agent = AnswerAgent(
        backend=backend,
        prompt_template=distil_prompt,
        baseline_prompt_template=_RETRIEVAL_PROMPT,
        max_retries=max_retries,
    )
    utility_updater = UtilityUpdater(
        backend=backend,
        alpha=hp["alpha"], q0=hp["q0"],
        summary_prompt_template=summary_prompt,
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
