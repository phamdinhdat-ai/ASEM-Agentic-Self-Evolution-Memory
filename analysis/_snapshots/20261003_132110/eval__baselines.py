"""Baseline implementations for evaluation.

Each baseline implements the common interface:

    def answer(self, query: str, history: List[str]) -> str
    def reset(self) -> None

History items are processed incrementally: the first call to answer() may see
a partial history, and subsequent calls within the same conversation see
cumulative histories.  Each baseline deduplicates against already-processed
content so that notes are never stored twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set, Tuple

from asem.answer_agent import AnswerAgent
from asem.backends.base import InferenceBackend
from asem.link_evolver import LinkEvolver
from asem.logging_utils import get_logger
from asem.memory_bank import MemoryBank
from asem.memory_manager import MemoryManager, Op
from asem.note import Note, NoteConstructor
from asem.retriever import HybridRetriever
from asem.token_budget import (
    clip_block,
    default_output_cap,
    fit_items,
    generate_with_cap,
    resolve_budget,
)
from asem.utility_updater import UtilityUpdater

_log = get_logger("eval.baselines")


@dataclass
class Baseline:
    """Common baseline interface."""

    def answer(self, query: str, history: List[str]) -> str:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset per-conversation state (called between conversations)."""
        pass

    def ingest_batch(self, turns: List[str], session_label: str) -> None:
        """Ingest a batch of turns from one session.

        Default: no-op. Memory-based baselines override this for efficient
        pre-ingestion before answering questions.
        """
        pass

    def ingest_conversation(
        self, session_batches: List[Tuple[str, List[str]]]
    ) -> None:
        """Ingest ALL sessions of a conversation in one pass.

        Default: falls back to calling ingest_batch() per session.
        """
        for label, turns in session_batches:
            self.ingest_batch(turns, label)

    def finalize_conversation(self) -> int:
        """Run post-ingestion cross-chunk link evolution.

        Default: no-op. ASEM overrides this.
        """
        return 0


class _BudgetedAnswer:
    """Token-budget plumbing shared by the direct-prompt baselines.

    Every system in a comparison answers with ONE model call, so the context it
    assembles — the whole history for ``FullContext``, the top-k notes for the
    retrieval baselines — must fit the SAME budget the ASEM answer agent uses
    (``asem.token_budget``)::

        prompt <= context_window - max_tokens - safety

    The mixin deliberately declares no dataclass fields (each baseline owns its
    ``max_tokens`` / ``context_window``); it only provides the shared behaviour.
    """

    backend: InferenceBackend
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None

    def answer_budget(self) -> Optional[int]:
        """Tokens this system may spend on the prompt (None = no window set)."""
        return resolve_budget(self.backend, self.context_window, self.max_tokens)

    def fit_context(
        self,
        blocks: List[str],
        render,
        *,
        drop: str = "tail",
        head_keep: int = 0,
        label: str = "context",
    ) -> Tuple[str, List[str]]:
        """Shrink the context blocks until the rendered prompt fits the budget.

        ``drop="tail"`` is for relevance-ordered blocks (retrieved notes),
        ``drop="oldest"`` for chronological history — the middle is dropped and
        the opening plus the most recent blocks survive.
        """
        return fit_items(
            blocks,
            render,
            budget=self.answer_budget(),
            min_keep=1,
            drop=drop,
            head_keep=head_keep,
            shrink=clip_block,   # a single oversized turn must not blow the window
            label=label,
        )

    def _generate(self, prompt: str, *, label: str) -> str:
        """The answer call, with the same per-call completion cap as ASEM.

        Reserving ``max_tokens`` while letting the request send a larger
        client-wide default would still overflow the window, so the cap goes out
        with the request; without ``answer.max_tokens`` the backend's own
        default applies (and was already reserved by ``answer_budget``).
        """
        cap = self.max_tokens or default_output_cap(self.backend)
        _log.debug(
            "{}: answer call with max_tokens={} (prompt budget {})",
            label, cap, self.answer_budget(),
        )
        return generate_with_cap(self.backend, prompt, cap)


@dataclass
class NoMemory(_BudgetedAnswer, Baseline):
    """Backbone-only baseline — ignores all history."""

    backend: InferenceBackend
    prompt_template: str
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None

    def answer(self, query: str, history: List[str]) -> str:
        prompt = self.prompt_template.format(query=query)
        return self._generate(prompt, label="NoMemory")


@dataclass
class FullContext(_BudgetedAnswer, Baseline):
    """All history concatenated into the context window.

    The history is trimmed to the answer call's token budget
    (``context_window - max_tokens - safety``), keeping the opening ``head_turns``
    turns and as many of the most RECENT ones as fit — the middle is what goes.
    ``max_history_turns`` remains a cheap turn-count pre-filter (0 = off); it is
    no longer the thing that decides how much context the model sees.
    """

    backend: InferenceBackend
    prompt_template: str
    max_history_turns: int = 0
    # Turns protected at the START of the history: LoCoMo conversations open with
    # the setup (who is who, where they live), which late questions still need.
    head_turns: int = 5
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None

    def answer(self, query: str, history: List[str]) -> str:
        h = list(history)
        if self.max_history_turns > 0 and len(h) > self.max_history_turns:
            keep_first = min(self.head_turns, self.max_history_turns // 4)
            keep_last = self.max_history_turns - keep_first
            h = h[:keep_first] + h[-keep_last:]

        prompt, _ = self.fit_context(
            h,
            lambda kept: self.prompt_template.format(
                query=query,
                context="\n".join(kept) if kept else "(no prior conversation)",
            ),
            drop="oldest",
            head_keep=self.head_turns,
            label="FullContext",
        )
        return self._generate(prompt, label="FullContext")


@dataclass
class SimRetrieval(_BudgetedAnswer, Baseline):
    """Flat ANN retrieval — writes all history as atomic notes, then retrieves.

    Deduplicates against already-processed content so that repeated calls with
    cumulative histories within the same conversation don't create duplicates.
    The retrieved context is trimmed to the answer call's token budget, dropping
    the least similar notes first.
    """

    backend: InferenceBackend
    memory_bank: MemoryBank
    note_constructor: NoteConstructor
    top_k: int
    prompt_template: str
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None

    # ---- private, not constructor args -----------------------------------
    _seen_hashes: Set[int] = field(default_factory=set, init=False, repr=False)

    def ingest_batch(self, turns: List[str], session_label: str) -> None:
        """Pre-ingest all turns from one session as a batch."""
        enriched = [f"[{session_label}] {t}" for t in turns]
        new = [t for t in enriched if hash(t) not in self._seen_hashes]
        if not new:
            return
        for t in new:
            self._seen_hashes.add(hash(t))
        notes = self.note_constructor.build_batch(new, datetime.utcnow())
        for note in notes:
            self.memory_bank.add(note)

    def answer(self, query: str, history: List[str]) -> str:
        # If not pre-ingested, fall back to per-question history processing
        if not self._seen_hashes:
            for item in history:
                h = hash(item)
                if h not in self._seen_hashes:
                    self._seen_hashes.add(h)
                    note = self.note_constructor.build(item, datetime.utcnow())
                    self.memory_bank.add(note)

        e_q = self.backend.embed(query)
        notes = self.memory_bank.ann_search(e_q, k=self.top_k)
        prompt, _ = self.fit_context(
            [n.c for n in notes],
            lambda kept: self.prompt_template.format(
                query=query,
                context="\n".join(kept) if kept else "(no relevant memory)",
            ),
            label="SimRetrieval",
        )
        return self._generate(prompt, label="SimRetrieval")

    def reset(self) -> None:
        self._seen_hashes.clear()
        self.memory_bank.clear()


@dataclass
class AtomicLinking(_BudgetedAnswer, Baseline):
    """Notes + bidirectional linking — writes all history with Stage 1 + Stage 3."""

    backend: InferenceBackend
    memory_bank: MemoryBank
    note_constructor: NoteConstructor
    link_evolver: LinkEvolver
    top_k: int
    prompt_template: str
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None

    _seen_hashes: Set[int] = field(default_factory=set, init=False, repr=False)

    def ingest_batch(self, turns: List[str], session_label: str) -> None:
        """Pre-ingest all turns from one session with linking."""
        enriched = [f"[{session_label}] {t}" for t in turns]
        new = [t for t in enriched if hash(t) not in self._seen_hashes]
        if not new:
            return
        for t in new:
            self._seen_hashes.add(hash(t))
        notes = self.note_constructor.build_batch(new, datetime.utcnow())
        for note in notes:
            self.memory_bank.add(note)
            self.link_evolver.link_and_evolve(note, self.memory_bank)

    def answer(self, query: str, history: List[str]) -> str:
        if not self._seen_hashes:
            for item in history:
                h = hash(item)
                if h not in self._seen_hashes:
                    self._seen_hashes.add(h)
                    note = self.note_constructor.build(item, datetime.utcnow())
                    self.memory_bank.add(note)
                    self.link_evolver.link_and_evolve(note, self.memory_bank)

        e_q = self.backend.embed(query)
        notes = self.memory_bank.ann_search(e_q, k=self.top_k)
        prompt, _ = self.fit_context(
            [n.c for n in notes],
            lambda kept: self.prompt_template.format(
                query=query,
                context="\n".join(kept) if kept else "(no relevant memory)",
            ),
            label="AtomicLinking",
        )
        return self._generate(prompt, label="AtomicLinking")

    def reset(self) -> None:
        self._seen_hashes.clear()
        self.memory_bank.clear()


@dataclass
class RLManagerOnly(_BudgetedAnswer, Baseline):
    """RL write ops + similarity retrieval — all history through Memory Manager."""

    backend: InferenceBackend
    memory_bank: MemoryBank
    note_constructor: NoteConstructor
    memory_manager: MemoryManager
    top_k: int
    prompt_template: str
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None

    _seen_hashes: Set[int] = field(default_factory=set, init=False, repr=False)

    def ingest_batch(self, turns: List[str], session_label: str) -> None:
        """Pre-ingest all turns from one session with Memory Manager decisions."""
        enriched = [f"[{session_label}] {t}" for t in turns]
        new_turns = [(t, e) for t, e in zip(turns, enriched)
                     if hash(e) not in self._seen_hashes]
        if not new_turns:
            return
        new_contents = [e for _, e in new_turns]
        for e in new_contents:
            self._seen_hashes.add(hash(e))
        notes = self.note_constructor.build_batch(new_contents, datetime.utcnow())
        for (raw_turn, enriched), note in zip(new_turns, notes):
            existing = self.memory_bank.list_notes()
            top_k2 = min(len(existing), 5)
            candidates = existing[:top_k2] if top_k2 > 0 else existing
            op, target = self.memory_manager.select_op(enriched, candidates)
            if op == Op.ADD:
                self.memory_bank.add(note)
            elif op == Op.UPDATE:
                updated = self._merge_update(target, note)
                self.memory_bank.add(updated)
            elif op == Op.DELETE and target is not None:
                self.memory_bank.delete(target.id)

    def answer(self, query: str, history: List[str]) -> str:
        if not self._seen_hashes:
            for item in history:
                h = hash(item)
                if h not in self._seen_hashes:
                    self._seen_hashes.add(h)
                    note = self.note_constructor.build(item, datetime.utcnow())
                    existing = self.memory_bank.list_notes()
                    top_k2 = min(len(existing), 5)
                    candidates = existing[:top_k2] if top_k2 > 0 else existing
                    op, target = self.memory_manager.select_op(item, candidates)
                    if op == Op.ADD:
                        self.memory_bank.add(note)
                    elif op == Op.UPDATE:
                        updated = self._merge_update(target, note)
                        self.memory_bank.add(updated)
                    elif op == Op.DELETE and target is not None:
                        self.memory_bank.delete(target.id)

        e_q = self.backend.embed(query)
        notes = self.memory_bank.ann_search(e_q, k=self.top_k)
        prompt, _ = self.fit_context(
            [n.c for n in notes],
            lambda kept: self.prompt_template.format(
                query=query,
                context="\n".join(kept) if kept else "(no relevant memory)",
            ),
            label="RLManagerOnly",
        )
        return self._generate(prompt, label="RLManagerOnly")

    def reset(self) -> None:
        self._seen_hashes.clear()
        self.memory_bank.clear()

    @staticmethod
    def _merge_update(target: Optional[Note], note: Note) -> Note:
        if target is None:
            return note
        return Note(
            id=target.id,
            c=note.c,
            t=note.t,
            K=note.K,
            G=note.G,
            X=note.X,
            e=note.e,
            L=target.L,
            z=note.z,
            q=target.q,
        )


@dataclass
class ValueRetrievalOnly(Baseline):
    """Value-aware retrieval + utility updates — writes all history, updates Q-values."""

    backend: InferenceBackend
    memory_bank: MemoryBank
    note_constructor: NoteConstructor
    retriever: HybridRetriever
    utility_updater: UtilityUpdater
    answer_agent: AnswerAgent

    _seen_hashes: Set[int] = field(default_factory=set, init=False, repr=False)

    def ingest_batch(self, turns: List[str], session_label: str) -> None:
        """Pre-ingest all turns from one session."""
        enriched = [f"[{session_label}] {t}" for t in turns]
        new = [t for t in enriched if hash(t) not in self._seen_hashes]
        if not new:
            return
        for t in new:
            self._seen_hashes.add(hash(t))
        notes = self.note_constructor.build_batch(new, datetime.utcnow())
        for note in notes:
            self.memory_bank.add(note)

    def answer(self, query: str, history: List[str]) -> str:
        if not self._seen_hashes:
            for item in history:
                h = hash(item)
                if h not in self._seen_hashes:
                    self._seen_hashes.add(h)
                    note = self.note_constructor.build(item, datetime.utcnow())
                    self.memory_bank.add(note)

        used_notes, answer = self.answer_agent.distil_and_answer(
            query,
            self.retriever.retrieve(query, self.memory_bank),
        )
        self.utility_updater.update(
            reward=1.0,
            used_notes=used_notes,
            memory_bank=self.memory_bank,
        )
        return answer

    def reset(self) -> None:
        self._seen_hashes.clear()
        self.memory_bank.clear()
