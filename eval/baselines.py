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
from asem.memory_bank import MemoryBank
from asem.memory_manager import MemoryManager, Op
from asem.note import Note, NoteConstructor
from asem.retriever import HybridRetriever
from asem.utility_updater import UtilityUpdater


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


@dataclass
class NoMemory(Baseline):
    """Backbone-only baseline — ignores all history."""

    backend: InferenceBackend
    prompt_template: str

    def answer(self, query: str, history: List[str]) -> str:
        prompt = self.prompt_template.format(query=query)
        return self.backend.generate(prompt)


@dataclass
class FullContext(Baseline):
    """All history concatenated into the context window."""

    backend: InferenceBackend
    prompt_template: str
    max_history_turns: int = 0

    def answer(self, query: str, history: List[str]) -> str:
        h = list(history)
        if self.max_history_turns > 0 and len(h) > self.max_history_turns:
            keep_first = min(5, self.max_history_turns // 4)
            keep_last = self.max_history_turns - keep_first
            h = h[:keep_first] + h[-keep_last:]
        context = "\n".join(h) if h else "(no prior conversation)"
        prompt = self.prompt_template.format(query=query, context=context)
        return self.backend.generate(prompt)


@dataclass
class SimRetrieval(Baseline):
    """Flat ANN retrieval — writes all history as atomic notes, then retrieves.

    Deduplicates against already-processed content so that repeated calls with
    cumulative histories within the same conversation don't create duplicates.
    """

    backend: InferenceBackend
    memory_bank: MemoryBank
    note_constructor: NoteConstructor
    top_k: int
    prompt_template: str

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
        context = "\n".join([n.c for n in notes]) if notes else "(no relevant memory)"
        prompt = self.prompt_template.format(query=query, context=context)
        return self.backend.generate(prompt)

    def reset(self) -> None:
        self._seen_hashes.clear()
        self.memory_bank.clear()


@dataclass
class AtomicLinking(Baseline):
    """Notes + bidirectional linking — writes all history with Stage 1 + Stage 3."""

    backend: InferenceBackend
    memory_bank: MemoryBank
    note_constructor: NoteConstructor
    link_evolver: LinkEvolver
    top_k: int
    prompt_template: str

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
        context = "\n".join([n.c for n in notes]) if notes else "(no relevant memory)"
        prompt = self.prompt_template.format(query=query, context=context)
        return self.backend.generate(prompt)

    def reset(self) -> None:
        self._seen_hashes.clear()
        self.memory_bank.clear()


@dataclass
class RLManagerOnly(Baseline):
    """RL write ops + similarity retrieval — all history through Memory Manager."""

    backend: InferenceBackend
    memory_bank: MemoryBank
    note_constructor: NoteConstructor
    memory_manager: MemoryManager
    top_k: int
    prompt_template: str

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
        context = "\n".join([n.c for n in notes]) if notes else "(no relevant memory)"
        prompt = self.prompt_template.format(query=query, context=context)
        return self.backend.generate(prompt)

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
