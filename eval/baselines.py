"""Baseline implementations for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

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


@dataclass
class NoMemory(Baseline):
    """Backbone-only baseline with no history."""
    backend: InferenceBackend
    prompt_template: str

    def answer(self, query: str, history: List[str]) -> str:
        prompt = self.prompt_template.format(query=query)
        return self.backend.generate(prompt)


@dataclass
class FullContext(Baseline):
    """Baseline that concatenates all history into context."""
    backend: InferenceBackend
    prompt_template: str

    def answer(self, query: str, history: List[str]) -> str:
        context = "\n".join(history)
        prompt = self.prompt_template.format(query=query, context=context)
        return self.backend.generate(prompt)


@dataclass
class SimRetrieval(Baseline):
    """Baseline with flat ANN retrieval and generation."""
    backend: InferenceBackend
    memory_bank: MemoryBank
    top_k: int
    prompt_template: str

    def answer(self, query: str, history: List[str]) -> str:
        e_q = self.backend.embed(query)
        notes = self.memory_bank.ann_search(e_q, k=self.top_k)
        context = "\n".join([note.c for note in notes])
        prompt = self.prompt_template.format(query=query, context=context)
        return self.backend.generate(prompt)


@dataclass
class AtomicLinking(Baseline):
    """Baseline with notes + linking, no RL or Q-values."""
    backend: InferenceBackend
    memory_bank: MemoryBank
    note_constructor: NoteConstructor
    link_evolver: LinkEvolver
    top_k: int
    prompt_template: str

    def answer(self, query: str, history: List[str]) -> str:
        if history:
            note = self.note_constructor.build(history[-1], datetime.utcnow())
            self.memory_bank.add(note)
            self.link_evolver.link_and_evolve(note, self.memory_bank)

        e_q = self.backend.embed(query)
        notes = self.memory_bank.ann_search(e_q, k=self.top_k)
        context = "\n".join([note.c for note in notes])
        prompt = self.prompt_template.format(query=query, context=context)
        return self.backend.generate(prompt)


@dataclass
class RLManagerOnly(Baseline):
    """Baseline with RL write ops and similarity retrieval."""
    backend: InferenceBackend
    memory_bank: MemoryBank
    note_constructor: NoteConstructor
    memory_manager: MemoryManager
    top_k: int
    prompt_template: str

    def answer(self, query: str, history: List[str]) -> str:
        if history:
            note = self.note_constructor.build(history[-1], datetime.utcnow())
            existing = self.memory_bank.list_notes()
            op, target = self.memory_manager.select_op(history[-1], existing)
            if op == Op.ADD:
                self.memory_bank.add(note)
            elif op == Op.UPDATE:
                updated = self._merge_update(target, note)
                self.memory_bank.add(updated)
            elif op == Op.DELETE and target is not None:
                self.memory_bank.delete(target.id)

        e_q = self.backend.embed(query)
        notes = self.memory_bank.ann_search(e_q, k=self.top_k)
        context = "\n".join([note.c for note in notes])
        prompt = self.prompt_template.format(query=query, context=context)
        return self.backend.generate(prompt)

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
    """Baseline with value-aware retrieval and utility updates."""
    backend: InferenceBackend
    memory_bank: MemoryBank
    note_constructor: NoteConstructor
    retriever: HybridRetriever
    utility_updater: UtilityUpdater
    answer_agent: AnswerAgent

    def answer(self, query: str, history: List[str]) -> str:
        if history:
            note = self.note_constructor.build(history[-1], datetime.utcnow())
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
