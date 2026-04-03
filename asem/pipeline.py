"""Full ASEM pipeline integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from .answer_agent import AnswerAgent
from .link_evolver import LinkEvolver
from .memory_bank import MemoryBank
from .memory_manager import MemoryManager, Op
from .note import Note, NoteConstructor
from .retriever import HybridRetriever
from .utility_updater import UtilityUpdater
from .profiling import StageProfiler, stage_timer


@dataclass
class ASEMPipeline:
    """Pipeline wiring for all ASEM stages."""

    memory_bank: MemoryBank
    note_constructor: NoteConstructor
    memory_manager: MemoryManager
    link_evolver: LinkEvolver
    retriever: HybridRetriever
    answer_agent: AnswerAgent
    utility_updater: UtilityUpdater

    def write_path(self, content: str, timestamp: datetime) -> Optional[Note]:
        note = self.note_constructor.build(content, timestamp)
        existing = self.memory_bank.list_notes()
        op, target = self.memory_manager.select_op(content, existing)

        if op == Op.ADD:
            self.memory_bank.add(note)
            self.link_evolver.link_and_evolve(note, self.memory_bank)
            return note

        if op == Op.UPDATE:
            updated = self._merge_update(target, note)
            self.memory_bank.add(updated)
            self.link_evolver.link_and_evolve(updated, self.memory_bank)
            return updated

        if op == Op.DELETE:
            if target is not None:
                self.memory_bank.delete(target.id)
            return None

        return None

    def read_path(self, query: str) -> Tuple[List[Note], str]:
        candidates = self.retriever.retrieve(query, self.memory_bank)
        return self.answer_agent.distil_and_answer(query, candidates)

    def update_path(
        self,
        reward: float,
        used_notes: List[Note],
        query: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> None:
        self.utility_updater.update(reward, used_notes, self.memory_bank, query, answer)

    def run_turn(
        self,
        content: str,
        query: str,
        reward: float,
        timestamp: datetime,
    ) -> str:
        self.write_path(content, timestamp)
        used_notes, answer = self.read_path(query)
        self.update_path(reward, used_notes, query, answer)
        return answer

    def profile_turn(
        self,
        content: str,
        query: str,
        reward: float,
        timestamp: datetime,
    ) -> tuple[str, StageProfiler]:
        profiler = StageProfiler()
        with stage_timer(profiler, "write"):
            self.write_path(content, timestamp)
        with stage_timer(profiler, "read"):
            used_notes, answer = self.read_path(query)
        with stage_timer(profiler, "update"):
            self.update_path(reward, used_notes, query, answer)
        return answer, profiler

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
