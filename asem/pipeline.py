"""Full ASEM pipeline integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from .answer_agent import AnswerAgent
from .link_evolver import LinkEvolver
from .logging_utils import get_logger
from .memory_bank import MemoryBank
from .memory_manager import MemoryManager, Op
from .note import Note, NoteConstructor
from .retriever import HybridRetriever
from .utility_updater import UtilityUpdater
from .profiling import StageProfiler, stage_timer

_log = get_logger("pipeline")


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
        _log.info("WRITE path | content={!r}", content[:80])
        note = self.note_constructor.build(content, timestamp)
        _log.debug("S1 note | id={}  K={}  G={}", note.id, note.K, note.G)

        # B4 — only pass top-k similar notes to Memory Manager, not all notes.
        # This caps the S2 prompt at ~2000 tokens regardless of bank size.
        e_new = self.note_constructor.backend.embed(content)
        existing = self.memory_bank.ann_search(e_new, k=self.retriever.k2)
        if not existing:
            existing = self.memory_bank.list_notes()[: self.retriever.k2]

        op, target = self.memory_manager.select_op(content, existing)
        _log.info("S2 manager | op={}  target_id={}  candidates={}  bank_size={}",
                  op.value, target.id if target else None, len(existing), self.memory_bank.size())

        if op == Op.ADD:
            self.memory_bank.add(note)
            self.link_evolver.link_and_evolve(note, self.memory_bank)
            _log.success("WRITE done | ADD  id={}  bank_size={}", note.id, self.memory_bank.size())
            return note

        if op == Op.UPDATE:
            updated = self._merge_update(target, note)
            self.memory_bank.add(updated)
            self.link_evolver.link_and_evolve(updated, self.memory_bank)
            _log.success("WRITE done | UPDATE  id={}  bank_size={}", updated.id, self.memory_bank.size())
            return updated

        if op == Op.DELETE:
            if target is not None:
                self.memory_bank.delete(target.id)
            _log.success("WRITE done | DELETE  id={}  bank_size={}", target.id if target else "N/A", self.memory_bank.size())
            return None

        _log.info("WRITE done | NOOP  bank_size={}", self.memory_bank.size())
        return None

    def read_path(self, query: str) -> Tuple[List[Note], str]:
        _log.info("READ path | query={!r}", query[:80])
        candidates = self.retriever.retrieve(query, self.memory_bank)
        _log.debug("S4 retriever | candidates={}  stats={}", len(candidates), self.retriever.stats)
        notes, answer = self.answer_agent.distil_and_answer(query, candidates)
        _log.success("READ done | distilled={}  answer={!r}", len(notes), answer[:80])
        return notes, answer

    def update_path(
        self,
        reward: float,
        used_notes: List[Note],
        query: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> None:
        _log.info("UPDATE path | reward={:.3f}  used_notes={}", reward, len(used_notes))
        self.utility_updater.update(reward, used_notes, self.memory_bank, query, answer)
        _log.success("UPDATE done | reward={:.3f}", reward)

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
