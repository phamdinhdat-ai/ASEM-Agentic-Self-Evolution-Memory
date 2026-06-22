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

_logger = get_logger(__name__)


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
        _logger.debug("write_path START | content={!r}", content[:120])

        note = self.note_constructor.build(content, timestamp)

        # B4 — only pass top-k similar notes to Memory Manager, not all notes.
        # This caps the S2 prompt at ~2000 tokens regardless of bank size.
        e_new = self.note_constructor.backend.embed(content)
        existing = self.memory_bank.ann_search(e_new, k=self.retriever.k2)
        if not existing:
            existing = self.memory_bank.list_notes()[: self.retriever.k2]

        op, target = self.memory_manager.select_op(content, existing)
        _logger.info("write_path | note_id={} op={} target_id={} | bank_size={}",
                     note.id, op.value,
                     (target.id if target else "-"),
                     len(self.memory_bank.list_notes()))

        if op == Op.ADD:
            self.memory_bank.add(note)
            self.link_evolver.link_and_evolve(note, self.memory_bank)
            _logger.success("ADD note {} | K={} G={}", note.id, note.K[:3], note.G[:3])
            return note

        if op == Op.UPDATE:
            updated = self._merge_update(target, note)
            self.memory_bank.add(updated)
            self.link_evolver.link_and_evolve(updated, self.memory_bank)
            _logger.success("UPDATE note {} | merged into {}", note.id, updated.id)
            return updated

        if op == Op.DELETE:
            if target is not None:
                self.memory_bank.delete(target.id)
                _logger.info("DELETE note {}", target.id)
            return None

        _logger.debug("write_path NOOP")
        return None

    def read_path(self, query: str) -> Tuple[List[Note], str]:
        _logger.debug("read_path START | query={!r}", query[:120])

        candidates = self.retriever.retrieve(query, self.memory_bank)
        _logger.info("read_path | candidates={} | bank_size={}",
                     len(candidates), len(self.memory_bank.list_notes()))

        if not candidates:
            _logger.warning("read_path | no candidates retrieved for query={!r}", query[:80])

        used_notes, answer = self.answer_agent.distil_and_answer(query, candidates)
        _logger.info("read_path | selected_notes={} | answer={!r}",
                     [n.id for n in used_notes], answer[:150])
        return used_notes, answer

    def update_path(
        self,
        reward: float,
        used_notes: List[Note],
        query: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> None:
        _logger.debug("update_path START | reward={:.3f} | used_notes={}",
                     reward, [n.id for n in used_notes])

        before_qs = {n.id: n.q for n in used_notes}
        self.utility_updater.update(reward, used_notes, self.memory_bank, query, answer)

        for note in used_notes:
            new_q = self.memory_bank.get(note.id)
            if new_q:
                old_q = before_qs.get(note.id, 0.0)
                _logger.info("update_path | q({}): {:.3f} → {:.3f} (reward={:.3f})",
                            note.id, old_q, new_q.q, reward)

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
