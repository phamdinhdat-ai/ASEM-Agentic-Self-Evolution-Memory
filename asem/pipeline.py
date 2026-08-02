"""Full ASEM pipeline integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .answer_agent import AnswerAgent
from .link_evolver import LinkEvolver
from .logging_utils import get_logger
from .memory_bank import MemoryBank
from .memory_manager import MemoryManager, Op
from .note import Note, NoteConstructor
from .retriever import HybridRetriever
from .utility_updater import UtilityUpdater
from .write_gate import WriteGate
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
    write_gate: WriteGate = field(default_factory=WriteGate)

    def write_path(self, content: str, timestamp: datetime) -> Optional[Note]:
        _log.info("WRITE path | content={!r}", content[:80])
        # Lazy embedding: z (raw content) is always computed for the gate and
        # similarity search; the content+K+G+X embedding is only computed for
        # notes that are actually written (embed budget).
        note = self.note_constructor.build(content, timestamp, embed_e=False)
        _log.debug("S1 note | id={}  K={}  G={}", note.id, note.K, note.G)

        # B4 — only pass top-k similar notes to Memory Manager, not all notes.
        # This caps the S2 prompt at ~2000 tokens regardless of bank size.
        existing = self.memory_bank.ann_search(note.z, k=self.retriever.k2)
        if not existing:
            existing = self.memory_bank.list_notes()[: self.retriever.k2]

        # NGMC Tier 0 — deterministic write gate. Only the ambiguous band pays
        # the S2 LLM cost; clearly-novel turns -> ADD, near-duplicates -> NOOP.
        gate_op, _max_sim = self.write_gate.propose(note, existing)
        if gate_op is not None:
            op, target = gate_op, None
            _log.debug("S2 gate | op={} (LLM skipped)", op.value)
        else:
            op, target = self.memory_manager.select_op(content, existing)
            self.write_gate.record_ambiguous_llm(op)
        _log.info("S2 manager | op={}  target_id={}  candidates={}  bank_size={}",
                  op.value, target.id if target else None, len(existing), self.memory_bank.size())

        if op == Op.ADD:
            self.note_constructor.complete_embedding(note)
            self.memory_bank.add(note)
            self.link_evolver.link_and_evolve(note, self.memory_bank)
            _log.success("WRITE done | ADD  id={}  bank_size={}", note.id, self.memory_bank.size())
            return note

        if op == Op.UPDATE:
            self.note_constructor.complete_embedding(note)
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

    def write_batch(
        self, contents: List[str], label: str, timestamp: datetime
    ) -> List[Note]:
        """Ingest many turns through S1→S2→S3 with one batched S1 LLM call.

        Note construction runs in a single batched call (fewer LLM round-
        trips), then the Memory Manager and Link Evolver run per note.
        Returns the notes that were written (ADD/UPDATE).
        """
        _log.info("WRITE batch | label={!r} turns={} timestamp={}",
                  label, len(contents), timestamp)
        notes = self.note_constructor.build_batch(contents, timestamp, embed_e=False)
        written: List[Note] = []
        for note in notes:
            existing = self.memory_bank.ann_search(note.z, k=self.retriever.k2)
            if not existing:
                existing = self.memory_bank.list_notes()[: self.retriever.k2]
            # NGMC Tier 0 — deterministic write gate (see write_path).
            gate_op, _max_sim = self.write_gate.propose(note, existing)
            if gate_op is not None:
                op, target = gate_op, None
            else:
                op, target = self.memory_manager.select_op(note.c, existing)
                self.write_gate.record_ambiguous_llm(op)
            if op == Op.ADD:
                self.note_constructor.complete_embedding(note)
                self.memory_bank.add(note)
                self.link_evolver.link_and_evolve(note, self.memory_bank)
                written.append(note)
            elif op == Op.UPDATE:
                self.note_constructor.complete_embedding(note)
                merged = self._merge_update(target, note)
                self.memory_bank.add(merged)
                self.link_evolver.link_and_evolve(merged, self.memory_bank)
                written.append(merged)
            elif op == Op.DELETE:
                if target is not None:
                    self.memory_bank.delete(target.id)
        _log.success("WRITE batch done | label={!r} notes={} written={} bank_size={}",
                     label, len(notes), len(written), self.memory_bank.size())
        return written

    def cross_chunk_link_evolve(self) -> int:
        """Run link evolution across chunk boundaries (post-ingestion pass).

        Links notes that still have no links so knowledge crosses
        session/chunk boundaries. Returns the number of new edges created.
        """
        _log.info("cross_chunk_link_evolve | bank_size={}", self.memory_bank.size())
        notes = self.memory_bank.list_notes()
        if len(notes) < 2:
            return 0
        count = 0
        for note in notes:
            if note.L:
                continue
            before = set(note.L)
            try:
                self.link_evolver.link_and_evolve(note, self.memory_bank)
            except Exception:  # linking is best-effort; never block ingestion
                continue
            after = set(note.L)
            count += len(after - before)
        _log.success("cross_chunk_link_evolve done | new_edges={}", count)
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Return a small snapshot of pipeline/bank state for logging."""
        return {
            "bank_size": self.memory_bank.size(),
            "retriever_stats": getattr(self.retriever, "stats", {}),
        }

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
