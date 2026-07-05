"""Full ASEM pipeline integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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

    # Batch ingestion stats (populated during write_batch)
    _batch_stats: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

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
        _logger.info("read_path | candidates={} | bank_size={} | context= {}", 
                     len(candidates), len(self.memory_bank.list_notes()), candidates)

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
            new_q = self.memory_bank.get_note(note.id)
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

    # ------------------------------------------------------------------
    # Batch ingestion (S1→S2→S3 for a whole session at once)
    # ------------------------------------------------------------------

    def write_batch(
        self,
        contents: List[str],
        session_label: str,
        timestamp: datetime,
    ) -> List[Note]:
        """Process a batch of dialogue turns through S1→S2→S3 with batched LLM.

        S1: build_batch() — 1 LLM call for all turns
        S2: select_ops_batch() — 1 LLM call for all decisions
        S3: link_evolve_all() — 1 global pass at the end
        """
        _logger.info("write_batch START | session={!r} | turns={} | bank_size={}",
                     session_label, len(contents),
                     len(self.memory_bank.list_notes()))

        # Phase 1: Batched Note Construction — 1 LLM call
        enriched = [f"[{session_label}] {c}" for c in contents]
        notes = self.note_constructor.build_batch(enriched, timestamp)
        _logger.info("write_batch | S1: {} notes built", len(notes))

        # Phase 2: Batched Memory Manager — 1 LLM call
        existing = self.memory_bank.list_notes()
        decisions = self.memory_manager.select_ops_batch(enriched, existing)
        _logger.info("write_batch | S2: {} decisions", len(decisions))

        # Phase 3: Apply decisions (local, no LLM)
        stats = {"added": 0, "updated": 0, "deleted": 0, "noop": 0}
        results: List[Note] = []

        for note, (op, target_id) in zip(notes, decisions):
            target = self.memory_bank.get_note(target_id) if target_id else None

            if op == Op.ADD:
                self.memory_bank.add(note)
                stats["added"] += 1
                results.append(note)
            elif op == Op.UPDATE and target is not None:
                updated = self._merge_update(target, note)
                self.memory_bank.add(updated)
                stats["updated"] += 1
                results.append(updated)
            elif op == Op.DELETE:
                if target is not None:
                    self.memory_bank.delete(target.id)
                    stats["deleted"] += 1
            else:
                stats["noop"] += 1

        # Phase 4: Global Link Evolution — 1 LLM call
        edges_before = self._count_edges()
        all_notes = self.memory_bank.list_notes()
        if len(all_notes) >= 2:
            self.link_evolver.link_evolve_all(all_notes, self.memory_bank)
        edges_after = self._count_edges()
        stats["edges"] = edges_after - edges_before

        self._batch_stats = stats
        _logger.info("write_batch DONE | session={!r} | added={} updated={} "
                     "deleted={} noop={} edges_new={} | bank_size={}",
                     session_label, stats["added"], stats["updated"],
                     stats["deleted"], stats["noop"], stats["edges"],
                     len(self.memory_bank.list_notes()))
        return results

    def write_conversation(
        self,
        session_batches: List[Tuple[str, List[str]]],
        timestamp: datetime,
    ) -> List[Note]:
        """Process ALL sessions of a conversation in one batched pass.

        Unlike write_batch (one session at a time), this processes all
        sessions together: S1 (Note Construction) for all turns first,
        then S2 (Memory Manager) sequentially, then S3 (Link Evolver)
        for cross-session connections.

        Args:
            session_batches: List of (session_label, turns) from all sessions.
            timestamp: Base timestamp for all notes.

        Returns:
            List of all notes created or updated.
        """
        total_turns = sum(len(turns) for _, turns in session_batches)
        _logger.info("write_conversation START | sessions={} | total_turns={} | bank_size={}",
                     len(session_batches), total_turns,
                     len(self.memory_bank.list_notes()))

        stats = {"added": 0, "updated": 0, "deleted": 0, "noop": 0, "edges": 0}
        all_results: List[Note] = []

        # Flatten all turns with their session labels
        all_turns: List[Tuple[str, str]] = []  # (session_label, turn_text)
        for label, turns in session_batches:
            for turn in turns:
                all_turns.append((label, turn))

        # Phase 1: Note Construction (S1) — batched: one LLM call for all turns
        _logger.info("write_conversation | Phase 1: Note Construction for {} turns", len(all_turns))
        all_contents = [f"[{label}] {turn}" for label, turn in all_turns]
        notes = self.note_constructor.build_batch(all_contents, timestamp)
        _logger.info("write_conversation | S1 done: {} notes built in 1 LLM call", len(notes))

        # Phase 2: Memory Manager (S2) — sequential decisions per turn
        _logger.info("write_conversation | Phase 2: Memory Manager for {} turns", len(notes))
        for i, (note, (label, turn)) in enumerate(zip(notes, all_turns)):
            content = f"[{label}] {turn}"
            e_new = self.note_constructor.backend.embed(content)
            existing = self.memory_bank.ann_search(e_new, k=self.retriever.k2)
            if not existing:
                existing = self.memory_bank.list_notes()[: self.retriever.k2]

            op, target = self.memory_manager.select_op(content, existing)

            if op == Op.ADD:
                self.memory_bank.add(note)
                stats["added"] += 1
                all_results.append(note)
            elif op == Op.UPDATE:
                updated = self._merge_update(target, note)
                self.memory_bank.add(updated)
                stats["updated"] += 1
                all_results.append(updated)
            elif op == Op.DELETE:
                if target is not None:
                    self.memory_bank.delete(target.id)
                    stats["deleted"] += 1
            else:
                stats["noop"] += 1

            if (i + 1) % 10 == 0:
                _logger.info("write_conversation | S2 progress {}/{} | bank_size={}",
                            i + 1, len(notes), len(self.memory_bank.list_notes()))

        # Phase 3: Link Evolution (S3) — global pass over all notes
        _logger.info("write_conversation | Phase 3: Link Evolution across all notes")
        edges_before = self._count_edges()
        self.link_evolver.link_evolve_all(
            self.memory_bank.list_notes(), self.memory_bank
        )
        edges_after = self._count_edges()
        stats["edges"] = edges_after - edges_before

        self._batch_stats = stats
        _logger.info("write_conversation DONE | added={} updated={} deleted={} "
                     "noop={} edges_new={} | bank_size={}",
                     stats["added"], stats["updated"], stats["deleted"],
                     stats["noop"], stats["edges"],
                     len(self.memory_bank.list_notes()))
        return all_results

    def cross_chunk_link_evolve(self) -> int:
        """Run a global link-evolution pass over all notes in the bank.

        Connects notes that were created in different session batches,
        which is critical for multi-hop temporal reasoning across sessions.
        Returns the number of new edges created.
        """
        notes = self.memory_bank.list_notes()
        if len(notes) < 2:
            _logger.debug("cross_chunk_link_evolve | too few notes ({}) — skipping", len(notes))
            return 0

        _logger.info("cross_chunk_link_evolve START | bank_size={}", len(notes))
        edges_before = self._count_edges()

        self.link_evolver.link_evolve_all(notes, self.memory_bank)

        edges_after = self._count_edges()
        new_edges = edges_after - edges_before
        _logger.info("cross_chunk_link_evolve DONE | new_edges={} | total_edges={}",
                     new_edges, edges_after)
        return new_edges

    def get_stats(self) -> Dict[str, object]:
        """Return pipeline statistics after batch ingestion.

        Returns dict with:
            total_nodes, total_edges, unique_keywords, unique_tags,
            batch_added, batch_updated, batch_deleted, batch_noop, batch_edges
        """
        notes = self.memory_bank.list_notes()
        all_keywords: List[str] = []
        all_tags: List[str] = []
        total_edges = 0
        for note in notes:
            all_keywords.extend(note.K)
            all_tags.extend(note.G)
            total_edges += len(note.L)

        return {
            "total_nodes": len(notes),
            "total_edges": total_edges,
            "unique_keywords": len(set(all_keywords)),
            "unique_tags": len(set(all_tags)),
            "batch_added": self._batch_stats.get("added", 0),
            "batch_updated": self._batch_stats.get("updated", 0),
            "batch_deleted": self._batch_stats.get("deleted", 0),
            "batch_noop": self._batch_stats.get("noop", 0),
            "batch_edges": self._batch_stats.get("edges", 0),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_edges(self) -> int:
        """Count total edges across all notes in the bank."""
        return sum(len(note.L) for note in self.memory_bank.list_notes())

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
