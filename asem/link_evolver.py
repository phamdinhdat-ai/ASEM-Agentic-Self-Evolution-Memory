"""Dynamic linking and memory evolution with batched sparse evolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Set

from .backends.base import InferenceBackend
from .logging_utils import get_logger
from .memory_bank import MemoryBank
from .note import Note

_logger = get_logger(__name__)

# Relations that trigger memory evolution (B2: sparse evolution gate).
# Weak relations like "same-topic" or "temporal" don't justify re-describing
# the neighbor — only structural changes (contradict, extend, causal) do.
_STRONG_RELATIONS: Set[str] = {"contradicts", "extends", "causal"}
_EVOLVE_BATCH_TEMPLATE = """Revise the following existing memory notes given new information. For each note, merge keywords/tags and update the description. Output a JSON array with one object per note.

Existing notes:
{existing_notes}

New note:
{new_note}

Return ONLY a JSON array like:
[{{"id": "<note_id>", "keywords": [...], "tags": [...], "description": "updated summary"}}, ...]
Output ONLY the JSON array, nothing else."""


@dataclass
class LinkEvolver:
    """Link a new note with neighbors and evolve their attributes.

    B1 (batch evolution): all qualifying neighbors are evolved in a single LLM
    call instead of one per neighbor, reducing k calls to 1.

    B2 (sparse evolution): only neighbors with a strong relationship
    (contradicts, extends, causal) are evolved.  Weak relations (same-topic,
    temporal, semantic) skip evolution, saving 40-60 % of evolution calls.
    """

    backend: InferenceBackend
    link_prompt_template: str
    evolve_prompt_template: str
    k: int = 5

    def link_and_evolve(self, m_new: Note, M: MemoryBank) -> None:
        neighbors = M.ann_search(m_new.e, k=self.k)
        if not neighbors:
            _logger.debug("link_and_evolve | note {} has no neighbors (empty bank)", m_new.id)
            return

        _logger.debug("link_and_evolve | note {} | neighbors={}",
                      m_new.id, [n.id for n in neighbors])

        relations = self._generate_links(m_new, neighbors)
        _logger.debug("link_and_evolve | {} relations generated", len(relations))

        self._apply_links(m_new, neighbors, relations, M)

        # B2 — only evolve neighbors with a strong relationship
        strong_ids = self._strong_neighbor_ids(relations, m_new.id)
        strong_neighbors = [n for n in neighbors if n.id in strong_ids]

        if not strong_neighbors:
            _logger.debug("link_and_evolve | no strong relations → skip evolution")
            return

        _logger.info("link_and_evolve | evolving {} neighbors (strong relations: {})",
                     len(strong_neighbors), strong_ids)

        # B1 — batch-evolve all strong neighbors in one LLM call
        updated_notes = self._evolve_notes_batched(m_new, strong_neighbors)
        neighbor_map = {n.id: n for n in strong_neighbors}
        for updated in updated_notes:
            orig = neighbor_map.get(updated.id)
            if orig is None:
                continue
            M.update(orig.id, {
                "K": updated.K,
                "G": updated.G,
                "X": updated.X,
            })
            _logger.debug("link_and_evolve | evolved note {} K={} G={}",
                          orig.id, updated.K[:3], updated.G[:3])

    # ------------------------------------------------------------------
    # B2 helper: extract neighbor IDs with strong relations
    # ------------------------------------------------------------------

    @staticmethod
    def _strong_neighbor_ids(relations: List[dict], new_id: str) -> Set[str]:
        strong: Set[str] = set()
        for rel in relations:
            relation_type = str(rel.get("relation", "")).lower()
            if relation_type not in _STRONG_RELATIONS:
                continue
            source = str(rel.get("source", ""))
            target = str(rel.get("target", ""))
            if source == new_id:
                strong.add(target)
            elif target == new_id:
                strong.add(source)
        return strong

    # ------------------------------------------------------------------
    # Link generation
    # ------------------------------------------------------------------

    def _generate_links(self, m_new: Note, neighbors: List[Note]) -> List[dict]:
        prompt = self.link_prompt_template.format(
            new_note=json.dumps(self._note_payload(m_new)),
            neighbors=json.dumps([self._note_payload(n) for n in neighbors]),
        )
        raw = self.backend.generate(prompt)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def _apply_links(
        self,
        m_new: Note,
        neighbors: List[Note],
        relations: List[dict],
        M: MemoryBank,
    ) -> None:
        if not relations:
            return

        neighbor_map = {note.id: note for note in neighbors}
        for rel in relations:
            source = str(rel.get("source", ""))
            target = str(rel.get("target", ""))
            if source == m_new.id and target in neighbor_map:
                self._add_link(m_new, target)
                self._add_link(neighbor_map[target], m_new.id)
                M.update(target, {"L": neighbor_map[target].L})
            elif target == m_new.id and source in neighbor_map:
                self._add_link(m_new, source)
                self._add_link(neighbor_map[source], m_new.id)
                M.update(source, {"L": neighbor_map[source].L})

        M.update(m_new.id, {"L": m_new.L})

    # ------------------------------------------------------------------
    # B1: batch evolution — one LLM call for all qualifying neighbors
    # ------------------------------------------------------------------

    def _evolve_notes_batched(
        self, m_new: Note, neighbors: List[Note]
    ) -> List[Note]:
        """Evolve multiple neighbors in a single batched LLM call."""
        existing_payload = json.dumps(
            [self._note_payload(n) for n in neighbors]
        )
        prompt = _EVOLVE_BATCH_TEMPLATE.format(
            existing_notes=existing_payload,
            new_note=json.dumps(self._note_payload(m_new)),
        )
        raw = self.backend.generate(prompt)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        result: List[Note] = []
        neighbor_map = {n.id: n for n in neighbors}
        for item in data:
            if not isinstance(item, dict):
                continue
            note_id = str(item.get("id", ""))
            orig = neighbor_map.get(note_id)
            if orig is None:
                continue

            result.append(Note(
                id=orig.id,
                c=orig.c,
                t=orig.t,
                K=list(item.get("keywords", orig.K)),
                G=list(item.get("tags", orig.G)),
                X=str(item.get("description", orig.X)),
                e=orig.e,
                L=orig.L,
                z=orig.z,
                q=orig.q,
            ))

        # Fallback: if batch parsing failed, fall through to individual
        # evolution for any neighbor not covered by the batch result
        covered = {n.id for n in result}
        for neighbor in neighbors:
            if neighbor.id not in covered:
                updated = self._evolve_note_single(neighbor, m_new)
                if updated is not None:
                    result.append(updated)

        return result

    def _evolve_note_single(self, note: Note, m_new: Note) -> Note | None:
        """Single-note evolution (fallback when batch parsing fails)."""
        prompt = self.evolve_prompt_template.format(
            existing_note=json.dumps(self._note_payload(note)),
            new_note=json.dumps(self._note_payload(m_new)),
        )
        raw = self.backend.generate(prompt)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        keywords = list(data.get("keywords", note.K))
        tags = list(data.get("tags", note.G))
        description = str(data.get("description", note.X))

        return Note(
            id=note.id,
            c=note.c,
            t=note.t,
            K=keywords,
            G=tags,
            X=description,
            e=note.e,
            L=note.L,
            z=note.z,
            q=note.q,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _note_payload(note: Note) -> dict:
        return {
            "id": note.id,
            "keywords": note.K,
            "tags": note.G,
            "description": note.X,
        }

    @staticmethod
    def _add_link(note: Note, target_id: str) -> None:
        if target_id not in note.L:
            note.L.append(target_id)
