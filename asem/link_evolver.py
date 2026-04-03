"""Dynamic linking and memory evolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from .backends.base import InferenceBackend
from .memory_bank import MemoryBank
from .note import Note


@dataclass
class LinkEvolver:
    """Link a new note with neighbors and evolve their attributes."""

    backend: InferenceBackend
    link_prompt_template: str
    evolve_prompt_template: str
    k: int = 5

    def link_and_evolve(self, m_new: Note, M: MemoryBank) -> None:
        neighbors = M.ann_search(m_new.e, k=self.k)
        if not neighbors:
            return

        relations = self._generate_links(m_new, neighbors)
        self._apply_links(m_new, neighbors, relations, M)

        for note in neighbors:
            updated = self._evolve_note(note, m_new)
            if updated is not None:
                M.update(note.id, {
                    "K": updated.K,
                    "G": updated.G,
                    "X": updated.X,
                })

    def _generate_links(self, m_new: Note, neighbors: List[Note]) -> List[dict]:
        prompt = self.link_prompt_template.format(
            new_note=self._note_payload(m_new),
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

    def _evolve_note(self, note: Note, m_new: Note) -> Note | None:
        prompt = self.evolve_prompt_template.format(
            existing_note=self._note_payload(note),
            new_note=self._note_payload(m_new),
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
