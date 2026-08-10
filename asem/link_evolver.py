"""Dynamic linking and memory evolution with batched sparse evolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import numpy as np

from .backends.base import InferenceBackend
from .llm_validator import (
    LLMRetryHandler,
    validate_batch_notes,
    validate_link_array,
    validate_note_fields,
)
from .logging_utils import get_logger
from .memory_bank import MemoryBank
from .note import LinkRecord, Note, _try_extract_json  # robust JSON parser for LLM output

_log = get_logger("S3.linker")

# Relations that trigger memory evolution (B2: sparse evolution gate).
# Weak relations like "same-topic" or "temporal" don't justify re-describing
# the neighbor — only structural changes (contradict, extend, causal) do.
_STRONG_RELATIONS: Set[str] = {"contradicts", "extends", "causal"}

# Every relation label that may be stored on an edge; anything else degrades
# to "semantic" before persistence.
_VALID_RELATIONS: Set[str] = (
    _STRONG_RELATIONS | {"same-topic", "temporal", "semantic"}
)
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts"


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
    link_tau: float = 0.35  # only feed the S3 link LLM neighbors at/above this cosine
    # > 0: re-issue prompts with a format correction when LLM output fails
    # to parse or violates the expected schema (small-model support).
    max_retries: int = 0
    # Batch evolution prompt (one LLM call for all qualifying neighbors).
    # If None, loads the enhanced file-based template from
    # data/prompts/P3_batch_evolution.txt.
    evolve_batch_template: Optional[str] = None

    def _retry(self) -> Optional[LLMRetryHandler]:
        if self.max_retries <= 0:
            return None
        return LLMRetryHandler(self.backend.generate, max_retries=self.max_retries)

    def link_and_evolve(self, m_new: Note, M: MemoryBank) -> None:
        neighbors = M.ann_search(m_new.e, k=self.k)
        if not neighbors:
            _log.debug("No neighbors found for linking")
            return

        # A note must never be its own neighbor — drop the just-added note
        # from the candidate set (it scores 1.0 against itself and would
        # otherwise consume a neighbor slot / self-link).
        neighbors = [n for n in neighbors if n.id != m_new.id]
        if not neighbors:
            _log.debug("No neighbors after self-exclusion for linking")
            return

        # NGMC Tier 1 — retrieval-proposed linking: only feed the link LLM
        # neighbors that are genuinely related (cosine >= link_tau). This cuts
        # the S3 prompt size and focuses generation on real relations.
        if self.link_tau > 0.0:
            neighbors = [
                n for n in neighbors
                if self._cosine(m_new.e, n.e) >= self.link_tau
            ]
            if not neighbors:
                _log.debug("No neighbors above link_tau={} for linking", self.link_tau)
                return

        relations = self._generate_links(m_new, neighbors)
        self._apply_links(m_new, neighbors, relations, M)
        _log.info("Links generated | new_id={}  neighbors={}  relations={}",
                  m_new.id[:8], len(neighbors), len(relations))

        # B2 — only evolve neighbors with a strong relationship
        strong_ids = self._strong_neighbor_ids(relations, m_new.id)
        strong_neighbors = [n for n in neighbors if n.id in strong_ids]

        if not strong_neighbors:
            _log.debug("No strong relations, skipping evolution")
            return

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
        _log.success("Evolved {}/{} strong neighbors", len(updated_notes), len(strong_neighbors))

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
            new_note=self._note_payload(m_new),
            neighbors=json.dumps([self._note_payload(n) for n in neighbors]),
        )
        retry = self._retry()
        if retry is not None:
            data, _attempt = retry.invoke(
                prompt,
                parse_fn=lambda raw: _try_extract_json(raw, expect_array=True),
                validate_fn=lambda d: validate_link_array(
                    d,
                    valid_source_id=m_new.id,
                    valid_target_ids={n.id for n in neighbors},
                    allow_unknown_relations=True,
                ),
            )
        else:
            data = _try_extract_json(self.backend.generate(prompt), expect_array=True)
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
            # Persist the LLM-identified relation type on both ends of the
            # edge so downstream consumers (get_link_graph, retrieval) can
            # use it instead of heuristically re-inferring it.
            relation = str(rel.get("relation", "")).lower() or "semantic"
            if relation not in _VALID_RELATIONS:
                relation = "semantic"  # unknown labels degrade to the fallback type
            if source == m_new.id and target in neighbor_map:
                self._add_link(m_new, target, relation)
                self._add_link(neighbor_map[target], m_new.id, relation)
                M.update(target, {"L": neighbor_map[target].L})
            elif target == m_new.id and source in neighbor_map:
                self._add_link(m_new, source, relation)
                self._add_link(neighbor_map[source], m_new.id, relation)
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
        template = self.evolve_batch_template
        if template is None:
            template = (_PROMPTS_DIR / "P3_batch_evolution.txt").read_text(
                encoding="utf-8"
            )
        prompt = template.format(
            existing_notes=existing_payload,
            new_note=json.dumps(self._note_payload(m_new)),
        )
        retry = self._retry()
        if retry is not None:
            data, _attempt = retry.invoke(
                prompt,
                parse_fn=lambda raw: _try_extract_json(raw, expect_array=True),
                validate_fn=validate_batch_notes,
            )
        else:
            data = _try_extract_json(self.backend.generate(prompt), expect_array=True)

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
            existing_note=self._note_payload(note),
            new_note=self._note_payload(m_new),
        )
        retry = self._retry()
        if retry is not None:
            data, _attempt = retry.invoke(
                prompt,
                parse_fn=lambda raw: _try_extract_json(raw, expect_array=False),
                validate_fn=validate_note_fields,
            )
        else:
            data = _try_extract_json(self.backend.generate(prompt), expect_array=False)
        if not isinstance(data, dict):
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
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _add_link(note: Note, target_id: str, relation: str = "linked") -> None:
        if not any(link.target_id == target_id for link in note.L):
            note.L.append(LinkRecord(target_id=target_id, relation=relation))
