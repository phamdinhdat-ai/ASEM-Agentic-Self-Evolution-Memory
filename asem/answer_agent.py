"""Answer agent for memory distillation and response generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

from .backends.base import InferenceBackend
from .logging_utils import get_logger
from .note import Note

_logger = get_logger(__name__)


@dataclass
class AnswerAgent:
    """Distil relevant notes and produce an answer."""

    backend: InferenceBackend
    prompt_template: str
    baseline_prompt_template: str

    def distil_and_answer(self, query: str, candidates: List[Note]) -> Tuple[List[Note], str]:
        if not candidates:
            _logger.debug("distil_and_answer | no candidates, using baseline")
            return [], self._baseline_answer(query, [])

        _logger.debug("distil_and_answer | query={!r} | candidates={}",
                      query[:120], [n.id for n in candidates])

        prompt = self.prompt_template.format(
            query=query,
            candidates=json.dumps([self._note_payload(n) for n in candidates]),
        )
        raw = self.backend.generate(prompt)
        parsed = self._parse_response(raw)
        if parsed is None:
            _logger.warning("distil_and_answer | parse failed, using baseline | raw={!r}", raw[:100])
            return candidates, self._baseline_answer(query, candidates)

        selected_ids, answer = parsed
        selected_notes = [n for n in candidates if n.id in selected_ids]
        if not selected_notes:
            _logger.debug("distil_and_answer | no selected_ids match, falling back to all candidates")
            selected_notes = candidates

        _logger.info("distil_and_answer → selected={} answer={!r}",
                     [n.id for n in selected_notes], answer[:150])
        return selected_notes, answer

    def _baseline_answer(self, query: str, candidates: List[Note]) -> str:
        context = "\n".join([
            f"- {note.c}" for note in candidates
        ])
        prompt = self.baseline_prompt_template.format(query=query, context=context)
        return self.backend.generate(prompt)

    def _parse_response(self, raw: str) -> Tuple[List[str], str] | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        selected_ids = data.get("selected_ids")
        answer = data.get("answer")
        if not isinstance(selected_ids, list) or answer is None:
            return None
        return [str(item) for item in selected_ids], str(answer)

    @staticmethod
    def _note_payload(note: Note) -> dict:
        return {
            "id": note.id,
            "keywords": note.K,
            "tags": note.G,
            "description": note.X,
            "content": note.c,
            "utility": note.q,
        }
