"""Answer agent for memory distillation and response generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

from .backends.base import InferenceBackend
from .note import Note


@dataclass
class AnswerAgent:
    """Distil relevant notes and produce an answer."""

    backend: InferenceBackend
    prompt_template: str
    baseline_prompt_template: str

    def distil_and_answer(self, query: str, candidates: List[Note]) -> Tuple[List[Note], str]:
        if not candidates:
            return [], self._baseline_answer(query, [])

        prompt = self.prompt_template.format(
            query=query,
            candidates=json.dumps([self._note_payload(n) for n in candidates]),
        )
        raw = self.backend.generate(prompt)
        parsed = self._parse_response(raw)
        if parsed is None:
            return candidates, self._baseline_answer(query, candidates)

        selected_ids, answer = parsed
        selected_notes = [n for n in candidates if n.id in selected_ids]
        if not selected_notes:
            selected_notes = candidates
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
