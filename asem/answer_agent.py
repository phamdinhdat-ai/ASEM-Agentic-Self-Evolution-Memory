"""Answer agent for memory distillation and response generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

from .backends.base import InferenceBackend
from .llm_validator import LLMRetryHandler, validate_distil_response
from .logging_utils import get_logger
from .note import Note, _try_extract_json

_log = get_logger("S4.answer")


@dataclass
class AnswerAgent:
    """Distil relevant notes and produce an answer."""

    backend: InferenceBackend
    prompt_template: str
    baseline_prompt_template: str
    # > 0: re-issue the prompt with a format correction when the LLM output
    # fails to parse or violates the expected schema (small-model support).
    max_retries: int = 0

    def distil_and_answer(self, query: str, candidates: List[Note]) -> Tuple[List[Note], str]:
        if not candidates:
            _log.debug("No candidates, using baseline answer")
            return [], self._baseline_answer(query, [])

        prompt = self.prompt_template.format(
            query=query,
            candidates=json.dumps([self._note_payload(n) for n in candidates]),
        )
        if self.max_retries > 0:
            retry = LLMRetryHandler(self.backend.generate, max_retries=self.max_retries)
            data, _attempt = retry.invoke(
                prompt,
                parse_fn=lambda raw: _try_extract_json(raw, expect_array=False),
                validate_fn=validate_distil_response,
            )
            parsed = self._parse_response_data(data)
        else:
            raw = self.backend.generate(prompt)
            parsed = self._parse_response(raw)
        if parsed is None:
            _log.warning("Distil JSON parse failed, falling back to all candidates")
            return candidates, self._baseline_answer(query, candidates)

        selected_ids, answer = parsed
        selected_notes = [n for n in candidates if n.id in selected_ids]
        if not selected_notes:
            selected_notes = candidates
        _log.debug("Distilled | selected={}/{}  answer={!r}",
                   len(selected_notes), len(candidates), answer[:60])
        return selected_notes, answer

    def _baseline_answer(self, query: str, candidates: List[Note]) -> str:
        context = "\n".join([
            f"- {note.c}" for note in candidates
        ])
        prompt = self.baseline_prompt_template.format(query=query, context=context)
        return self.backend.generate(prompt)

    def _parse_response(self, raw: str) -> Tuple[List[str], str] | None:
        data = _try_extract_json(raw, expect_array=False)
        return self._parse_response_data(data)

    @staticmethod
    def _parse_response_data(data) -> Tuple[List[str], str] | None:
        if not isinstance(data, dict):
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
