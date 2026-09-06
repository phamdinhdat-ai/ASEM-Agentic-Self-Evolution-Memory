"""Answer agent for memory distillation and response generation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

from .backends.base import InferenceBackend
from .llm_validator import LLMRetryHandler, _is_transient_network_error, validate_distil_response
from .logging_utils import get_logger
from .note import Note, _try_extract_json

_log = get_logger("S4.answer")


@dataclass
class AnswerAgent:
    """Distil relevant notes and produce an answer."""

    backend: InferenceBackend
    prompt_template: str
    baseline_prompt_template: str
    direct_mode: bool = False
    max_retries: int = 0

    def _generate_resilient(self, prompt: str) -> str:
        """Generate with retry on transient network errors (DNS blips, etc.).

        The direct/baseline answer paths call ``backend.generate`` directly
        (no JSON parsing), so they bypass the LLMRetryHandler. Without this,
        a single transient connection error during the QA phase would crash
        the whole benchmark run.
        """
        attempts = max(1, self.max_retries + 1)
        for attempt in range(attempts):
            try:
                return self.backend.generate(prompt)
            except Exception as exc:  # noqa: BLE001
                if _is_transient_network_error(exc) and attempt < attempts - 1:
                    backoff = min(2 ** attempt, 15)
                    _log.warning(
                        "Transient network error in answer (attempt {}/{}): {} — retrying in {}s",
                        attempt + 1, attempts, type(exc).__name__, backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise

    def distil_and_answer(self, query: str, candidates: List[Note]) -> Tuple[List[Note], str]:
        if not candidates:
            _log.debug("No candidates, using baseline answer")
            return [], self._baseline_answer(query, [])

        if self.direct_mode:
            answer = self.direct_answer(query, candidates)
            return candidates, answer

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

    def direct_answer(self, query: str, candidates: List[Note]) -> str:
        """Fast single-pass temporal QA answering without JSON distillation."""
        if not candidates:
            return "I don't know"

        # Chronologically sort notes
        sorted_notes = sorted(candidates, key=lambda n: n.t if n.t else datetime.min)
        context_items = []
        for n in sorted_notes:
            date_prefix = f"[{n.session_date}] " if n.session_date else f"[{n.t.strftime('%d %B %Y')}] "
            entities_str = f" (Entities: {', '.join(n.entities)})" if n.entities else ""
            # Surface keywords/description too: ingestion may merge facts into
            # K/G/X while c holds only a single headline sentence. Without these
            # the LLM cannot see facts that were folded into a merged note.
            keywords_str = f" (Keywords: {', '.join(n.K[:12])})" if n.K else ""
            desc_str = f" (Description: {n.X})" if (n.X and n.X != n.c) else ""
            context_items.append(f"- {date_prefix}{n.c}{entities_str}{keywords_str}{desc_str}")

        context = "\n".join(context_items)
        prompt = self.baseline_prompt_template.format(query=query, context=context)
        return self._generate_resilient(prompt).strip()

    def _baseline_answer(self, query: str, candidates: List[Note]) -> str:
        context = "\n".join([
            f"- {note.c}" for note in candidates
        ])
        prompt = self.baseline_prompt_template.format(query=query, context=context)
        return self._generate_resilient(prompt).strip()

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
        return [str(item) for item in selected_ids], str(answer).strip()

    @staticmethod
    def _note_payload(note: Note) -> dict:
        return {
            "id": note.id,
            "keywords": note.K,
            "tags": note.G,
            "description": note.X,
            "content": note.c,
            "utility": note.q,
            "session_date": note.session_date,
            "entities": note.entities,
        }
