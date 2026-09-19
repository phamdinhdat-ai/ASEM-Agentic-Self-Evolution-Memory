"""Answer agent for memory distillation and response generation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from .backends.base import InferenceBackend
from .llm_validator import LLMRetryHandler, _is_transient_network_error, validate_distil_response
from .logging_utils import get_logger
from .note import Note, _try_extract_json

_log = get_logger("S4.answer")

# Answers that refuse to answer. Used to trigger a second-chance retrieval pass
# and to report an abstention rate alongside the accuracy metrics.
_REFUSAL_START_RE = re.compile(
    r"^\s*(?:sorry[,.]?\s*)?(?:i\s*(?:don'?t|do not)\s+know|i'?m not sure|"
    r"i\s+(?:cannot|can'?t)\s+(?:determine|find|answer)|"
    r"unable to (?:determine|find)|no information|not mentioned|"
    r"insufficient (?:information|evidence))",
    re.IGNORECASE,
)
# "…but the notes do not specify its title" — the question is left unanswered.
_NOTE_GAP_RE = re.compile(
    r"\bnotes?\s+(?:do(?:es)?\s+not|don'?t)\s+"
    r"(?:say|mention|specify|contain|include|give)",
    re.IGNORECASE,
)

# A *long* answer that merely appends a small caveat to a real fact is treated as
# an answer, so the recovery pass never throws away a correct partial answer.
_ABSTENTION_MAX_CHARS = 240


def is_abstention(answer: str, *, max_chars: int = _ABSTENTION_MAX_CHARS) -> bool:
    """True when an answer refuses / leaves the question unanswered.

    Two shapes count as an abstention:
      * the answer OPENS with a refusal ("I don't know", "I cannot determine…");
      * the answer is short AND reports what the notes fail to say, i.e. the
        model produced no fact for the question.
    """
    text = (answer or "").strip()
    if not text:
        return True
    if _REFUSAL_START_RE.match(text):
        return True
    return bool(_NOTE_GAP_RE.search(text)) and len(text) <= max_chars


@dataclass
class AnswerAgent:
    """Distil relevant notes and produce an answer."""

    backend: InferenceBackend
    prompt_template: str
    baseline_prompt_template: str
    direct_mode: bool = False
    max_retries: int = 0
    # Per-call output cap. Without this the backend's client-wide `max_tokens`
    # is used, which on a small-context model (e.g. an 8192-token window) can
    # leave too little room for the prompt and cause a hard HTTP 400.
    max_tokens: Optional[int] = None
    # The model's total context window. The rendered prompt is trimmed so that
    # `prompt + max_tokens <= context_window`. None disables trimming.
    context_window: Optional[int] = None

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Cheap token estimate (~4 chars/token).

        Deliberately conservative: it only decides how many low-ranked notes to
        drop, and dropping one low-ranked note is harmless while failing to drop
        one causes a hard 400 on a small-context model.
        """
        return max(1, len(text) // 4)

    def _fit(self, candidates: List[Note], render) -> Tuple[str, List[Note]]:
        """Render a prompt that fits ``context_window`` minus ``max_tokens``.

        Drops the LOWEST-ranked candidate (last in retrieval order) until the
        prompt fits, always keeping at least one note.
        """
        kept = list(candidates)
        prompt = render(kept)
        if not self.context_window:
            return prompt, kept
        budget = int(self.context_window) - int(self.max_tokens or 0)
        while len(kept) > 1 and self._estimate_tokens(prompt) > budget:
            kept.pop()
            prompt = render(kept)
        est = self._estimate_tokens(prompt)
        if est > budget:
            _log.warning(
                "Answer prompt still ~{} tokens after trimming to 1 note "
                "(budget {} = context_window {} - max_tokens {}). Raise the "
                "model context window or lower max_tokens.",
                est, budget, self.context_window, int(self.max_tokens or 0),
            )
        elif len(kept) < len(candidates):
            _log.debug(
                "Prompt budget: kept {}/{} notes (~{} tokens, budget {})",
                len(kept), len(candidates), est, budget,
            )
        return prompt, kept

    def _generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate, using a per-call output cap when the backend supports it."""
        if not max_tokens:
            return self.backend.generate(prompt)
        try:
            return self.backend.generate(prompt, max_tokens=max_tokens)
        except TypeError as exc:
            # Backend/wrapper without per-call kwargs — fall back to its default.
            if "unexpected keyword" in str(exc) or "positional argument" in str(exc):
                _log.debug("Backend ignores per-call max_tokens: {}", exc)
                return self.backend.generate(prompt)
            raise

    def _generate_resilient(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate with retry on transient network errors (DNS blips, etc.).

        The direct/baseline answer paths call ``backend.generate`` directly
        (no JSON parsing), so they bypass the LLMRetryHandler. Without this,
        a single transient connection error during the QA phase would crash
        the whole benchmark run.
        """
        attempts = max(1, self.max_retries + 1)
        for attempt in range(attempts):
            try:
                return self._generate(prompt, max_tokens)
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

    def distil_and_answer(
        self,
        query: str,
        candidates: List[Note],
        notice: Optional[str] = None,
    ) -> Tuple[List[Note], str]:
        if not candidates:
            _log.debug("No candidates, using baseline answer")
            return [], self._baseline_answer(query, [])

        if self.direct_mode:
            answer = self.direct_answer(query, candidates)
            return candidates, answer

        def render(notes: List[Note]) -> str:
            body = self.prompt_template.format(
                query=query,
                candidates=json.dumps([self._note_payload(n) for n in notes]),
            )
            # Second-chance pass: the previous answer was a refusal, so say so
            # and make the model re-read a wider candidate set.
            return f"{notice}\n\n{body}" if notice else body

        prompt, candidates = self._fit(candidates, render)
        if self.max_retries > 0:
            retry = LLMRetryHandler(self.backend.generate, max_retries=self.max_retries)
            data, _attempt = retry.invoke(
                prompt,
                parse_fn=lambda raw: _try_extract_json(raw, expect_array=False),
                validate_fn=validate_distil_response,
            )
            parsed = self._parse_response_data(data)
        else:
            raw = self._generate(prompt, self.max_tokens)
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
        """Fast single-pass temporal QA answering without JSON distillation.

        Each candidate is rendered as a labelled graph node — who said it, the
        entities it mentions, and its typed relations to the other notes in the
        same block. The relation labels and the ``speaker`` are what let the
        model resolve attribution (who said/did what) instead of guessing from
        flat, de-contextualised text.
        """
        if not candidates:
            return "I don't know"

        prompt, _ = self._fit(
            candidates,
            lambda notes: self.baseline_prompt_template.format(
                query=query, context=self._render_graph_context(notes)
            ),
        )
        return self._generate_resilient(prompt, max_tokens=self.max_tokens).strip()

    def _render_graph_context(self, candidates: List[Note]) -> str:
        """Render notes as numbered graph nodes (speaker / entities / relations)."""
        # Chronologically sort notes, then number them so relation edges can
        # reference other notes inside the SAME context block.
        sorted_notes = sorted(candidates, key=lambda n: n.t if n.t else datetime.min)
        index_by_id = {n.id: i + 1 for i, n in enumerate(sorted_notes)}

        context_items = []
        for i, n in enumerate(sorted_notes):
            date_str = n.session_date or (n.t.strftime('%d %B %Y') if n.t else "")
            header = f"[{i + 1}] date={date_str}"
            if n.speaker:
                header += f" speaker={n.speaker}"
            if n.entities:
                header += f" entities=[{', '.join(n.entities)}]"
            lines = [header, f"    content: {n.c}"]

            # Surface K/G/X: ingestion may merge facts into these while `c`
            # holds only the raw turn. Without them the LLM cannot see facts
            # that were folded into a merged note.
            if n.X and n.X != n.c:
                lines.append(f"    summary: {n.X}")
            if n.K:
                lines.append(f"    keywords: {', '.join(n.K[:12])}")

            # Typed links to notes present in this same context block.
            edges = []
            for link in n.L:
                target = index_by_id.get(link.target_id)
                if target is not None:
                    edges.append(f"{link.relation or 'linked'} -> [{target}]")
            if edges:
                lines.append(f"    relations: {'; '.join(edges)}")

            context_items.append("\n".join(lines))

        return "\n".join(context_items)

    def _baseline_answer(self, query: str, candidates: List[Note]) -> str:
        context_items = []
        for note in candidates:
            date_prefix = f"[{note.session_date}] " if note.session_date else (f"[{note.t.strftime('%d %B %Y')}] " if note.t else "")
            context_items.append(f"- {date_prefix}{note.c}")
        context = "\n".join(context_items)
        prompt = self.baseline_prompt_template.format(query=query, context=context)
        return self._generate_resilient(prompt, max_tokens=self.max_tokens).strip()

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
            "timestamp_iso": note.timestamp_iso or (note.t.isoformat() if note.t else None),
            "entities": note.entities,
            "speaker": note.speaker,
            # Typed edges to other candidates (matched by target_id). The
            # relation label is what lets the model connect and attribute facts
            # instead of reading each note in isolation.
            "relations": [
                {"relation": link.relation or "linked", "target_id": link.target_id}
                for link in note.L
            ],
        }
