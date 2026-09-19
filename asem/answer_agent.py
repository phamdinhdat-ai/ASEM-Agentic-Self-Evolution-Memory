"""Answer agent for memory distillation and response generation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

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

# Reserve a few tokens when budgeting the prompt: the ~4-chars/token estimate can
# undercount (JSON punctuation, non-ASCII), and a 1-token overflow is a hard 400.
_SAFETY_MARGIN_TOKENS = 64

# Substrings identifying a "prompt too long" 400 from an OpenAI-compatible
# endpoint (vLLM: "This model's maximum context length is 8192 tokens").
_CONTEXT_OVERFLOW_HINTS = (
    "maximum context length",
    "context_length_exceeded",
    "reduce the length of the input",
    "too many tokens",
)

# `content` is the RAW conversation turn and it dominates the prompt (measured
# at ~60% of characters, ~600 tokens/note). The decision procedure only skims
# `description` / `keywords` / `entities` / `speaker` / `session_date`, so a
# short prefix of the turn is enough grounding and the rest is pure cost.
# Measured on ds_fixed: this cuts the answer prompt by roughly two thirds.
_CONTENT_CHAR_LIMIT = 200

# Fields that carry no answer signal but used to be serialised on every note.
# `timestamp_iso` is redundant with `session_date`, `utility` (the Q-value) is a
# retrieval-side number, and `tags` duplicate `keywords`.
_DROP_PAYLOAD_FIELDS = ("timestamp_iso", "utility", "tags")

# MEASURED (ds_fixed, 1790 questions, ASEM): the full edge list was **49.9% of
# the prompt** — 873 chars/note — because every note serialised all of its graph
# edges, including the thousands whose target was never retrieved and therefore
# cannot be read. Only edges BETWEEN notes in the same prompt are actionable, so
# those are kept verbatim (capped) and the remainder collapses to a tiny count
# map that still tells the model "this note is contradicted / corroborated".
_MAX_RELATIONS_PER_NOTE = 6
_MAX_RELATION_TYPES = 4


def _relation_digest(counts: dict) -> str:
    """Compact "same-topic:14, extends:3" form of the out-of-context edges."""
    if not counts:
        return ""
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:_MAX_RELATION_TYPES]
    return ", ".join(f"{rel}:{n}" for rel, n in top)


def _clip(text: str, limit: int) -> str:
    """Truncate `text` to `limit` chars on a word boundary (adds an ellipsis)."""
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:. ")
    return (cut or text[:limit]) + " …"


def _is_context_overflow(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in _CONTEXT_OVERFLOW_HINTS)


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
    # Hard ceiling on how many notes reach the prompt. Notes arrive in
    # relevance order, so the LOWEST-ranked are dropped first. None = no cap.
    # Fewer, better notes beat "everything that matched": a small backbone
    # degrades on long, low-signal contexts.
    max_context_notes: Optional[int] = None
    # Chars of the raw turn kept per note (0 = drop `content` entirely).
    content_char_limit: int = _CONTENT_CHAR_LIMIT

    def _select(self, candidates: List[Note]) -> List[Note]:
        """Keep the most relevant notes up to `max_context_notes`.

        Candidates are assumed to be in relevance order (the retriever ranks
        them; the recovery pass re-ranks by query similarity).
        """
        if not self.max_context_notes or len(candidates) <= self.max_context_notes:
            return list(candidates)
        kept = list(candidates[: self.max_context_notes])
        _log.debug(
            "Context cap: kept {}/{} notes (max_context_notes={})",
            len(kept), len(candidates), self.max_context_notes,
        )
        return kept

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
        budget = int(self.context_window) - int(self.max_tokens or 0) - _SAFETY_MARGIN_TOKENS
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

    def _call(self, prompt: str, max_tokens: Optional[int]) -> str:
        """One backend call, passing a per-call output cap where supported."""
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

    def _generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate, with a last-resort retry if the endpoint rejects the prompt length.

        The character-based estimate can undercount, and callers other than this
        class (e.g. an LLMRetryHandler) may not pass a cap at all. Rather than
        failing the whole benchmark run on an avoidable HTTP 400, shrink the
        output budget so that prompt + output fits the declared window.
        """
        try:
            return self._call(prompt, max_tokens)
        except Exception as exc:  # noqa: BLE001
            if not (self.context_window and _is_context_overflow(exc)):
                raise
            est = self._estimate_tokens(prompt)
            affordable = int(self.context_window) - est - _SAFETY_MARGIN_TOKENS
            if affordable < 64:
                # Even a minimal completion cannot fit — nothing left to give.
                raise
            requested = int(max_tokens) if max_tokens else None
            configured = int(self.max_tokens) if self.max_tokens else None
            new_cap = min(v for v in (requested, configured, affordable) if v is not None)
            _log.warning(
                "Context overflow (prompt ~{} tokens, requested cap {}): "
                "retrying with max_tokens={} against a {}-token window.",
                est, max_tokens, new_cap, self.context_window,
            )
            return self._call(prompt, new_cap)

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
            in_context = {n.id for n in notes}
            body = self.prompt_template.format(
                query=query,
                candidates=json.dumps(
                    [
                        self._note_payload(
                            n,
                            content_chars=self.content_char_limit,
                            in_context=in_context,
                        )
                        for n in notes
                    ]
                ),
            )
            # Second-chance pass: the previous answer was a refusal, so say so
            # and make the model re-read a wider candidate set.
            return f"{notice}\n\n{body}" if notice else body

        prompt, candidates = self._fit(self._select(candidates), render)
        if self.max_retries > 0:
            # Bind the cap INTO the retry handler: LLMRetryHandler calls
            # generate_fn(prompt) with NO kwargs, so passing self.backend.generate
            # directly silently dropped the per-call cap and fell back to the
            # client-wide max_tokens — which overflowed the context window
            # (6145 + 2048 > 8192) on the Qwen3-4B 8k endpoint.
            retry = LLMRetryHandler(
                lambda p: self._generate(p, self.max_tokens),
                max_retries=self.max_retries,
            )
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
            self._select(candidates),
            lambda notes: self.baseline_prompt_template.format(
                query=query, context=self._render_graph_context(notes)
            ),
        )
        return self._generate_resilient(prompt, max_tokens=self.max_tokens).strip()

    def _render_graph_context(self, candidates: List[Note]) -> str:
        """Render notes as compact numbered graph nodes.

        Each node carries only what answers a question: who said it, when, the
        entities it names, the distilled summary, its keywords and its typed
        edges. The raw turn is clipped to `content_char_limit` because it is the
        single largest field and the least information-dense.
        """
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
            lines = [header]

            # `summary` (X) is the distilled fact; `content` (c) is the raw turn.
            # Ingestion may fold facts into X while c holds only the dialogue, so
            # render both — but never twice, and never a full turn.
            if n.X and n.X != n.c:
                lines.append(f"    summary: {n.X}")
            content = _clip(n.c, self.content_char_limit)
            if content and content.rstrip(" …") != (n.X or "").rstrip(" …"):
                lines.append(f"    content: {content}")
            if n.K:
                lines.append(f"    keywords: {', '.join(n.K[:12])}")

            # Typed links: keep the edges whose target IS in this block (the model
            # can read those), collapse the rest to a count map. Serialising the
            # whole edge list was 49.9% of the prompt for zero extra signal.
            edges, other = [], {}
            for link in n.L:
                rel = link.relation or "linked"
                target = index_by_id.get(link.target_id)
                if target is not None and len(edges) < _MAX_RELATIONS_PER_NOTE:
                    edges.append(f"{rel} -> [{target}]")
                else:
                    other[rel] = other.get(rel, 0) + 1
            if edges:
                lines.append(f"    relations: {'; '.join(edges)}")
            digest = _relation_digest(other)
            if digest:
                lines.append(f"    also_linked: {digest}")

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
    def _note_payload(
        note: Note,
        *,
        content_chars: int = _CONTENT_CHAR_LIMIT,
        in_context: Optional[Iterable[str]] = None,
    ) -> dict:
        """Serialise a note for the answer prompt, keeping only what answers questions.

        Deliberately excludes `timestamp_iso` / `utility` / `tags` and clips the
        raw turn: the prompt is a *reading* context, not a dump of the bank row.
        `in_context` is the set of note ids sharing the prompt; relation edges are
        kept only when their target is in it (defaults to just this note), and the
        remainder is summarised as counts. Measured: the unfiltered edge list was
        49.9% of the prompt at 873 chars/note.
        """
        present = set(in_context) if in_context is not None else {note.id}

        edges, other = [], {}
        for link in note.L:
            rel = link.relation or "linked"
            if link.target_id in present and len(edges) < _MAX_RELATIONS_PER_NOTE:
                edges.append({"relation": rel, "target_id": link.target_id})
            else:
                other[rel] = other.get(rel, 0) + 1

        payload = {
            "id": note.id,
            "keywords": note.K,
            "description": note.X,
            "session_date": note.session_date,
            "entities": note.entities,
            "speaker": note.speaker,
            # Typed edges to other candidates. The relation label is what lets the
            # model connect and attribute facts instead of reading each note in
            # isolation.
            "relations": edges,
        }
        digest = _relation_digest(other)
        if digest:
            payload["also_linked"] = digest
        if content_chars > 0:
            content = _clip(note.c, content_chars)
            # Never pay for the same text twice: FastASEM-style notes set
            # `description == content`, and merged ASEM notes often embed the
            # raw turn in their description already.
            if content and content.rstrip(" …") not in (note.X or ""):
                payload["content"] = content
        return payload
