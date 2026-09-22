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
from .token_budget import (
    SAFETY_MARGIN_TOKENS as _SAFETY_MARGIN_TOKENS,
    default_output_cap,
    estimate_tokens,
    fit_items,
    generate_with_cap,
    resolve_budget,
)

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

# The prompt-budget arithmetic (estimate, safety margin, trimming) is shared with
# the baselines so every system in a comparison respects the same number of
# tokens for the answer call. See `asem/token_budget.py`.

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

# ---------------------------------------------------------------------------
# Rendering the retrieved notes
# ---------------------------------------------------------------------------
# The distil path used to hand the model a JSON array of note objects. Two
# problems with that, both visible in the dumped prompts
# (`scratch_diag/context_samples/`):
#
#   1. INDIRECTION. Copying a 36-char UUID is a failure mode for a small model,
#      and `relations[].target_id` forced a three-step join (find id -> match to
#      another object's id -> read that object). Rank numbers ("[3]") remove
#      both, and numbers survive tokenisation far better than UUIDs.
#   2. UNREADABLE PROSE. An evolved note's `description` can reach ~1,700 chars
#      of run-on third-person text (the evolution prompt asks for 1-2 sentences
#      but this is not enforced), serialised as ONE line with `\u2014` escapes.
#      Splitting it into sentence bullets with an explicit speaker/date header
#      is what makes the same information readable.
#
# So the context is now a numbered, line-oriented block per note. The model is
# still asked for JSON *output* (`{"selected_ids": [1, 3], "answer": ...}`), but
# the INPUT is prose it can skim.
_FACT_CHAR_LIMIT = 420        # chars of `description` rendered per note
_MAX_FACT_BULLETS = 4         # sentences of `description` rendered per note
_MAX_TOPICS = 8               # keywords rendered (evolved notes carry up to ~50)
_WRAP_WIDTH = 104             # soft wrap for rendered facts / raw turns
_INDENT = "      "            # continuation indent inside a note block

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\u201c])")


def _human_date(note: Note) -> str:
    """Render a note's date as `20 April 2023` (fallback: the stored string).

    `session_date` is either an ISO timestamp ("2023-04-20T16:15:00Z") or a raw
    LoCoMo date ("8 May 2023"). ISO timestamps are unreadable inside a prompt and
    invite the model to copy the time-of-day into a date answer, so both forms
    are normalised to day-month-year.
    """
    for raw in (note.session_date, getattr(note, "timestamp_iso", None)):
        if not raw:
            continue
        text = str(raw).strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text).strftime("%d %B %Y").lstrip("0")
        except ValueError:
            pass
        for fmt in ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(raw).strip(), fmt).strftime("%d %B %Y").lstrip("0")
            except ValueError:
                continue
    if note.t is not None:
        try:
            return note.t.strftime("%d %B %Y").lstrip("0")
        except Exception:  # noqa: BLE001
            pass
    return "date unknown"


def _sentences(text: str) -> list:
    """Split prose into sentences (cheap, punctuation-based)."""
    text = " ".join((text or "").split())
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _wrap(text: str, width: int = _WRAP_WIDTH, indent: str = _INDENT) -> str:
    """Soft-wrap `text` on spaces, indenting continuation lines."""
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    if not lines:
        return indent + "…"
    return ("\n" + indent).join(lines)


def _render_notes_block(
    notes: List[Note],
    *,
    content_chars: int = _CONTENT_CHAR_LIMIT,
    fact_chars: int = _FACT_CHAR_LIMIT,
    max_bullets: int = _MAX_FACT_BULLETS,
) -> str:
    """Render the retrieved notes as numbered, readable evidence blocks.

    Every note becomes:

        [n] DATE · said by SPEAKER · about: ent1, ent2
            • first fact sentence (wrapped)
            • second fact sentence
            • [+k more facts]
            turn: "raw turn, clipped"
            links: [3] same-topic, [6] extends (+4 more not in this list)

    Relation targets are printed as the RANK of the other note in this same
    block, so the model never has to join ids. Edges whose target is not in the
    block collapse to a count, because their content cannot be read.
    """
    if not notes:
        return "(no memory notes retrieved)"

    index_by_id = {n.id: i + 1 for i, n in enumerate(notes)}
    blocks: List[str] = []

    for rank, note in enumerate(notes, start=1):
        header = f"[{rank}] {_human_date(note)}"
        if note.speaker:
            header += f" · said by {note.speaker}"
        entities = [e for e in (note.entities or []) if e]
        if entities:
            header += f" · about: {', '.join(entities[:6])}"
        lines = [header]

        # --- facts: the stored description, split into readable sentences -----
        facts = _sentences(note.X)
        shown, used, clipped = [], 0, False
        for sentence in facts[:max_bullets]:
            room = fact_chars - used
            if room <= 40:
                break
            piece = _clip(sentence, room)
            if len(piece) < len(sentence):
                clipped = True
            shown.append(piece)
            used += len(piece)
        if not shown and note.X:
            shown = [_clip(note.X, fact_chars)]
        lines += [f"{_INDENT}• {_wrap(s, indent=_INDENT + '  ')}" for s in shown]
        # Sentences that never made it in full (dropped by the bullet cap, the char
        # budget, or cut mid-way) are reported so the model knows the note continues.
        hidden = max(0, len(facts) - len(shown)) + (1 if clipped else 0)
        if hidden:
            lines.append(f"{_INDENT}• [+{hidden} more fact(s) in this note, not shown]")

        # --- the raw turn, only when it adds words the facts do not have ------
        if content_chars > 0:
            raw = _clip(note.c, content_chars)
            if raw and raw.rstrip(" …") not in (note.X or ""):
                lines.append(f"{_INDENT}turn: \"{_wrap(raw, indent=_INDENT + ' ' * 7)}\"")

        # --- topics: keywords are retrieval metadata, capped hard -------------
        topics = [k for k in (note.K or []) if k][:_MAX_TOPICS]
        if topics:
            lines.append(f"{_INDENT}topics: {', '.join(topics)}")

        # --- links: ranks inside this block, counts outside -------------------
        edges, other = [], {}
        for link in note.L:
            rel = link.relation or "linked"
            target = index_by_id.get(link.target_id)
            if target is not None and len(edges) < _MAX_RELATIONS_PER_NOTE:
                edges.append(f"[{target}] {rel}")
            else:
                other[rel] = other.get(rel, 0) + 1
        if edges or other:
            link_line = f"{_INDENT}links: "
            if edges:
                link_line += ", ".join(edges)
            digest = _relation_digest(other)
            if digest:
                prefix = "  (" if edges else "("
                link_line += (f"{prefix}+{sum(other.values())} more not in this list: "
                              f"{digest})")
            lines.append(link_line)

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _relation_digest(counts: dict) -> str:
    """Compact "same-topic:14, extends:3" form of the out-of-context edges."""
    if not counts:
        return ""
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:_MAX_RELATION_TYPES]
    return ", ".join(f"{rel}:{n}" for rel, n in top)


def _clip(text: str, limit: int) -> str:
    """Truncate `text` to `limit` chars on a word boundary (adds an ellipsis).

    Falls back to a hard cut when the word-boundary search would throw away most
    of the budget — a description padded with dots/hyphens or one long token has
    no space to break on, and `rsplit` used to return just the first word.
    """
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:. ")
    if len(cut) < limit * 0.5:
        cut = text[:limit].rstrip(",;:. ")
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
    # Chars of the stored `description` rendered per note. An evolved note's
    # description can reach ~1,700 chars; without a cap a single note would eat a
    # fifth of an 8k window. The cap is applied per note, after sentence splitting.
    fact_char_limit: int = _FACT_CHAR_LIMIT
    # Sentences of `description` rendered per note (the rest becomes
    # "[+k more fact(s) in this note, not shown]").
    max_fact_bullets: int = _MAX_FACT_BULLETS

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

    @property
    def effective_max_tokens(self) -> Optional[int]:
        """The completion cap this agent will actually request.

        ``answer.max_tokens`` wins; otherwise the backend's client-wide cap is
        used, which is what the request would send anyway. Reserving it in the
        prompt budget is what keeps ``prompt + completion`` inside the window.
        """
        return int(self.max_tokens) if self.max_tokens else default_output_cap(self.backend)

    def prompt_budget(self) -> Optional[int]:
        """Tokens this agent may spend on the prompt (None = no window declared)."""
        return resolve_budget(self.backend, self.context_window, self.max_tokens)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Cheap token estimate (~4 chars/token). See `asem.token_budget`."""
        return estimate_tokens(text)

    def _fit(self, candidates: List[Note], render) -> Tuple[str, List[Note]]:
        """Render a prompt that fits ``context_window`` minus ``max_tokens``.

        Drops the LOWEST-ranked candidate (last in retrieval order) until the
        prompt fits, always keeping at least one note, then shrinks the largest
        remaining one if a single note alone overflows the budget.
        """
        prompt, kept = fit_items(
            candidates,
            render,
            budget=self.prompt_budget(),
            min_keep=1,
            drop="tail",
            label="answer prompt",
        )
        return prompt, kept

    def _call(self, prompt: str, max_tokens: Optional[int]) -> str:
        """One backend call, passing a per-call output cap where supported."""
        return generate_with_cap(self.backend, prompt, max_tokens)

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
            configured = self.effective_max_tokens
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
            body = self.prompt_template.format(
                query=query,
                candidates=_render_notes_block(
                    notes,
                    content_chars=self.content_char_limit,
                    fact_chars=self.fact_char_limit,
                    max_bullets=self.max_fact_bullets,
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
        # `selected_ids` holds the bracketed RANKS printed in the context
        # ("[1]" ... "[8]") — a small model reproduces "3" far more reliably than
        # a 36-char UUID, and a mangled UUID used to silently degrade to "all
        # candidates". Plain ids are still accepted (older prompts, other
        # callers), so a UUID-shaped token is matched directly.
        resolved: set = set()
        for item in selected_ids:
            token = str(item).strip().strip("[]")
            if token.isdigit() and 1 <= int(token) <= len(candidates):
                resolved.add(candidates[int(token) - 1].id)
            else:
                resolved.add(token)
        selected_notes = [n for n in candidates if n.id in resolved]
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
        """Render notes as numbered evidence blocks, OLDEST first.

        Same grammar as the distil context (`_render_notes_block`) so both answer
        paths read the same way — the raw turn is the only field that needs
        clipping, since `description` is capped by `fact_char_limit`.

        Only the ORDER differs from the distil path: here it is chronological,
        which is what temporal questions need, while distil keeps the retriever's
        relevance order.
        """
        ordered = sorted(candidates, key=lambda n: n.t if n.t else datetime.min)
        return _render_notes_block(
            ordered,
            content_chars=self.content_char_limit,
            fact_chars=self.fact_char_limit,
            max_bullets=self.max_fact_bullets,
        )

    def _baseline_answer(self, query: str, candidates: List[Note]) -> str:
        context_items = []
        for note in candidates:
            date_prefix = f"[{_human_date(note)}] "
            context_items.append(f"- {date_prefix}{_clip(note.c, self.content_char_limit or 200)}")
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
