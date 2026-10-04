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
    CHARS_PER_TOKEN as _CHARS_PER_TOKEN,
    default_output_cap,
    estimate_tokens,
    fit_items,
    fit_render_exact,
    fits,
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

# ---------------------------------------------------------------------------
# W10.3 — question-aware context
# ---------------------------------------------------------------------------
# The retrieval context used to be assembled identically for every question.
# But the analysis (`temporal_grounding_improvements.md` §5) shows the answer
# model's remaining failures are *question-class-specific*: date arithmetic
# fails on "loose relative-unit conversion" (`5.5 weeks` -> "five weeks and
# three and a half days"; `Two weeks` -> `21 days`), while ordering and
# knowledge-update questions need different structure again. P_temporal_qa
# guideline 7 already tells the model to restate both dates and subtract — but
# nothing in the CONTEXT said *which* memories are date-comparable, so the model
# had to re-find the operands by scanning every block.
#
# The fix is structural, not a second LLM call: classify the question from its
# surface form and render the three things that class needs to be legible:
#   * arithmetic      -> a `Date anchors:` line listing the distinct absolute
#                        dates present, chronological, so the operands are
#                        pre-extracted;
#   * ordering        -> an explicit ordinal position per block;
#   * knowledge-update-> the CURRENT fact first, with a `superseded` marker on
#                        the replaced one;
#   * multi-hop / attribution -> the bridge explains WHY two memories connect.
# Everything here is OFF unless `AnswerAgent.context_style == "thg_v1"`, so the
# default ("legacy") reproduces today's bytes exactly and every frozen-bank run
# stays reproducible.
_QUESTION_KINDS = (
    "arithmetic",
    "ordering",
    "knowledge_update",
    "count",
    "multi_hop",
    "single_fact",
)

# "How many days/weeks between A and B", "how long after", "how many days before".
_ARITHMETIC_RE = re.compile(
    r"\b(?:how\s+(?:many|much)\s+(?:days?|weeks?|months?|years?|hours?|minutes?)|"
    r"how\s+long\s+(?:after|before|between|until|since)|"
    r"how\s+(?:long|far\s+apart)|"
    r"number\s+of\s+(?:days?|weeks?|months?|years?)|"
    r"days?\s+(?:between|after|before|apart)|"
    r"time\s+(?:between|gap))\b",
    re.IGNORECASE,
)
# "in what order", "which came first", "before/after X", "earliest/latest".
_ORDERING_RE = re.compile(
    r"\b(?:in\s+what\s+order|which\s+(?:came|happened|was)\s+(?:first|last|earlier|later)|"
    r"what\s+order|chronologic|"
    r"first\s+(?:did|was|to|time)|"
    r"(?:earliest|latest|earlier|later|first|last|before|after|prior\s+to|preceded|"
    r"followed\s+by|most\s+recent))\b",
    re.IGNORECASE,
)
# The fact CHANGED: "current", "now", "used to", "before that", "update".
_KNOWLEDGE_UPDATE_RE = re.compile(
    r"\b(?:current(?:ly)?|nowadays|now\b|at\s+present|these\s+days|"
    # "used to" / "did she use to" / "she doesn't use to": allow the auxiliary
    # and a pronoun to sit between, which is how the question is actually asked.
    r"(?:did|do|does|didn'?t|doesn'?t)\s+(?:\w+\s+){0,2}use\s+to|"
    r"used\s+to|no\s+longer|formerly|previously|"
    r"(?:did|do|does|has|have|had)\s+\w+\s+(?:change|update|switch)|"
    r"still\b|anymore|any\s+more|"
    r"before\s+(?:that|this)|after\s+(?:that|this)|"
    r"update[d]?\b|latest\b|new(?:est)?\b|"
    r"what\s+is\s+\w+['\u2019]?s?\s+\w*\s*(?:now|currently))\b",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    r"\b(?:how\s+many|how\s+often|how\s+much\s+time|number\s+of|count\s+of)\b",
    re.IGNORECASE,
)
# Multi-hop: a chain cue, or attribution of one fact to another party.
_MULTI_HOP_RE = re.compile(
    r"\b(?:why\s+did|why\s+does|why\s+was|why\s+is|"
    r"what\s+(?:led|caused|made)|"
    r"who\s+(?:introduced|told|convinced|suggested|recommended|gave|showed)|"
    r"which\s+(?:session|conversation|time)|"
    r"what\s+did\s+\w+\s+say\s+(?:about|first)|"
    r"both|each\s+other|together|"
    r"connect(?:ed|ion)?|related\s+to|linked\s+to|"
    r"who\s+else|what\s+else)\b",
    re.IGNORECASE,
)
# A CAUSAL why/how question about a change: it needs the chain of events, so it
# is a multi-hop question even though it contains an update cue. Checked BEFORE
# the knowledge-update rule.
_MULTI_HOP_CAUSAL_RE = re.compile(
    r"\b(?:why\s+(?:did|does|was|is|has|have)|"
    r"what\s+(?:led|caused|made|prompted)\b|"
    r"how\s+did\s+\w+\s+(?:come\s+to|end\s+up|start|begin))\b",
    re.IGNORECASE,
)


def detect_question_kind(query: str) -> str:
    """Classify a question's *structure* from its surface form (no LLM call).

    Order matters and is deliberate:

    1. ``arithmetic`` — the model must subtract two dates; this is the class with
       the lowest measured accuracy, so it wins over everything else.
    2. ``ordering`` — needs chronological order + ordinal position.
    3. ``knowledge_update`` — needs the CURRENT fact singled out. Checked BEFORE
       ``count`` because "how many times has X changed" is really an update
       question, and before ``multi_hop`` because "why did X switch" is the
       update that the graph dated.
    4. ``count`` — sum values across notes.
    5. ``multi_hop`` — the answer lives in a chain between notes.
    6. ``single_fact`` — the default: one note holds the answer.

    Returns one of ``_QUESTION_KINDS``. Pure function of the query string, so it
    is trivially testable and cannot introduce nondeterminism into a run.
    """
    text = (query or "").strip()
    if not text:
        return "single_fact"
    if _ARITHMETIC_RE.search(text):
        return "arithmetic"
    if _ORDERING_RE.search(text):
        return "ordering"
    # A question asking WHY/HOW a change came about needs the CHAIN, even though
    # its wording contains an update cue ("switch careers"). Checked before
    # knowledge_update so the bridge is available; a plain "what is X now?" still
    # lands on knowledge_update.
    if _MULTI_HOP_CAUSAL_RE.search(text):
        return "multi_hop"
    if _KNOWLEDGE_UPDATE_RE.search(text):
        return "knowledge_update"
    if _COUNT_RE.search(text):
        return "count"
    if _MULTI_HOP_RE.search(text):
        return "multi_hop"
    return "single_fact"


def _human_date(note: Note) -> str:
    """Render a note's date as `20 April 2023` (fallback: `date unknown`).

    `session_date` is either an ISO timestamp ("2023-04-20T16:15:00Z") or a raw
    corpus date ("8 May 2023", `2023/02/20 (Mon) 12:15`). ISO timestamps are
    unreadable inside a prompt and invite the model to copy the time-of-day into
    a date answer, so both forms are normalised to day-month-year.

    An UNKNOWN date is reported as "date unknown" rather than being filled in
    from `note.t`. `parse_session_datetime` signals "this input was not a date"
    with a `None` ISO string while leaving `note.t` as a `now()` placeholder, so
    falling through to `note.t` would publish today's date as fact. That is the
    exact failure that made LongMemEval's date arithmetic unanswerable: every
    memory block claimed "recorded on 3 October 2026", giving the model a
    confident wrong anchor. Saying "unknown" keeps the model from doing
    arithmetic on a fabricated baseline.
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
        # A non-empty date string that parsed above but not here may still be a
        # corpus format handled by the central parser (e.g. the LongMemEval
        # slash form). Ask it directly so both modules agree on one contract.
        try:
            from .temporal import parse_session_datetime

            _dt, iso = parse_session_datetime(raw)
            if iso is None:
                return "date unknown"
            return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%d %B %Y").lstrip("0")
        except Exception:  # noqa: BLE001
            return "date unknown"

    # No date string at all: `t` is a real timestamp when the caller supplied
    # one, but a `now()` placeholder when nothing was supplied. Only trust it
    # when no `session_date` field was set AND `t` is not today-ish-by-default.
    if note.t is not None and note.session_date:
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


_NL_RELATION_PHRASE = {
    "same-entity": "shares the same entity",
    "temporal": "is from the same session as",
    "semantic": "is on a related topic to",
    "superseded_by": "replaces an earlier version of this fact in",
    "contradicts": "contradicts",
    "extends": "extends",
    "causal": "is causally linked to",
    "same-topic": "is on the same topic as",
    "linked": "is linked to",
}
_NL_RELATION_BULK = {
    "same-entity": "sharing the same entity",
    "temporal": "from the same session",
    "semantic": "on related topics",
    "superseded_by": "earlier versions of this fact",
    "contradicts": "contradicting",
    "extends": "that extend this one",
    "causal": "causally linked",
    "same-topic": "on the same topic",
    "linked": "linked",
}


def _join_ranks(ranks: List[int]) -> str:
    labels = [f"Memory {r}" for r in ranks]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _validity_phrase(note: Note, index_by_id: dict) -> str:
    """Describe when a note's fact was true, if a supersession dates it.

    Returns ``""`` unless a ``superseded_by`` link carries an interval, so the
    common case (and every pre-W4 bank) renders exactly as before.

    Two directions matter and both are phrased from THIS note's point of view:

    * this note is the OLDER one (its link carries ``valid_to``) -> the fact is
      historical: "true until 3 March 2022, replaced by Memory 4";
    * this note is the NEWER one (``valid_from`` set) -> "true from 3 March 2022".

    The reader needs the difference: without it a superseded fact and its
    replacement look equally current, which is how "the notes contradict each
    other" answers are produced.
    """
    older_until: Optional[str] = None
    newer_from: Optional[str] = None
    replaced_by: Optional[int] = None

    for link in note.L:
        if link.relation != "superseded_by" or not link.has_interval:
            continue
        if link.valid_to:
            older_until = link.valid_to
            replaced_by = index_by_id.get(link.target_id)
        if link.valid_from:
            newer_from = link.valid_from

    if older_until:
        phrase = f"true until {_iso_to_day(older_until)}, superseded"
        if replaced_by:
            phrase += f" by Memory {replaced_by}"
        return phrase
    if newer_from:
        return f"true from {_iso_to_day(newer_from)}"
    return ""


def _iso_to_day(value: str) -> str:
    """`2022-03-03T09:00:00Z` -> `3 March 2022`; returns the input on failure."""
    try:
        return (
            datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
            .strftime("%d %B %Y")
            .lstrip("0")
        )
    except (ValueError, AttributeError):
        return str(value)


# ---------------------------------------------------------------------------
# W10.1 — session provenance and the graph-as-a-graph
# ---------------------------------------------------------------------------
# The block header used to be `Memory {rank} — recorded on {date}, said by
# {speaker}, about {entities}.` — `session_id` was rendered NOWHERE, and since
# `content_chars` clips `note.c` to a 200-char PREFIX of the turn, the
# `[Session N]` marker LoCoMo puts on turn 4+ is usually not even inside the
# rendered text. So a 6-session haystack collapsed into ~8 indistinguishable
# blocks, and "which session did X happen in" / "what did I say first" / "who
# said what" were unanswerable *from the context* — the model cannot cite a
# source it cannot see.
#
# `session_id` is the ONLY session provenance a THG note carries, and it is
# usually already a short ordinal-ish label at ingest (`session_3`,
# `session_3 — 8 May 2023`, `locomo_0000`). `_session_label` therefore keeps the
# id when it is short and reduces it to its ordinal when it is long, so
# provenance costs a few chars per block instead of blowing the budget.
_MAX_SESSION_LABEL_CHARS = 32
# Any run of digits in the id is candidate ordinal material (`locomo_0042`,
# `conv-26/session_7`). Anchoring on `$` would miss every id that ends in a word.
_SESSION_ORDINAL_RE = re.compile(r"(\d+)")


def _session_label(note: Note) -> str:
    """Short, human-readable session provenance for a note, or ``""``.

    Preference order:

    1. the trailing ordinal of `session_id` when the id is long
       (`locomo_0042` -> `session 42`) — measured cheaper than the full id and
       carries the same ordering information;
    2. the id verbatim when it is already short;
    3. ``""`` when there is no id, so a bank that predates session ids renders
       exactly as it did before.
    """
    raw = str(note.session_id or "").strip()
    if not raw:
        return ""
    if len(raw) <= _MAX_SESSION_LABEL_CHARS:
        return raw
    # A long id: keep only its LAST digit run, which is the ordinal part for
    # every id shape the loaders produce (`locomo_0042`, `conv-26_session_7`).
    runs = _SESSION_ORDINAL_RE.findall(raw)
    if runs:
        return f"session {int(runs[-1])}"
    return ""


def _entity_set(note: Note) -> set:
    return {str(e).strip().lower() for e in (note.entities or []) if str(e).strip()}


def _render_chains(notes: List[Note], index_by_id: dict) -> List[str]:
    """Render the multi-hop BRIDGE between notes in this block (W10.1 / P2-1).

    A multi-hop question needs the model to walk from one memory to another, and
    on a list/inference question (roadmap P2-1) the answer exists only in the
    *union* of several notes. The `Connections:` line already says two notes are
    related, but not *through what*, so the model has no bridge to walk.

    This renders the bridge explicitly::

        Memory 1 → Memory 3 via Caroline (shares the same entity);

    Computed ONLY from persisted fields — `note.L` (the typed link set) and
    `note.entities` — so it needs no live `TemporalHyperGraph` object. That
    matters because a bank can be loaded without one (the graph is not
    persisted); the whole section degrades silently to nothing when `note.L` is
    empty, rather than emitting a placeholder the model might treat as a fact.

    A bridge entity is the intersection of the two notes' entity sets when there
    is one; otherwise the relation label alone carries the meaning, because an
    empty intersection still tells the model the notes are linked.
    """
    chains: List[str] = []
    seen: set = set()
    for rank, note in enumerate(notes, start=1):
        for link in note.L:
            other = index_by_id.get(link.target_id)
            if other is None or other == rank:
                continue
            # One bridge per unordered pair, so the connection is not stated
            # twice (once from each endpoint) and the prompt does not pay twice.
            pair = (min(rank, other), max(rank, other))
            if pair in seen:
                continue
            other_note = notes[other - 1]
            shared = _entity_set(note) & _entity_set(other_note)
            via = f" via {sorted(shared)[0].title()}" if shared else ""
            relation = link.relation or "linked"
            phrase = _NL_CHAIN_PHRASE.get(relation, f"({relation})")
            seen.add(pair)
            chains.append(f"Memory {rank} → Memory {other}{via} {phrase}")
    return chains


_NL_CHAIN_PHRASE = {
    "same-entity": "(shares the same entity)",
    "temporal": "(from the same session)",
    "semantic": "(related topic)",
    "superseded_by": "(updated fact)",
    "contradicts": "(contradicting)",
    "extends": "(extends)",
    "causal": "(causally linked)",
    "same-topic": "(same topic)",
    "linked": "(linked)",
}


def _date_anchor(note: Note) -> str:
    """The note's absolute date for the `Date anchors:` line, or ``""``.

    `date unknown` is deliberately excluded: the whole point of the anchor line
    is to hand the arithmetic over pre-extracted, and publishing "unknown" as an
    operand invites the model to subtract from nothing.
    """
    rendered = _human_date(note)
    return "" if rendered == "date unknown" else rendered


@dataclass
class RenderStats:
    """What the renderer actually dropped, per note (W10.2 bookkeeping).

    The old renderer reported a *sentence* count in the block header but threw
    away the count of what the block-header machinery and the char cap actually
    removed, so a run could not state how much evidence it had silently cut. A
    benchmark report that cannot say "the median note lost 2 of 4 sentences" is
    not measuring its own context.
    """

    notes_rendered: int = 0
    sentences_total: int = 0
    sentences_shown: int = 0
    dropped_sentences: int = 0
    dropped_chars: int = 0
    chains_rendered: int = 0
    sessions_labelled: int = 0
    knowledge_updates_ordered: int = 0

    def as_dict(self) -> dict:
        return {
            "notes_rendered": self.notes_rendered,
            "sentences_total": self.sentences_total,
            "sentences_shown": self.sentences_shown,
            "dropped_sentences": self.dropped_sentences,
            "dropped_chars": self.dropped_chars,
            "chains_rendered": self.chains_rendered,
            "sessions_labelled": self.sessions_labelled,
            "knowledge_updates_ordered": self.knowledge_updates_ordered,
        }


def _render_notes_block(
    notes: List[Note],
    *,
    content_chars: int = _CONTENT_CHAR_LIMIT,
    fact_chars: int = _FACT_CHAR_LIMIT,
    max_bullets: int = _MAX_FACT_BULLETS,
    context_style: str = "legacy",
    question_kind: str = "single_fact",
    stats: Optional[RenderStats] = None,
) -> str:
    """Render the retrieved notes as numbered, NATURAL-LANGUAGE evidence blocks.

    ``context_style="legacy"`` (the default) reproduces the historical rendering
    BYTE-FOR-BYTE — it is the contract that keeps every frozen-bank run and every
    published number reproducible, and `tests/test_answer_context.py` pins the
    exact string. ``context_style="thg_v1"`` adds the three W10 fixes:

    * **W10.1 provenance + graph-as-graph** — the block header carries the session
      label, and a `Evidence chains:` section renders the multi-hop bridge
      (`Memory 1 → Memory 3 via Caroline (shares the same entity)`).
    * **W10.2 dropped-content counters** — `stats` receives per-render counts of
      the sentences and chars that did not make it into the block.
    * **W10.3 question-aware structure** — driven by ``question_kind``: a
      `Date anchors:` line for arithmetic questions, ordinal positions for
      ordering questions, and current-fact-first ordering with a `superseded`
      marker for knowledge-update questions.

    Every note becomes (legacy):

        Memory 1 — recorded on 8 May 2023, said by Caroline, about Caroline.
              • On 7 May 2023, Caroline attended an LGBTQ support group.
              Themes: LGBTQ, support group, attendance.
              Connections: shares the same entity (Caroline) with Memory 2 and
              Memory 3; is on a related topic to Memory 4.
              Also connected, not shown here: 28 sharing the same entity, 15 on
              related topics.

    Relationships are phrased in plain English ("shares the same entity",
    "is from the same session as", "replaces an earlier version") instead of
    raw labels, so the model can reason about *why* two memories connect.
    Relation targets are printed as the rank of the other note in this block.
    """
    if not notes:
        return "(no memory notes retrieved)"

    thg = str(context_style or "legacy").lower() == "thg_v1"
    index_by_id = {n.id: i + 1 for i, n in enumerate(notes)}
    stat = stats if stats is not None else RenderStats()

    # W10.3 (knowledge_update): surface the CURRENT fact first. "Current" is read
    # off the bi-temporal interval — a note whose supersession closed its
    # validity (`true until …`) is the obsolete version, and its replacement is
    # whatever it points at. Ordering the replacement first means the model reads
    # the answer before the thing it replaced, which is the order the question
    # ("what does X do now?") is actually phrased in.
    ordered_notes = notes
    if thg and question_kind == "knowledge_update":
        superseded_ranks = {
            i
            for i, n in enumerate(notes)
            if any(
                link.relation == "superseded_by" and link.valid_to
                for link in (n.L or [])
            )
        }
        if superseded_ranks and len(superseded_ranks) < len(notes):
            current = [n for i, n in enumerate(notes) if i not in superseded_ranks]
            obsolete = [n for i, n in enumerate(notes) if i in superseded_ranks]
            ordered_notes = current + obsolete
            stat.knowledge_updates_ordered += len(obsolete)
            # Ranks are positional, so the index must be rebuilt after reordering
            # or every `Connections:` target would point at the wrong memory.
            index_by_id = {n.id: i + 1 for i, n in enumerate(ordered_notes)}

    blocks: List[str] = []

    for rank, note in enumerate(ordered_notes, start=1):
        entities = [e for e in (note.entities or []) if e]
        header = f"Memory {rank} — recorded on {_human_date(note)}"
        if thg:
            # W10.1: session provenance. Without it a 6-session haystack is ~8
            # indistinguishable blocks and "which session" is unanswerable.
            label = _session_label(note)
            if label:
                header += f" [session: {label}]"
                stat.sessions_labelled += 1
            if question_kind == "ordering":
                # W10.3 (ordering): the ordinal makes position explicit instead of
                # leaving the model to infer it from the print order.
                header += f" (position {rank} of {len(ordered_notes)})"
        # Bi-temporal validity: when the fact was TRUE, which is not the same as
        # when it was recorded. Rendered only when a supersession actually dates
        # the interval, so legacy notes (and every frozen bank) render exactly as
        # before. This is what lets the model answer "which is current?" instead
        # of treating an obsolete fact and its replacement as equally valid.
        interval = _validity_phrase(note, index_by_id)
        if interval:
            header += f" ({interval})"
        if note.speaker:
            header += f", said by {note.speaker}"
        if entities:
            header += f", about {', '.join(entities[:6])}"
        header += "."
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
            lines.append(f"{_INDENT}…and {hidden} more fact(s) in this memory (not shown).")

        # W10.2: account for what was actually dropped, so a run can report its
        # own context loss instead of only hinting at it in the block header.
        stat.notes_rendered += 1
        stat.sentences_total += len(facts)
        stat.sentences_shown += len(shown)
        stat.dropped_sentences += max(0, len(facts) - len(shown))
        stat.dropped_chars += max(0, len(note.X or "") - used)

        # --- the raw turn, only when it adds words the facts do not have ------
        # NOTE (`rstrip` set semantics, left as-is on purpose): `str.rstrip(" …")`
        # strips ANY of those characters, not the two-character suffix, so it also
        # removes a trailing '.' or space from `raw` before the containment test.
        # That makes the dedup test MORE permissive, never less — it can only
        # suppress the "Original wording" line, never emit a wrong one — so it is
        # a latent permissiveness, not a rendering defect, and changing it would
        # move the legacy bytes that frozen-bank runs are pinned to.
        if content_chars > 0:
            raw = _clip(note.c, content_chars)
            if raw and raw.rstrip(" …") not in (note.X or ""):
                lines.append(f"{_INDENT}Original wording: \"{_wrap(raw, indent=_INDENT + ' ' * 18)}\"")

        # --- themes: keywords are retrieval metadata, capped hard -------------
        topics = [k for k in (note.K or []) if k][:_MAX_TOPICS]
        if topics:
            lines.append(f"{_INDENT}Themes: {', '.join(topics)}.")

        # --- connections: phrased as plain-English relationships --------------
        in_block: dict = {}
        out_block: dict = {}
        for link in note.L:
            rel = link.relation or "linked"
            target = index_by_id.get(link.target_id)
            if target is not None:
                bucket = in_block.setdefault(rel, [])
                if len(bucket) < _MAX_RELATIONS_PER_NOTE:
                    bucket.append(target)
            else:
                out_block[rel] = out_block.get(rel, 0) + 1

        _REL_PRIORITY = ("superseded_by", "temporal", "same-entity", "semantic",
                         "same-topic", "extends", "contradicts", "causal")
        ordered = sorted(
            in_block.items(),
            key=lambda kv: _REL_PRIORITY.index(kv[0]) if kv[0] in _REL_PRIORITY else len(_REL_PRIORITY),
        )
        sentences: List[str] = []
        covered: List[set] = []
        for rel, targets in ordered:
            tset = set(targets)
            # Skip a relation whose targets are already all covered by a stronger
            # one (same-entity and semantic almost always overlap), so the model
            # is not told the same connection twice.
            if any(tset <= seen for seen in covered):
                continue
            covered.append(tset)
            ranks = _join_ranks(sorted(tset))
            if rel == "same-entity":
                ent = f" ({entities[0]})" if entities else ""
                sentences.append(f"shares the same entity{ent} with {ranks}")
            elif rel == "superseded_by":
                # `superseded_by` is stored symmetrically, so the phrase must be
                # chosen per-note or BOTH endpoints claim to be "the replacement"
                # (which is how a versioned pair reads as a contradiction).
                # Direction comes from the interval: an edge where THIS note is
                # the one whose validity ended (valid_to set) means this note is
                # the older, replaced fact. An edge where valid_to is empty and
                # valid_from is set marks this note as the newer replacement.
                older_ranks = {
                    index_by_id[link.target_id]
                    for link in note.L
                    if link.relation == "superseded_by"
                    and link.valid_to
                    and link.target_id in index_by_id
                }
                self_is_older = bool(older_ranks & tset)
                if self_is_older:
                    sentences.append(f"is replaced by a later version of this fact in {ranks}")
                else:
                    sentences.append(f"replaces an earlier version of this fact in {ranks}")
            else:
                phrase = _NL_RELATION_PHRASE.get(rel, f"is linked by '{rel}' to")
                sentences.append(f"{phrase} {ranks}")
        if sentences:
            lines.append(f"{_INDENT}Connections: " + "; ".join(sentences) + ".")

        bulk = []
        for rel, cnt in sorted(out_block.items(), key=lambda kv: (-kv[1], kv[0]))[:_MAX_RELATION_TYPES]:
            if rel == "same-entity":
                bulk.append(f"{cnt} other {'memory shares' if cnt == 1 else 'memories share'} the same entity")
            elif rel == "semantic":
                bulk.append(f"{cnt} {'is on a related topic' if cnt == 1 else 'are on related topics'}")
            elif rel == "temporal":
                bulk.append(f"{cnt} {'is from the same session' if cnt == 1 else 'are from the same session'}")
            elif rel == "superseded_by":
                bulk.append(f"{cnt} {'is an earlier version' if cnt == 1 else 'are earlier versions'} of this fact")
            else:
                bulk.append(f"{cnt} {'is' if cnt == 1 else 'are'} linked by '{rel}'")
        if bulk:
            lines.append(f"{_INDENT}Also connected, not shown here: " + "; ".join(bulk) + ".")

        blocks.append("\n".join(lines))

    body = "\n\n".join(blocks)

    if not thg:
        return body

    # --- W10.1: the graph, as a graph ---------------------------------------
    # Rendered AFTER the blocks so the model reads the evidence first, then the
    # bridges. Degrades silently to nothing when note.L is empty (a bank loaded
    # without a graph object), never to a placeholder the model could read as a
    # fact.
    chains = _render_chains(ordered_notes, index_by_id)
    if chains:
        stat.chains_rendered += len(chains)
        body += "\n\nEvidence chains: " + "; ".join(chains) + "."

    # --- W10.3: pre-extracted operands for date arithmetic ------------------
    # Only the DISTINCT absolute dates, chronological. The analysis measured the
    # remaining arithmetic failures as operand-hunting and loose unit conversion;
    # listing the operands removes the hunting without a second LLM call.
    if question_kind == "arithmetic":
        anchors: List[str] = []
        for note in ordered_notes:
            anchor = _date_anchor(note)
            if anchor and anchor not in anchors:
                anchors.append(anchor)
        if len(anchors) >= 2:
            anchors.sort(key=_date_sort_key)
            body += "\n\nDate anchors (distinct absolute dates in the evidence, earliest first): " + "; ".join(anchors) + "."

    return body


def _date_sort_key(value: str):
    """Chronological sort key for an already-rendered `8 May 2023` string."""
    try:
        return datetime.strptime(value, "%d %B %Y")
    except (ValueError, TypeError):
        return datetime.max


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
    # Sampling temperature for answering. None = inherit the backend's config
    # value. This exists because `answer.temperature` in the YAML was previously
    # read by NOTHING: every eval run silently sampled at the *inference* block's
    # temperature (0.3), so the config documented a determinism it did not have
    # and identical questions scored differently between runs.
    temperature: Optional[float] = None
    # W10 — retrieval-context style. "legacy" (the default) reproduces the
    # historical rendering byte-for-byte, which is what keeps every frozen-bank
    # result and every published number reproducible. "thg_v1" enables the
    # session-provenance header, the evidence-chain section, the question-aware
    # structure (date anchors / ordinals / current-fact-first) and the render
    # counters. One flag, so the two arms differ in exactly one place.
    context_style: str = "legacy"

    # Populated by the most recent render (`distil_and_answer` / `direct_answer`)
    # so a benchmark run can report what the builder actually dropped. Reset per
    # call, never shared across questions.
    last_render_stats: Optional[RenderStats] = None
    last_question_kind: Optional[str] = None
    # Ids of the notes that ACTUALLY reached the prompt after `_select` +
    # `_fit`. This is distinct from the retriever's candidate list: `_fit`
    # drops lowest-ranked notes until the prompt fits `context_window`, so a
    # candidate can be retrieved and still never be shown to the model. The
    # evidence-recall instrument (`analysis/evidence_recall.py`) needs the
    # post-fit set to measure context recall rather than candidate recall.
    last_rendered_note_ids: Optional[List[str]] = None
    # Estimated tokens of the prompt the model was actually sent, and of the
    # completion it returned. Together with ``retrieval_sec`` (recorded by the
    # pipeline) this is the end-to-end efficiency axis: what a question cost in
    # context and latency, measured at the call site rather than inferred from
    # the context builder. ``estimate_tokens`` is the same conservative
    # ~4-chars/token estimate the budget arithmetic uses, so the reported number
    # is the one the trimming logic itself believed.
    last_prompt_tokens: Optional[int] = None
    last_answer_tokens: Optional[int] = None

    @property
    def _thg_context(self) -> bool:
        return str(self.context_style or "legacy").lower() == "thg_v1"

    def _render_with_stats(
        self, notes: List[Note], *, query: str = "", chronological: bool = False
    ) -> str:
        """Render ``notes`` in the configured style, recording drop counters.

        Single entry point for both answer paths so the provenance / question-kind
        behaviour cannot drift between `distil_and_answer` and `direct_answer`.
        """
        kind = detect_question_kind(query) if self._thg_context else "single_fact"
        stats = RenderStats()
        rendered = _render_notes_block(
            notes,
            content_chars=self.content_char_limit,
            fact_chars=self.fact_char_limit,
            max_bullets=self.max_fact_bullets,
            context_style=self.context_style,
            question_kind=kind,
            stats=stats,
        )
        self.last_render_stats = stats
        self.last_question_kind = kind
        return rendered

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

        The result is a **hard contract** (W10.2): the returned prompt never
        exceeds the budget it was fitted to. ``fit_items`` alone cannot promise
        that — when the budget is smaller than the fixed prompt scaffolding it
        logs a warning and returns an over-budget prompt, which is a hard HTTP
        400 on an OpenAI-compatible endpoint and therefore a lost benchmark row.
        ``fit_render_exact`` closes that hole by truncating the rendered string
        as a last resort.
        """
        budget = self.prompt_budget()
        prompt, kept = fit_items(
            candidates,
            render,
            budget=budget,
            min_keep=1,
            drop="tail",
            label="answer prompt",
        )
        if not fits(prompt, budget):
            prompt, kept = fit_render_exact(
                candidates,
                render,
                budget=budget,
                # The no-context render: the prompt scaffolding the caller cannot
                # trim any further.
                fallback=render([]),
            )
        return prompt, kept

    def _call(self, prompt: str, max_tokens: Optional[int]) -> str:
        """One backend call, passing a per-call output cap where supported."""
        return generate_with_cap(
            self.backend, prompt, max_tokens, temperature=self.temperature
        )

    def _generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate, with a last-resort retry if the endpoint rejects the prompt length.

        The character-based estimate can undercount, and callers other than this
        class (e.g. an LLMRetryHandler) may not pass a cap at all. Rather than
        failing the whole benchmark run on an avoidable HTTP 400, shrink the
        output budget so that prompt + output fits the declared window.

        Every answer path funnels through here, so this is also where the
        per-query efficiency telemetry is recorded: the prompt length the model
        actually received and the length of what came back. When a retry handler
        calls this more than once for one question, the LAST call wins — which is
        the one whose answer is used.
        """
        # Recorded before the call so an exception mid-generation still leaves the
        # prompt size observable (the prompt was still sent, or attempted).
        self.last_prompt_tokens = self._estimate_tokens(prompt)
        try:
            answer = self._call(prompt, max_tokens)
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
            answer = self._call(prompt, new_cap)
        self.last_answer_tokens = self._estimate_tokens(answer)
        return answer

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
                candidates=self._render_with_stats(notes, query=query),
            )
            # Second-chance pass: the previous answer was a refusal, so say so
            # and make the model re-read a wider candidate set.
            return f"{notice}\n\n{body}" if notice else body

        prompt, candidates = self._fit(self._select(candidates), render)
        self.last_rendered_note_ids = [n.id for n in candidates]
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

        prompt, kept = self._fit(
            self._select(candidates),
            lambda notes: self.baseline_prompt_template.format(
                query=query,
                context=self._render_with_stats(notes, query=query, chronological=True),
            ),
        )
        self.last_rendered_note_ids = [n.id for n in kept]
        return self._generate_resilient(prompt, max_tokens=self.max_tokens).strip()

    def _render_graph_context(self, candidates: List[Note], query: str = "") -> str:
        """Render notes as numbered evidence blocks, OLDEST first.

        Same grammar as the distil context (`_render_notes_block`) so both answer
        paths read the same way — the raw turn is the only field that needs
        clipping, since `description` is capped by `fact_char_limit`.

        Only the ORDER differs from the distil path: here it is chronological,
        which is what temporal questions need, while distil keeps the retriever's
        relevance order.
        """
        ordered = sorted(candidates, key=lambda n: n.t if n.t else datetime.min)
        return self._render_with_stats(ordered, query=query, chronological=True)

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
