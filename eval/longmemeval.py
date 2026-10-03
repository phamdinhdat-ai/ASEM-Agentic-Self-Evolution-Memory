"""LongMemEval dataset adapter + recall/abstention helpers.

LongMemEval ships as a JSON list of *question* records, each carrying its own
retrieval haystack:

    {question_id, question_type, question, answer, question_date,
     haystack_dates[], haystack_session_ids[], haystack_sessions[[]],
     answer_session_ids[]}

A ``haystack_session`` is a flat list of ``{role, content, has_answer}``
messages (NOT LoCoMo's ``speaker``/``text`` dicts), and there is **no absolute
per-session datetime** — the date of session *i* is ``haystack_dates[i]``.
Everything is paired **by index**.

This module is a pure adapter: it converts a record into the
:class:`eval.phase_runner.Session` shape the existing ingestion dispatch already
consumes, and provides the recall/abstention stats the runner reports.  It
imports from the rest of the package but changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from asem.answer_agent import is_abstention
from eval.phase_runner import Session

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

LIST_FIELDS = (
    "haystack_dates",
    "haystack_session_ids",
    "haystack_sessions",
    "answer_session_ids",
)


@dataclass
class LongMemEvalItem:
    """One LongMemEval question + its retrieval haystack."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_dates: List[str] = field(default_factory=list)
    haystack_session_ids: List[str] = field(default_factory=list)
    haystack_sessions: List[List[Dict[str, Any]]] = field(default_factory=list)
    answer_session_ids: List[str] = field(default_factory=list)

    @property
    def is_abstention(self) -> bool:
        """Abstention questions are suffixed ``_abs`` in the official split."""
        return str(self.question_id).endswith("_abs")

    # Convenience -----------------------------------------------------------
    @property
    def gold_session_ids(self) -> List[str]:
        return list(self.answer_session_ids)

    def n_sessions(self) -> int:
        return len(self.haystack_sessions)


def load_longmemeval(path: str) -> List[LongMemEvalItem]:
    """Load an official LongMemEval ``.json`` into :class:`LongMemEvalItem`s."""
    import json

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path!r} is not a JSON list of LongMemEval records")

    items: List[LongMemEvalItem] = []
    for record in data:
        items.append(
            LongMemEvalItem(
                question_id=str(record.get("question_id", "")),
                question_type=str(record.get("question_type", "")),
                question=str(record.get("question", "")),
                answer=str(record.get("answer", "")),
                question_date=str(record.get("question_date", "")),
                haystack_dates=[str(x) for x in (record.get("haystack_dates") or [])],
                haystack_session_ids=[
                    str(x) for x in (record.get("haystack_session_ids") or [])
                ],
                haystack_sessions=[list(s) for s in (record.get("haystack_sessions") or [])],
                answer_session_ids=[
                    str(x) for x in (record.get("answer_session_ids") or [])
                ],
            )
        )
    return items


# ---------------------------------------------------------------------------
# Record -> Session conversion (reuses eval.phase_runner.Session verbatim)
# ---------------------------------------------------------------------------

def format_session_turns(session: Sequence[Dict[str, Any]]) -> List[str]:
    """Render one haystack session as ``[User] …`` / ``[Assistant] …`` turns."""
    turns: List[str] = []
    for message in session or []:
        role = str(message.get("role", "unknown")).strip().capitalize() or "Unknown"
        content = str(message.get("content", ""))
        turns.append(f"[{role}] {content}")
    return turns


def build_sessions(item: LongMemEvalItem) -> List[Session]:
    """Pair the three haystack arrays BY INDEX into ``Session`` objects.

    ``Session.num`` is 1-based so the ``sessions_to_fast`` ids (``s1``, ``s2``…)
    and the ``session_N`` batch labels match the LoCoMo path exactly.
    """
    sessions: List[Session] = []
    for i, raw in enumerate(item.haystack_sessions):
        date = item.haystack_dates[i] if i < len(item.haystack_dates) else ""
        turns = format_session_turns(raw)
        sessions.append(Session(num=i + 1, date=date, turns=turns))
    return sessions


def history_text(item: LongMemEvalItem) -> List[str]:
    """Flat history for the bank-less baselines (NoMemory / FullContext).

    Every haystack session is prefixed with an ``[Session N — date]`` header so
    dates survive into the transcript, then all turns are concatenated.
    """
    history: List[str] = []
    for session in build_sessions(item):
        header = f"[Session {session.num}"
        if session.date:
            header += f" — {session.date}"
        header += "]"
        history.append(header)
        history.extend(session.turns)
    return history


def to_eval_item(item: LongMemEvalItem) -> Dict[str, Any]:
    """Flatten to the dict shape the existing eval flow already understands."""
    sessions = build_sessions(item)
    return {
        "query": item.question,
        "raw_question": item.question,
        "answer": item.answer,
        "category_name": item.question_type,
        "category": item.question_type,
        "session_id": item.question_id,
        "history": history_text(item),
        "evidence": item.gold_session_ids,
        "question_date": item.question_date,
        "is_abstention": item.is_abstention,
        "n_sessions": item.n_sessions(),
        "_sessions": sessions,
        "_item": item,
    }


def group_by_session(items: Sequence[LongMemEvalItem]) -> List[List[Dict[str, Any]]]:
    """One group per question — each question owns its own haystack."""
    return [[to_eval_item(item)] for item in items]


def sessions_for_group(group: Sequence[Dict[str, Any]]) -> List[Session]:
    """``Session`` list for a group produced by :func:`group_by_session`."""
    if not group:
        return []
    return list(group[0].get("_sessions", []))


# ---------------------------------------------------------------------------
# Recall / abstention stats (pure functions, no architecture deps)
# ---------------------------------------------------------------------------

def session_recall(
    retrieved_ids: Sequence[str], gold_ids: Sequence[str]
) -> Tuple[float, float]:
    """Return ``(recall, hit)`` of gold evidence sessions covered by retrieval."""
    gold = {str(g) for g in gold_ids if str(g)}
    if not gold:
        return 0.0, 0.0
    retrieved = {str(r) for r in retrieved_ids if str(r)}
    covered = len(gold & retrieved)
    recall = covered / len(gold)
    return recall, (1.0 if covered else 0.0)


def turn_recall(
    retrieved_texts: Sequence[str], has_answer_texts: Sequence[str]
) -> float:
    """Fraction of ``has_answer`` turns whose text appears in the retrieved set.

    Matching is a substring test: a retrieved turn counts for a gold turn when
    either contains the other (the dense store may hold a note carved out of the
    turn rather than the verbatim turn).  Returns ``0.0`` when there is no gold.
    """
    gold = [str(t).strip() for t in has_answer_texts if str(t).strip()]
    if not gold:
        return 0.0
    haystack = " \u241f ".join(str(t) for t in retrieved_texts)
    covered = 0
    for g in gold:
        if g in haystack:
            covered += 1
            continue
        if any(g in str(r) or str(r) in g for r in retrieved_texts if str(r)):
            covered += 1
    return covered / len(gold)


def has_answer_turns(item: LongMemEvalItem) -> List[str]:
    """The raw text of every turn flagged ``has_answer`` across the haystack."""
    out: List[str] = []
    for session in item.haystack_sessions:
        for message in session or []:
            if message.get("has_answer"):
                out.append(str(message.get("content", "")))
    return out


def is_abstention_answer(text: str) -> bool:
    """Reuse the canonical ASEM refusal detector (import only, no code change)."""
    try:
        return bool(is_abstention(text))
    except Exception:  # noqa: BLE001 — never let a matcher abort a run
        return False


def note_ids_from_telemetry(used: Sequence[Any]) -> List[str]:
    """Normalise the ``system.last_used_notes`` payload into a list of ids.

    The telemetry contract is inconsistent across systems: ``ASEMTHGSystem``
    stores **note id strings**, while other systems (and any future one) may
    store the note objects themselves.  Both are accepted; unidentifiable
    entries are dropped rather than guessed at.
    """
    ids: List[str] = []
    for entry in used or []:
        if isinstance(entry, str):
            ids.append(entry)
            continue
        if isinstance(entry, dict):
            value = entry.get("id")
        else:
            value = getattr(entry, "id", None)
        if value:
            ids.append(str(value))
    return ids


def session_ids_from_telemetry(
    used: Sequence[Any], bank: Any
) -> List[str]:
    """Resolve ``last_used_notes`` telemetry into session ids via the bank.

    Note ids are opaque (uuid-ish), so the session attribution comes from the
    bank: ``bank.get_note(id).session_id``.  Systems whose builder ignores the
    ``db_dir`` (see :func:`eval.systems.build_system`) share one bank per
    process, so this is best-effort and returns ``[]`` when it cannot resolve.
    """
    ids = note_ids_from_telemetry(used)
    if not ids or bank is None:
        return []
    out: List[str] = []
    for note_id in ids:
        try:
            note = bank.get_note(note_id)
        except Exception:  # noqa: BLE001
            note = None
        sid = getattr(note, "session_id", None) if note is not None else None
        if sid:
            out.append(str(sid))
    return out


def normalize_session_id(session_id: str, item: LongMemEvalItem) -> str:
    """Map a note's ``session_id`` back onto the haystack id space.

    ``s1``/``session_1`` (1-based ordinals) become ``haystack_session_ids[0]``;
    ids already in the haystack pass through unchanged.
    """
    raw = str(session_id)
    if raw in item.haystack_session_ids:
        return raw
    ordinal = None
    if raw.startswith("session_"):
        ordinal = raw[len("session_"):]
    elif raw.startswith("s") and raw[1:].isdigit():
        ordinal = raw[1:]
    if ordinal is not None and ordinal.isdigit():
        idx = int(ordinal) - 1
        if 0 <= idx < len(item.haystack_session_ids):
            return item.haystack_session_ids[idx]
    return raw


__all__ = [
    "LIST_FIELDS",
    "LongMemEvalItem",
    "build_sessions",
    "format_session_turns",
    "group_by_session",
    "has_answer_turns",
    "history_text",
    "is_abstention_answer",
    "load_longmemeval",
    "note_ids_from_telemetry",
    "normalize_session_id",
    "session_ids_from_telemetry",
    "session_recall",
    "sessions_for_group",
    "to_eval_item",
    "turn_recall",
]
