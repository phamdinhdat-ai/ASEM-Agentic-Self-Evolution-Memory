"""Source-grounded temporal claims and conservative relation checks.

This is an opt-in prototype. It never treats a later utterance as an update by
itself, and an unmatched source is UNKNOWN rather than evidence of falsity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from typing import Literal

from .sourcefirst import SourceTurn


@dataclass(frozen=True)
class TimeSpan:
    """Half-open interval [start, end); either bound may be unknown."""

    start: date | None = None
    end: date | None = None

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("TimeSpan end must be later than start")

    def contains(self, point: date) -> bool:
        return (self.start is not None or self.end is not None) and (
            self.start is None or self.start <= point) and (
            self.end is None or point < self.end)

    def overlaps(self, other: "TimeSpan") -> bool:
        if (self.start is None and self.end is None) or (other.start is None and other.end is None):
            return False
        return not ((self.end is not None and other.start is not None and self.end <= other.start)
                    or (other.end is not None and self.start is not None and other.end <= self.start))

    def before(self, other: "TimeSpan") -> bool:
        return self.end is not None and other.start is not None and self.end <= other.start


@dataclass(frozen=True)
class TemporalClaim:
    claim_id: str
    source_id: str
    subject: str
    relation: str
    value: str
    quote: str
    said_at: date | None = None
    recorded_at: datetime | None = None
    occurred: TimeSpan | None = None
    valid: TimeSpan | None = None
    temporal_label: str | None = None
    modality: Literal["asserted", "planned", "hypothetical"] = "asserted"
    explicit_update: bool = False
    functional: bool = False


@dataclass(frozen=True)
class TemporalEdge:
    source: str
    relation: Literal["before", "after", "overlaps", "supersedes", "conflicts"]
    target: str


@dataclass(frozen=True)
class AsOfResult:
    status: Literal["supported", "conflict", "unknown"]
    claims: tuple[TemporalClaim, ...]


@dataclass(frozen=True)
class GuardResult:
    status: Literal["supported", "incompatible", "unknown"]
    reason: str | None = None
    source_ids: tuple[str, ...] = ()


class TemporalRelationGraph:
    """Add-only claim graph; source quotes are checked against immutable turns."""

    def __init__(self, turns: list[SourceTurn]):
        self.turns = {turn.source_id: turn for turn in turns}
        self.claims: dict[str, TemporalClaim] = {}
        self.edges: list[TemporalEdge] = []

    def add(self, claim: TemporalClaim) -> None:
        turn = self.turns.get(claim.source_id)
        if turn is None or claim.claim_id in self.claims or not claim.quote or claim.quote not in turn.text:
            raise ValueError("Claim must have a unique ID and an exact raw-source quote")
        for older in self.claims.values():
            if claim.occurred and older.occurred:
                if older.occurred.before(claim.occurred):
                    self.edges.append(TemporalEdge(older.claim_id, "before", claim.claim_id))
                elif claim.occurred.before(older.occurred):
                    self.edges.append(TemporalEdge(claim.claim_id, "before", older.claim_id))
                elif claim.occurred.overlaps(older.occurred):
                    self.edges.append(TemporalEdge(claim.claim_id, "overlaps", older.claim_id))
            if (claim.subject != older.subject or claim.relation != older.relation
                    or claim.value == older.value):
                continue
            later_said = (claim.said_at is not None and older.said_at is not None
                          and claim.said_at > older.said_at)
            earlier_said = (claim.said_at is not None and older.said_at is not None
                            and older.said_at > claim.said_at)
            if claim.explicit_update and later_said:
                self.edges.append(TemporalEdge(claim.claim_id, "supersedes", older.claim_id))
            elif older.explicit_update and earlier_said:
                self.edges.append(TemporalEdge(older.claim_id, "supersedes", claim.claim_id))
            elif (claim.functional and older.functional and claim.valid and older.valid
                  and claim.valid.overlaps(older.valid)):
                self.edges.append(TemporalEdge(claim.claim_id, "conflicts", older.claim_id))
        self.claims[claim.claim_id] = claim

    def as_of(self, subject: str, relation: str, point: date,
              *, known_at: datetime | None = None) -> AsOfResult:
        eligible = {claim.claim_id: claim for claim in self.claims.values()
                    if claim.subject.casefold() == subject.casefold()
                    and claim.relation == relation and claim.valid is not None
                    and claim.valid.contains(point)
                    and (known_at is None or (claim.recorded_at is not None
                                              and claim.recorded_at <= known_at))}
        for edge in self.edges:
            if edge.relation == "supersedes" and edge.source in eligible and edge.target in eligible:
                eligible.pop(edge.target)
        selected = tuple(eligible.values())
        if not selected:
            return AsOfResult("unknown", ())
        if len({claim.value.casefold() for claim in selected}) > 1 and all(
                claim.functional for claim in selected):
            return AsOfResult("conflict", selected)
        return AsOfResult("supported", selected)


_SESSION_DATE = re.compile(r"\b(\d{1,2}) ([A-Za-z]+),? (\d{4})\b")
_MONTHS = {name.casefold(): number for number, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"), 1)}
_HOLIDAYS = ("halloween", "thanksgiving", "christmas", "easter", "new year")
_PREFERENCE = re.compile(r"\bmy\s+(least\s+)?favorite\s+character\s+is\s+([^,.!?]+)", re.I)
_ACTIVITY = re.compile(r"\bI\s+(started|began|quit|stopped|gave up)\s+(.+?)(?=\s+(?:one year ago|last year|yesterday|in \w+ \d{4})\b|[.!?]|$)", re.I)
_RELATIVE = re.compile(r"\b(one year ago|last year|yesterday)\b", re.I)


def session_date(text: str) -> date | None:
    match = _SESSION_DATE.search(text)
    if not match:
        return None
    month = _MONTHS.get(match.group(2).casefold())
    if month is None:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(1)))
    except ValueError:
        return None


def event_time(phrase: str, said_at: date | None) -> TimeSpan | None:
    if said_at is None:
        return None
    phrase = phrase.casefold()
    if phrase == "yesterday":
        day = said_at - timedelta(days=1)
        return TimeSpan(day, day + timedelta(days=1))
    if phrase == "last year":
        return TimeSpan(date(said_at.year - 1, 1, 1), date(said_at.year, 1, 1))
    if phrase == "one year ago":
        try:
            day = said_at.replace(year=said_at.year - 1)
        except ValueError:  # 29 February -> 28 February in the previous year.
            day = said_at.replace(year=said_at.year - 1, day=28)
        return TimeSpan(day, day + timedelta(days=1))
    return None


def extract_high_confidence(turn: SourceTurn) -> list[TemporalClaim]:
    """Extract only narrow patterns; other language remains unparsed."""
    claims: list[TemporalClaim] = []
    said_at = session_date(turn.date)
    for number, match in enumerate(_PREFERENCE.finditer(turn.text)):
        degree = "least_favorite_character" if match.group(1) else "favorite_character"
        claims.append(TemporalClaim(f"{turn.source_id}:preference:{number}", turn.source_id,
            turn.speaker, degree, match.group(2).strip(), match.group(0), said_at,
            functional=True))
    for number, match in enumerate(_ACTIVITY.finditer(turn.text)):
        verb = match.group(1).casefold()
        relation = "start_activity" if verb in {"started", "began"} else "end_activity"
        following = turn.text[match.end():]
        relative = _RELATIVE.search(following[:40])
        occurred = event_time(relative.group(1), said_at) if relative else None
        claims.append(TemporalClaim(f"{turn.source_id}:activity:{number}", turn.source_id,
            turn.speaker, relation, match.group(2).strip(), match.group(0), said_at,
            occurred=occurred))
    for number, sentence in enumerate(re.findall(r"[^.!?;]+[.!?;]?", turn.text)):
        for holiday in _HOLIDAYS:
            if re.search(r"\bduring\s+" + re.escape(holiday) + r"\b", sentence, re.I):
                quote = sentence.strip()
                claims.append(TemporalClaim(f"{turn.source_id}:holiday:{number}:{holiday}",
                    turn.source_id, turn.speaker, "holiday_activity", quote, quote, said_at,
                    temporal_label=holiday))
    return claims


class TemporalProofGuard:
    """Reject only explicit relation/time mismatches; abstain from other judgments."""

    def check(self, question: str, answer: str, cited_turns: list[SourceTurn]) -> GuardResult:
        if not cited_turns or not answer.strip():
            return GuardResult("unknown")
        graph = TemporalRelationGraph(cited_turns)
        for turn in cited_turns:
            for claim in extract_high_confidence(turn):
                graph.add(claim)
        q = question.casefold()
        answer_norm = re.sub(r"[^\w]+", " ", answer.casefold()).strip()
        def phrase_in(phrase: str, text: str) -> bool:
            needle = re.sub(r"[^\w]+", " ", phrase.casefold()).strip()
            haystack = re.sub(r"[^\w]+", " ", text.casefold()).strip()
            return bool(needle and re.search(r"(?:^| )" + re.escape(needle)
                                             + r"(?: |$)", haystack))
        recognized = []
        for holiday in _HOLIDAYS:
            if re.search(r"\b" + re.escape(holiday) + r"\b", q):
                for claim in graph.claims.values():
                    if claim.relation == "holiday_activity" and phrase_in(answer, claim.value):
                        recognized.append((claim.temporal_label == holiday,
                                           "event_time_scope_mismatch", claim.source_id))
        if "least favorite character" in q or "least favourite character" in q:
            for claim in graph.claims.values():
                if claim.relation in {"favorite_character", "least_favorite_character"}:
                    if phrase_in(claim.value, answer_norm):
                        recognized.append((claim.relation == "least_favorite_character",
                                           "preference_degree_mismatch", claim.source_id))
        if re.search(r"\b(?:quit|quitting|stopped|stop|gave up)\b", q):
            for claim in graph.claims.values():
                if claim.relation in {"start_activity", "end_activity"}:
                    if phrase_in(claim.value, answer_norm):
                        recognized.append((claim.relation == "end_activity",
                                           "change_direction_mismatch", claim.source_id))
        if any(match for match, _, _ in recognized):
            return GuardResult("supported", source_ids=tuple(
                dict.fromkeys(sid for match, _, sid in recognized if match)))
        if recognized:
            return GuardResult("incompatible", recognized[0][1],
                               tuple(dict.fromkeys(sid for _, _, sid in recognized)))
        return GuardResult("unknown")
