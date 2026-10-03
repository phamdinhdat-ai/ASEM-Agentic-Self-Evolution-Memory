"""Note schema and construction utilities."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .backends.base import InferenceBackend
from .llm_validator import (
    LLMRetryHandler,
    validate_batch_notes,
    validate_note_fields,
)
from .logging_utils import get_logger
from .temporal import extract_session_header, parse_session_datetime

_logger = get_logger(__name__)

# Neutral relation label assigned to links migrated from the legacy flat-ID
# format (where the LLM-identified relation type was discarded).
LEGACY_LINK_RELATION = "linked"

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts"

# ---------------------------------------------------------------------------
# Bounds on the stored attributes
# ---------------------------------------------------------------------------
# Memory evolution rewrites a note's `description` by merging in a new note. The
# prompt asks for "1-2 sentences", but a 4B backbone does not reliably obey it:
# an audit of the frozen `ds_fixed` bank found descriptions up to ~1,700 chars
# that concatenate 10+ turns from different sessions into one paragraph, mixing
# speakers. That blob is unreadable in a retrieval context (and it pushed the
# answer prompt from ~3.4k to a size where the extractable fact is buried).
#
# The prompt got tighter AND the bound is enforced here, at the write path, so a
# model that ignores the instruction cannot poison the bank. Descriptions in
# `ds_fixed` are mostly 150-350 chars, so the cap only ever trims run-ons.
MAX_DESCRIPTION_CHARS = 600
MAX_KEYWORDS = 12


def cap_description(text: str, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    """Keep whole sentences up to `limit` chars (word-boundary cut as fallback).

    Cuts on a sentence boundary when one is available before `limit`, so the
    stored fact never ends mid-clause; otherwise cuts on a word boundary.
    """
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if cut >= limit // 3:
        return head[: cut + 1].strip()
    return head.rsplit(" ", 1)[0].rstrip(",;:.") + " …"


def cap_keywords(keywords: Any, limit: int = MAX_KEYWORDS) -> List[str]:
    """Deduplicate (case-insensitive) and keep the first `limit` keywords.

    Evolution merges the keyword lists of both notes, and an audited note carried
    49 keywords — most of them restatements. Keywords are retrieval metadata, not
    answer text, so a hard cap costs nothing and keeps the context readable.
    """
    out: List[str] = []
    seen: set = set()
    for raw in keywords or []:
        word = str(raw).strip()
        key = word.lower()
        if not word or key in seen:
            continue
        seen.add(key)
        out.append(word)
        if len(out) >= limit:
            break
    return out


@dataclass(frozen=True)
class LinkRecord:
    """A typed, undirected link between two notes.

    ``target_id`` is the ID of the linked note; ``relation`` is one of
    ``contradicts / extends / causal / same-topic / temporal / semantic``
    (or ``linked`` for legacy links migrated from the old flat-ID format).

    Bi-temporal validity (``valid_from`` / ``valid_to``)
    ---------------------------------------------------
    A link may carry the interval during which the RELATIONSHIP held, in
    addition to the note's own record time. This follows the bi-temporal model
    used by Zep/Graphiti (arXiv:2501.13956 §2.2.3), which stores
    ``t_valid``/``t_invalid`` on edges so the system can answer "what was true
    when" rather than only "when was this written down".

    Why it is needed: ``Note.t`` records when a fact was *stated*, which is not
    the same as when it was *true*. LongMemEval's ``knowledge-update`` questions
    and LoCoMo's contradiction resolution both ask for the latter. Without an
    interval, a retriever that sees "user lives in Boston" superseded by "user
    lives in Seattle" cannot tell whether the first is obsolete or was simply
    an earlier state.

    Both fields are ``None`` by default and are OMITTED from ``to_dict`` when
    unset, so:
      * notes written before this change serialise byte-identically,
      * frozen banks (e.g. the LoCoMo ``ds_fixed`` banks) load unchanged, and
      * the answer renderer falls back to exactly the previous format.
    """

    target_id: str
    relation: str = LEGACY_LINK_RELATION
    # ISO-8601 string when the relationship became true, else None.
    valid_from: Optional[str] = None
    # ISO-8601 string when it stopped being true, else None (= still open).
    valid_to: Optional[str] = None

    @property
    def has_interval(self) -> bool:
        """True when this link carries any validity information."""
        return bool(self.valid_from or self.valid_to)

    def to_dict(self) -> Dict[str, str]:
        data: Dict[str, str] = {"target_id": self.target_id, "relation": self.relation}
        # Emitted only when present, so legacy notes stay byte-identical and a
        # bank round-trips without gaining null fields.
        if self.valid_from:
            data["valid_from"] = self.valid_from
        if self.valid_to:
            data["valid_to"] = self.valid_to
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "LinkRecord":
        """Build a LinkRecord from a dict, a bare note-ID string (legacy),
        or an already-constructed LinkRecord (idempotent)."""
        if isinstance(data, LinkRecord):
            return data
        if isinstance(data, dict):
            return cls(
                target_id=str(data.get("target_id", data.get("id", ""))),
                relation=str(data.get("relation", LEGACY_LINK_RELATION)),
                valid_from=(str(data["valid_from"]) if data.get("valid_from") else None),
                valid_to=(str(data["valid_to"]) if data.get("valid_to") else None),
            )
        return cls(target_id=str(data), relation=LEGACY_LINK_RELATION)


@dataclass(frozen=True)
class TripletRecord:
    """The (subject, predicate, object) fact a note was extracted from.

    Why this is persisted on the note
    ---------------------------------
    ``TemporalHyperGraph`` decides supersession ("this fact changed") from two
    indices keyed on the extracted triplet. Those indices live only in memory
    and were built only in ``add_note``; ``hydrate`` -- which loads the notes a
    bank already holds -- deliberately did NOT rebuild them, because the triplet
    was never stored anywhere. The graph object is constructed per system
    instance while the bank lives on disk, so every resumed or pre-existing bank
    contributed ZERO versioning knowledge: a fact ingested in session 5 could
    never supersede one ingested in session 1.

    Measured on the LongMemEval bank build: **0 ``superseded_by`` edges** over 27
    notes, against a corpus where ``knowledge-update`` is 78/500 of the oracle
    questions.

    Persisting the triplet on the note makes the ``(subject, predicate, object)``
    and ``(subject, predicate)`` indices rebuildable by ``hydrate()``, which is
    what turns versioning into a property of the BANK rather than of one process
    lifetime.

    Serialisation follows the ``LinkRecord.valid_from``/``valid_to`` precedent:
    the field is omitted from ``to_dict`` when the note carries no triplets, so
    legacy notes round-trip byte-identically and frozen banks (e.g. the LoCoMo
    ``ds_fixed`` banks) load unchanged.
    """

    subject: str = ""
    predicate: str = ""
    object: str = ""
    # ISO-8601 string when the fact was recorded, else None.
    recorded_at: Optional[str] = None

    @property
    def is_usable(self) -> bool:
        """True when the triplet can drive versioning (needs subject+predicate).

        Mirrors the guard in ``TemporalHyperGraph.add_note``: an empty
        subject/predicate must not claim a version-chain slot.
        """
        return bool(str(self.subject or "").strip() and str(self.predicate or "").strip())

    def to_dict(self) -> Dict[str, str]:
        data: Dict[str, str] = {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }
        if self.recorded_at:
            data["recorded_at"] = self.recorded_at
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "TripletRecord":
        """Build from a dict, or an already-constructed TripletRecord (idempotent)."""
        if isinstance(data, TripletRecord):
            return data
        if not isinstance(data, dict):
            return cls()
        return cls(
            subject=str(data.get("subject") or ""),
            predicate=str(data.get("predicate") or ""),
            object=str(data.get("object") or ""),
            recorded_at=(str(data["recorded_at"]) if data.get("recorded_at") else None),
        )


# A malformed response is repaired by regex; never spend more than this on it.
_MAX_REPAIR_CHARS = 200_000


def _strip_fences(text: str) -> str:
    """Remove a wrapped markdown fence (```json ... ```)."""
    for open_pat, close_pat in [(r"```json\s*", r"\s*```"), (r"```\s*", r"\s*```")]:
        text = re.sub(rf"^{open_pat}", "", text)
        text = re.sub(rf"{close_pat}$", "", text)
    return text


def _window_brackets(text: str, expect_array: bool) -> str:
    """Narrow to the outermost bracket pair, keeping leading prose out."""
    open_br = "[" if expect_array else "{"
    close_br = "]" if expect_array else "}"
    start = text.find(open_br)
    end = text.rfind(close_br)
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def _from_open_bracket(text: str, expect_array: bool) -> str:
    """Drop leading prose without truncating the tail (unlike `_window_brackets`).

    Used for the truncation salvage, where everything after the opening bracket
    still matters even though the array was never closed.
    """
    open_br = "[" if expect_array else "{"
    start = text.find(open_br)
    return text[start:] if start >= 0 else text


def _parse_candidate(text: str) -> Any:
    """Parse one candidate string, repairing the usual LLM syntax slips."""
    if not text or len(text) > _MAX_REPAIR_CHARS:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Trailing commas before a closer are the single most common slip
    # ("..., }" / "[1, 2, ]") and `json.loads` rejects them.
    repaired = re.sub(r",\s*([\]}])", r"\1", text)
    if repaired != text:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass

    if text.count("'") > text.count('"'):
        try:
            return json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            pass

    return None


def _salvage_truncated_array(text: str) -> Optional[list]:
    """Recover every COMPLETE object from an array the model never finished.

    A small backend running out of completion budget mid-array used to yield
    `None`, which made the ingestor throw away all the facts it *did* produce
    and fall back to one note per dialogue line (no entities, no triplet, no
    graph edges). Walk backwards to the last closing brace and close the array
    there instead.
    """
    start = text.find("[")
    if start < 0:
        return None
    body = text[start:]
    if len(body) > _MAX_REPAIR_CHARS:
        return None
    for cut in range(len(body) - 1, 0, -1):
        if body[cut] != "}":
            continue
        try:
            parsed = json.loads(body[:cut + 1] + "]")
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed:
            return parsed
    return None


def _try_extract_json(raw: str, expect_array: bool = True) -> Any:
    """Robust JSON extraction from LLM output.

    Handles markdown fences, leading/trailing text, single-quoted JSON, trailing
    commas, and — for arrays — a completion the backend cut off mid-object.
    """
    cleaned = _strip_fences((raw or "").strip())
    windowed = _window_brackets(cleaned, expect_array)

    # Try the bracket-windowed form first, then the untouched text: on a
    # truncated array `rfind("]")` can hit an *inner* array (e.g. the closing
    # bracket of `"entities": [...]`) and chop off the last complete object.
    for candidate in (windowed, cleaned):
        parsed = _parse_candidate(candidate)
        if parsed is not None:
            return parsed

    if expect_array:
        return _salvage_truncated_array(_from_open_bracket(cleaned, expect_array=True))
    return None


@dataclass
class Note:
    """Atomic memory note with temporal and entity grounding."""

    id: str
    c: str
    t: datetime
    K: List[str]
    G: List[str]
    X: str
    e: Optional[np.ndarray]   # None until NoteConstructor.complete_embedding()
    L: List[LinkRecord]       # typed links (relation-aware, backward-compatible)
    z: np.ndarray
    q: float
    session_id: Optional[str] = None
    session_date: Optional[str] = None
    timestamp_iso: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    speaker: Optional[str] = None
    # The extracted (subject, predicate, object) facts this note came from.
    # Persisted so ``TemporalHyperGraph.hydrate`` can rebuild the versioning
    # indices from a bank on disk — without this, a note that pre-dates the
    # current process can never be superseded by a newly ingested one. Omitted
    # from ``to_dict`` when empty, so legacy notes are byte-identical.
    triplets: List[TripletRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "c": self.c,
            "t": self.t.isoformat(),
            "K": list(self.K),
            "G": list(self.G),
            "X": self.X,
            "e": None if self.e is None else self.e.tolist(),
            "L": [lr.to_dict() for lr in self.L],
            "z": self.z.tolist(),
            "q": float(self.q),
            "session_id": self.session_id,
            "session_date": self.session_date,
            "timestamp_iso": self.timestamp_iso,
            # `or []` keeps this total: legacy notes may still carry
            # entities=None (they were persisted as JSON null).
            "entities": list(self.entities or []),
            "speaker": self.speaker,
        }
        # Emitted only when present (same contract as LinkRecord.valid_from),
        # so a note with no triplet serialises byte-identically to before and
        # the frozen banks gain no new key.
        if self.triplets:
            data["triplets"] = [t.to_dict() for t in self.triplets]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Note":
        return cls(
            id=str(data["id"]),
            c=str(data["c"]),
            t=datetime.fromisoformat(data["t"]),
            K=list(data.get("K", [])),
            G=list(data.get("G", [])),
            X=str(data.get("X", "")),
            e=np.asarray(data.get("e", []), dtype=float) if data.get("e") is not None else None,
            L=[LinkRecord.from_dict(item) for item in data.get("L", [])],
            z=np.asarray(data.get("z", []), dtype=float),
            q=float(data.get("q", 0.0)),
            session_id=data.get("session_id"),
            session_date=data.get("session_date"),
            timestamp_iso=data.get("timestamp_iso"),
            # `or []` tolerates `"entities": null` payloads written by older code.
            entities=list(data.get("entities") or []),
            speaker=data.get("speaker"),
            # `or []` also tolerates an explicit `"triplets": null`. Notes from a
            # frozen pre-GB1 bank simply carry no triplets and hydrate() then
            # rebuilds no versioning index for them — same as before the change.
            triplets=[TripletRecord.from_dict(t) for t in (data.get("triplets") or [])],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "Note":
        return cls.from_dict(json.loads(payload))


@dataclass
class NoteConstructor:
    """Build notes from raw content via an inference backend."""

    backend: InferenceBackend
    prompt_template: str
    q0: float = 0.5
    # > 0: re-issue the prompt with a format correction when the LLM output
    # fails to parse or violates the expected schema (small-model support).
    max_retries: int = 0
    # Batch extraction prompt (one LLM call for many turns). If None, loads
    # the enhanced file-based template from data/prompts/P1_batch_note_construction.txt.
    batch_prompt_template: Optional[str] = None

    def _retry(self) -> Optional[LLMRetryHandler]:
        if self.max_retries <= 0:
            return None
        return LLMRetryHandler(self.backend.generate, max_retries=self.max_retries)

    def build(
        self,
        content: str,
        timestamp: Optional[datetime | str] = None,
        embed_e: bool = True,
        session_date: Optional[str] = None,
        session_id: Optional[str] = None,
        entities: Optional[List[str]] = None,
        speaker: Optional[str] = None,
    ) -> Note:
        """Build a note with temporal grounding.

        When ``embed_e=False`` the content+K+G+X embedding is deferred
        (``note.e`` is None) - call :meth:`complete_embedding` before storing.
        ``z`` (raw-content embedding) is always computed so the write gate and
        similarity search can run without the extra embed.
        """
        _logger.debug("NoteConstructor.build | content={!r}", content[:120])

        # 1. Resolve temporal datetime and session metadata
        dt_obj: datetime
        iso_str: Optional[str] = None

        if session_date:
            dt_obj, iso_str = parse_session_datetime(session_date)
        elif isinstance(timestamp, str):
            dt_obj, iso_str = parse_session_datetime(timestamp)
            session_date = timestamp
        elif isinstance(timestamp, datetime):
            dt_obj = timestamp
            iso_str = dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
            session_date = dt_obj.strftime("%d %B %Y")
        else:
            # Check if content has session header e.g. [Session 1 - 8 May 2023]
            header_num, header_date = extract_session_header(content)
            if header_date:
                dt_obj, iso_str = parse_session_datetime(header_date)
                session_date = header_date
                if header_num and not session_id:
                    session_id = f"session_{header_num}"
            else:
                dt_obj, iso_str = parse_session_datetime(None)

        # 2. Extract speaker attribution from content if not explicitly provided
        if not speaker:
            spk_match = re.match(r"^\s*\[([A-Za-z0-9_\s-]+)\]", content)
            if spk_match:
                speaker = spk_match.group(1).strip()

        # 3. Format prompt template with temporal context if available
        fmt_kwargs: Dict[str, Any] = {"content": content}
        if "{session_date}" in self.prompt_template:
            fmt_kwargs["session_date"] = session_date or (dt_obj.strftime("%d %B %Y") if dt_obj else "")
        if "{timestamp}" in self.prompt_template:
            fmt_kwargs["timestamp"] = iso_str or (dt_obj.isoformat() if dt_obj else "")
        prompt = self.prompt_template.format(**fmt_kwargs)

        retry = self._retry()
        if retry is not None:
            data, _attempt = retry.invoke(
                prompt,
                parse_fn=lambda raw: _try_extract_json(raw, expect_array=False),
                validate_fn=validate_note_fields,
            )
        else:
            data = _try_extract_json(self.backend.generate(prompt), expect_array=False)
        K, G, X = self._fields_from_dict(data)

        if not K and not G and not X:
            _logger.warning(
                "NoteConstructor.build | empty parse result for content={!r}",
                content[:80],
            )

        # Extract entities from LLM response if present
        extracted_entities: List[str] = []
        if isinstance(data, dict) and "entities" in data and isinstance(data["entities"], list):
            extracted_entities = [str(e).strip() for e in data["entities"] if str(e).strip()]
        note_entities = list(entities if entities is not None else extracted_entities)

        # Extract speaker from LLM response if present
        if isinstance(data, dict) and data.get("speaker") and not speaker:
            speaker = str(data["speaker"]).strip()

        e_text = " ".join([content, " ".join(K), " ".join(G), X])
        e_vec = self.backend.embed(e_text) if embed_e else None
        z_vec = self.backend.embed(content)

        note = Note(
            id=str(uuid.uuid4()),
            c=content,
            t=dt_obj,
            K=K,
            G=G,
            X=X,
            e=e_vec,
            L=[],
            z=z_vec,
            q=self.q0,
            session_id=session_id,
            session_date=session_date,
            timestamp_iso=iso_str,
            entities=note_entities,
            speaker=speaker,
        )
        _logger.debug(
            "NoteConstructor.build -> note {} | date={} K={} G={} X={!r}",
            note.id, note.session_date, K[:3], G[:3], X[:80],
        )
        return note

    def complete_embedding(self, note: Note) -> Note:
        """Compute the note\'s content+K+G+X embedding if not yet computed.

        Used with ``embed_e=False`` so notes that are never written (NOOP /
        DELETE) never pay for the embedding.
        """
        if note.e is None:
            e_text = " ".join([note.c, " ".join(note.K), " ".join(note.G), note.X])
            note.e = self.backend.embed(e_text)
        return note

    def build_batch(
        self,
        contents: List[str],
        timestamp: Optional[datetime | str] = None,
        embed_e: bool = True,
        session_date: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Note]:
        """Build notes for multiple turns in a single LLM call with temporal grounding.

        Args:
            contents: List of content strings (one per turn).
            timestamp: Base timestamp for all notes (datetime or string).
            embed_e: If False, the content+K+G+X embeddings are deferred
                (``note.e`` is None); call :meth:`complete_embedding` before
                storing. ``z`` embeddings are always computed.
            session_date: Explicit human-readable session date.
            session_id: Optional session identifier.

        Returns:
            List of Note objects, one per content item.
        """
        n = len(contents)
        _logger.info("NoteConstructor.build_batch | turns={}", n)

        # 1. Resolve session date and timestamp
        dt_obj: datetime
        iso_str: Optional[str] = None

        if session_date:
            dt_obj, iso_str = parse_session_datetime(session_date)
        elif isinstance(timestamp, str):
            dt_obj, iso_str = parse_session_datetime(timestamp)
            session_date = timestamp
        elif isinstance(timestamp, datetime):
            dt_obj = timestamp
            iso_str = dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
            session_date = dt_obj.strftime("%d %B %Y")
        else:
            # Inspect first turn for session header
            header_num, header_date = None, None
            if contents:
                header_num, header_date = extract_session_header(contents[0])
            if header_date:
                dt_obj, iso_str = parse_session_datetime(header_date)
                session_date = header_date
                if header_num and not session_id:
                    session_id = f"session_{header_num}"
            else:
                dt_obj, iso_str = parse_session_datetime(None)

        # Build the batch prompt with numbered turns
        turns_lines = []
        for i, content in enumerate(contents):
            turns_lines.append(f"[Turn {i + 1}] {content}")
        turns_text = "\n\n".join(turns_lines)

        template = self.batch_prompt_template
        if template is None:
            template = (_PROMPTS_DIR / "P1_batch_note_construction.txt").read_text(
                encoding="utf-8"
            )

        fmt_kwargs: Dict[str, Any] = {"turns_text": turns_text}
        if "{session_date}" in template:
            # Prefer the human-readable date FOR THE PROMPT: the stored
            # session_date is an ISO instant (e.g. 2023-05-08T13:56:00Z), which
            # is awkward to reason about and blocks relative-date resolution.
            # This does NOT change the note's stored session_date.
            friendly = dt_obj.strftime("%d %B %Y") if dt_obj else ""
            fmt_kwargs["session_date"] = friendly or (session_date or "")
        if "{timestamp}" in template:
            fmt_kwargs["timestamp"] = iso_str or (dt_obj.isoformat() if dt_obj else "")
        prompt = template.format(**fmt_kwargs)

        retry = self._retry()
        if retry is not None:
            data, _attempt = retry.invoke(
                prompt,
                parse_fn=lambda raw: _try_extract_json(raw, expect_array=True),
                validate_fn=lambda d: validate_batch_notes(d, expected_count=n),
            )
        else:
            data = _try_extract_json(self.backend.generate(prompt), expect_array=True)

        # Parse the JSON array
        parsed_list = self._parse_batch_list(data, n)

        notes: List[Note] = []
        for i, (content, fields) in enumerate(zip(contents, parsed_list)):
            K, G, X = fields
            e_text = " ".join([content, " ".join(K), " ".join(G), X])
            e_vec = self.backend.embed(e_text) if embed_e else None
            z_vec = self.backend.embed(content)

            # Extract speaker from turn header if available
            speaker = None
            spk_match = re.match(r"^\s*\[([A-Za-z0-9_\s-]+)\]", content)
            if spk_match:
                speaker = spk_match.group(1).strip()

            # Extract entities if item in data is a dict containing entities
            turn_entities: List[str] = []
            if isinstance(data, list) and i < len(data) and isinstance(data[i], dict):
                # `or []` guards an explicit `"entities": null` from the LLM.
                turn_entities = [str(e).strip() for e in (data[i].get("entities") or [])
                                 if str(e).strip()]
                if not speaker and data[i].get("speaker"):
                    speaker = str(data[i]["speaker"]).strip()

            note = Note(
                id=str(uuid.uuid4()),
                c=content,
                t=dt_obj,
                K=K,
                G=G,
                X=X,
                e=e_vec,
                L=[],
                z=z_vec,
                q=self.q0,
                session_id=session_id,
                session_date=session_date,
                timestamp_iso=iso_str,
                entities=turn_entities,
                speaker=speaker,
            )
            notes.append(note)

        _logger.info(
            "NoteConstructor.build_batch -> {} notes (session_date={!r})",
            len(notes), session_date,
        )
        return notes

    @staticmethod
    def _fields_from_dict(data: Any) -> Tuple[List[str], List[str], str]:
        """Extract (K, G, X) from a parsed note dict (or empty fields)."""
        if not isinstance(data, dict):
            return ([], [], "")
        # `or []` guards explicit `null` values from the LLM for any list field.
        K = list(data.get("keywords") or [])
        G = list(data.get("tags") or [])
        X = str(data.get("description", ""))
        return (K, G, X)

    def _parse_note_fields(self, raw: str) -> Tuple[List[str], List[str], str]:
        """Parse K, G, X from a single-note JSON LLM output."""
        data = _try_extract_json(raw, expect_array=False)
        if not isinstance(data, dict):
            _logger.warning(
                "NoteConstructor._parse_note_fields | JSON parse failed, raw={!r}",
                raw[:200],
            )
            return ([], [], "")
        return self._fields_from_dict(data)

    def _parse_batch_result(
        self, raw: str, expected_count: int
    ) -> List[Tuple[List[str], List[str], str]]:
        """Parse a JSON array of note field dicts from batch LLM output.

        Falls back to empty fields for missing/unparseable items.
        """
        data = _try_extract_json(raw, expect_array=True)
        if not isinstance(data, list):
            _logger.warning(
                "NoteConstructor._parse_batch_result | JSON parse failed, raw={!r}",
                raw[:200],
            )
            return [([], [], "")] * expected_count
        return self._parse_batch_list(data, expected_count)

    @staticmethod
    def _parse_batch_list(
        data: Any, expected_count: int
    ) -> List[Tuple[List[str], List[str], str]]:
        """Extract (K, G, X) tuples from a parsed batch JSON list."""
        if not isinstance(data, list):
            return [([], [], "")] * expected_count

        results: List[Tuple[List[str], List[str], str]] = []
        for item in data:
            if isinstance(item, dict):
                # `or []` guards explicit `null` values from the LLM.
                K = list(item.get("keywords") or [])
                G = list(item.get("tags") or [])
                X = str(item.get("description", ""))
                results.append((K, G, X))
            else:
                results.append(([], [], ""))

        # Pad if fewer results than expected
        while len(results) < expected_count:
            results.append(([], [], ""))

        return results[:expected_count]
