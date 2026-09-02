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

_logger = get_logger(__name__)

# Neutral relation label assigned to links migrated from the legacy flat-ID
# format (where the LLM-identified relation type was discarded).
LEGACY_LINK_RELATION = "linked"

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts"


@dataclass(frozen=True)
class LinkRecord:
    """A typed, undirected link between two notes.

    ``target_id`` is the ID of the linked note; ``relation`` is one of
    ``contradicts / extends / causal / same-topic / temporal / semantic``
    (or ``linked`` for legacy links migrated from the old flat-ID format).
    """

    target_id: str
    relation: str = LEGACY_LINK_RELATION

    def to_dict(self) -> Dict[str, str]:
        return {"target_id": self.target_id, "relation": self.relation}

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
            )
        return cls(target_id=str(data), relation=LEGACY_LINK_RELATION)


def _try_extract_json(raw: str, expect_array: bool = True) -> Any:
    """Robust JSON extraction from LLM output.

    Handles markdown fences, leading/trailing text, and single-quoted JSON.
    """
    cleaned = raw.strip()

    # 1. Strip markdown fences
    for open_pat, close_pat in [(r"```json\s*", r"\s*```"), (r"```\s*", r"\s*```")]:
        cleaned = re.sub(rf"^{open_pat}", "", cleaned)
        cleaned = re.sub(rf"{close_pat}$", "", cleaned)

    # 2. Find outermost bracket pair
    open_br = "[" if expect_array else "{"
    close_br = "]" if expect_array else "}"
    start = cleaned.find(open_br)
    end = cleaned.rfind(close_br)
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]

    # 3. Try direct JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 4. Try ast.literal_eval
    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        pass

    # 5. Try replacing single quotes with double quotes
    if cleaned.count("'") > cleaned.count('"'):
        try:
            return json.loads(cleaned.replace("'", '"'))
        except json.JSONDecodeError:
            pass

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

    def to_dict(self) -> Dict[str, Any]:
        return {
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
            "entities": list(self.entities),
            "speaker": self.speaker,
        }

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
            entities=list(data.get("entities", [])),
            speaker=data.get("speaker"),
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
        self, content: str, timestamp: datetime, embed_e: bool = True
    ) -> Note:
        """Build a note. When ``embed_e=False`` the content+K+G+X embedding
        is deferred (``note.e`` is None) — call :meth:`complete_embedding`
        before storing. ``z`` (raw-content embedding) is always computed so
        the write gate and similarity search can run without the extra embed.
        """
        _logger.debug("NoteConstructor.build | content={!r}", content[:120])

        prompt = self.prompt_template.format(content=content)
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
            _logger.warning("NoteConstructor.build | empty parse result for content={!r} | raw={!r}",
                           content[:80], raw[:100])

        e_text = " ".join([content, " ".join(K), " ".join(G), X])
        e_vec = self.backend.embed(e_text) if embed_e else None
        z_vec = self.backend.embed(content)

        note = Note(
            id=str(uuid.uuid4()),
            c=content,
            t=timestamp,
            K=K,
            G=G,
            X=X,
            e=e_vec,
            L=[],
            z=z_vec,
            q=self.q0,
        )
        _logger.debug("NoteConstructor.build → note {} | K={} G={} X={!r}",
                      note.id, K[:3], G[:3], X[:80])
        return note

    def complete_embedding(self, note: Note) -> Note:
        """Compute the note's content+K+G+X embedding if not yet computed.

        Used with ``embed_e=False`` so notes that are never written (NOOP /
        DELETE) never pay for the embedding.
        """
        if note.e is None:
            e_text = " ".join([note.c, " ".join(note.K), " ".join(note.G), note.X])
            note.e = self.backend.embed(e_text)
        return note

    def build_batch(
        self, contents: List[str], timestamp: datetime, embed_e: bool = True
    ) -> List[Note]:
        """Build notes for multiple turns in a single LLM call.

        Args:
            contents: List of content strings (one per turn).
            timestamp: Base timestamp for all notes.
            embed_e: If False, the content+K+G+X embeddings are deferred
                (``note.e`` is None); call :meth:`complete_embedding` before
                storing. ``z`` embeddings are always computed.

        Returns:
            List of Note objects, one per content item.
        """
        n = len(contents)
        _logger.info("NoteConstructor.build_batch | turns={}", n)

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
        prompt = template.format(turns_text=turns_text)
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

            note = Note(
                id=str(uuid.uuid4()),
                c=content,
                t=timestamp,
                K=K,
                G=G,
                X=X,
                e=e_vec,
                L=[],
                z=z_vec,
                q=self.q0,
            )
            notes.append(note)

        _logger.info("NoteConstructor.build_batch → {} notes", len(notes))
        return notes

    @staticmethod
    def _fields_from_dict(data: Any) -> Tuple[List[str], List[str], str]:
        """Extract (K, G, X) from a parsed note dict (or empty fields)."""
        if not isinstance(data, dict):
            return ([], [], "")
        K = list(data.get("keywords", []))
        G = list(data.get("tags", []))
        X = str(data.get("description", ""))
        return (K, G, X)

    def _parse_note_fields(self, raw: str) -> Tuple[List[str], List[str], str]:
        """Parse K, G, X from a single-note JSON LLM output."""
        data = _try_extract_json(raw, expect_array=False)
        if not isinstance(data, dict):
            _logger.warning("NoteConstructor._parse_note_fields | JSON parse failed, raw={!r}",
                           raw[:200])
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
            _logger.warning("NoteConstructor._parse_batch_result | JSON parse failed, raw={!r}",
                           raw[:200])
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
                K = list(item.get("keywords", []))
                G = list(item.get("tags", []))
                X = str(item.get("description", ""))
                results.append((K, G, X))
            else:
                results.append(([], [], ""))

        # Pad if fewer results than expected
        while len(results) < expected_count:
            results.append(([], [], ""))

        return results[:expected_count]
