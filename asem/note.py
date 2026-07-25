"""Note schema and construction utilities."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np

from .backends.base import InferenceBackend
from .logging_utils import get_logger

_logger = get_logger(__name__)


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
    """Atomic memory note."""

    id: str
    c: str
    t: datetime
    K: List[str]
    G: List[str]
    X: str
    e: np.ndarray
    L: List[str]
    z: np.ndarray
    q: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "c": self.c,
            "t": self.t.isoformat(),
            "K": list(self.K),
            "G": list(self.G),
            "X": self.X,
            "e": self.e.tolist(),
            "L": list(self.L),
            "z": self.z.tolist(),
            "q": float(self.q),
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
            e=np.asarray(data.get("e", []), dtype=float),
            L=list(data.get("L", [])),
            z=np.asarray(data.get("z", []), dtype=float),
            q=float(data.get("q", 0.0)),
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

    # Batch prompt: extract notes for multiple turns in one LLM call
    _BATCH_PROMPT = (
        "You are a precise memory extraction agent. Extract structured memory "
        "attributes from MULTIPLE dialogue turns at once.\n\n"
        "For EACH turn below, extract:\n"
        "- keywords: 3-8 lowercase words/phrases capturing core entities\n"
        "- tags: 2-5 broad category labels (personal, professional, event, "
        "preference, fact, question, plan, relationship, location, temporal)\n"
        "- description: 1-2 sentence third-person summary\n\n"
        "Output a JSON array with one object per turn, in order. "
        'Each object: {{"keywords": [...], "tags": [...], "description": "..."}}\n\n'
        "Turns:\n{turns_text}\n\n"
        "Output ONLY the JSON array. No markdown fences, no extra text."
    )

    def build(self, content: str, timestamp: datetime) -> Note:
        _logger.debug("NoteConstructor.build | content={!r}", content[:120])

        prompt = self.prompt_template.format(content=content)
        raw = self.backend.generate(prompt)
        K, G, X = self._parse_note_fields(raw)

        if not K and not G and not X:
            _logger.warning("NoteConstructor.build | empty parse result for content={!r} | raw={!r}",
                           content[:80], raw[:100])

        e_text = " ".join([content, " ".join(K), " ".join(G), X])
        e_vec = self.backend.embed(e_text)
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

    def build_batch(
        self, contents: List[str], timestamp: datetime
    ) -> List[Note]:
        """Build notes for multiple turns in a single LLM call.

        Args:
            contents: List of content strings (one per turn).
            timestamp: Base timestamp for all notes.

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

        prompt = self._BATCH_PROMPT.format(turns_text=turns_text)
        raw = self.backend.generate(prompt)

        # Parse the JSON array
        parsed_list = self._parse_batch_result(raw, n)

        notes: List[Note] = []
        for i, (content, fields) in enumerate(zip(contents, parsed_list)):
            K, G, X = fields
            e_text = " ".join([content, " ".join(K), " ".join(G), X])
            e_vec = self.backend.embed(e_text)
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

    def _parse_note_fields(self, raw: str) -> Tuple[List[str], List[str], str]:
        """Parse K, G, X from a single-note JSON LLM output."""
        data = _try_extract_json(raw, expect_array=False)
        if not isinstance(data, dict):
            _logger.warning("NoteConstructor._parse_note_fields | JSON parse failed, raw={!r}",
                           raw[:200])
            return ([], [], "")

        K = list(data.get("keywords", []))
        G = list(data.get("tags", []))
        X = str(data.get("description", ""))
        return (K, G, X)

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
