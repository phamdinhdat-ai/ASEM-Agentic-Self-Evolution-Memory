"""Note schema and construction utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
from typing import Any, Dict, List

import numpy as np

from .backends.base import InferenceBackend
from .logging_utils import get_logger

_logger = get_logger(__name__)


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

    def _parse_note_fields(self, raw: str) -> tuple[List[str], List[str], str]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [], [], ""

        keywords = data.get("keywords", [])
        tags = data.get("tags", [])
        description = data.get("description", "")

        return list(keywords), list(tags), str(description)
