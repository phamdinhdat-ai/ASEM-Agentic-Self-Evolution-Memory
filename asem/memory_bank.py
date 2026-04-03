"""MemoryBank storage with FAISS and SQLite metadata."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .note import Note

try:
    import faiss
except ImportError:  # pragma: no cover - optional at runtime
    faiss = None


class MemoryBank:
    """FAISS-backed ANN index with SQLite metadata storage."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

        self._dim = self._get_dim()
        self._index = None
        self._matrix: Optional[np.ndarray] = None
        self._id_map: List[str] = []
        self._rebuild_index()

    def add(self, note: Note) -> None:
        self._set_dim_if_missing(note.e)
        payload = self._note_to_row(note)
        columns = ",".join(payload.keys())
        placeholders = ",".join(["?"] * len(payload))
        self._conn.execute(
            f"INSERT OR REPLACE INTO notes ({columns}) VALUES ({placeholders})",
            list(payload.values()),
        )
        self._conn.commit()
        self._rebuild_index()

    def update(self, note_id: str, delta: Dict[str, Any]) -> None:
        note = self._get_note(note_id)
        if note is None:
            return

        updated = note.to_dict()
        for key, value in delta.items():
            if key in {"K", "G", "L"} and value is not None:
                updated[key] = list(value)
            elif key in {"e", "z"} and value is not None:
                updated[key] = np.asarray(value, dtype=float).tolist()
            elif key == "t" and isinstance(value, datetime):
                updated[key] = value.isoformat()
            else:
                updated[key] = value

        note = Note.from_dict(updated)
        self.add(note)

    def delete(self, note_id: str) -> None:
        self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self._conn.commit()
        self._rebuild_index()

    def ann_search(self, vector: np.ndarray, k: int) -> List[Note]:
        if not self._id_map:
            return []
        query = self._normalize(vector).reshape(1, -1)
        if self._index is not None:
            _, indices = self._index.search(query, k)
            hits = [self._id_map[i] for i in indices[0] if i >= 0]
            return self._get_notes(hits)

        if self._matrix is None or self._matrix.size == 0:
            return []

        scores = np.dot(self._matrix, query.reshape(-1))
        top = np.argsort(scores)[::-1][:k]
        hits = [self._id_map[int(i)] for i in top]
        return self._get_notes(hits)

    def list_notes(self) -> List[Note]:
        rows = self._conn.execute("SELECT * FROM notes").fetchall()
        return [self._row_to_note(row) for row in rows]

    def save(self, path: str) -> None:
        if path == self._db_path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with sqlite3.connect(path) as dest:
            self._conn.backup(dest)

    @classmethod
    def load(cls, path: str) -> "MemoryBank":
        return cls(path)

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                c TEXT,
                t TEXT,
                K TEXT,
                G TEXT,
                X TEXT,
                e TEXT,
                L TEXT,
                z TEXT,
                q REAL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        self._conn.commit()

    def _get_dim(self) -> Optional[int]:
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
        if row is None:
            return None
        return int(row["value"])

    def _set_dim_if_missing(self, vec: np.ndarray) -> None:
        if self._dim is not None:
            return
        self._dim = int(vec.shape[0])
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("dim", str(self._dim)),
        )
        self._conn.commit()

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype="float32")
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _rebuild_index(self) -> None:
        if self._dim is None:
            self._index = None
            self._matrix = None
            self._id_map = []
            return

        self._index = faiss.IndexFlatIP(self._dim) if faiss is not None else None
        self._matrix = None
        self._id_map = []

        rows = self._conn.execute("SELECT id, e FROM notes").fetchall()
        if not rows:
            return

        vectors = []
        for row in rows:
            vec = np.asarray(json.loads(row["e"]), dtype="float32")
            vectors.append(self._normalize(vec))
            self._id_map.append(row["id"])

        matrix = np.vstack(vectors)
        if self._index is not None:
            self._index.add(matrix)
        else:
            self._matrix = matrix

    def _note_to_row(self, note: Note) -> Dict[str, Any]:
        return {
            "id": note.id,
            "c": note.c,
            "t": note.t.isoformat(),
            "K": json.dumps(note.K),
            "G": json.dumps(note.G),
            "X": note.X,
            "e": json.dumps(note.e.tolist()),
            "L": json.dumps(note.L),
            "z": json.dumps(note.z.tolist()),
            "q": float(note.q),
        }

    def _get_note(self, note_id: str) -> Optional[Note]:
        row = self._conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_note(row)

    def _get_notes(self, note_ids: Iterable[str]) -> List[Note]:
        notes = []
        for note_id in note_ids:
            note = self._get_note(note_id)
            if note is not None:
                notes.append(note)
        return notes

    def _row_to_note(self, row: sqlite3.Row) -> Note:
        return Note(
            id=row["id"],
            c=row["c"],
            t=datetime.fromisoformat(row["t"]),
            K=json.loads(row["K"]),
            G=json.loads(row["G"]),
            X=row["X"],
            e=np.asarray(json.loads(row["e"]), dtype=float),
            L=json.loads(row["L"]),
            z=np.asarray(json.loads(row["z"]), dtype=float),
            q=float(row["q"]),
        )
