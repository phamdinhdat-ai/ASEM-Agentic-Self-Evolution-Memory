"""MemoryBank storage with FAISS and SQLite metadata."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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

    def clear(self) -> None:
        """Remove all notes from the bank and rebuild the (empty) index."""
        self._conn.execute("DELETE FROM notes")
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

    def get_note(self, note_id: str) -> Optional[Note]:
        """Retrieve a single note by its ID. Returns None if not found."""
        return self._get_note(note_id)

    def get_notes_by_ids(self, note_ids: List[str]) -> List[Note]:
        """Retrieve multiple notes by their IDs (batch lookup)."""
        return self._get_notes(note_ids)

    def get_connected_notes(
        self, note_id: str, max_hops: int = 1
    ) -> List[Note]:
        """Traverse the link graph from a seed note up to max_hops away.

        Uses BFS over the bidirectional link set L stored in each note.
        Returns all unique reachable notes (excluding the seed) at or
        below max_hops distance.

        Parameters
        ----------
        note_id : str
            Seed note ID to start traversal from.
        max_hops : int
            Maximum number of link hops to traverse (default 1).

        Returns
        -------
        List[Note]
            Unique notes reachable within max_hops, excluding the seed.
        """
        if max_hops < 1:
            return []

        visited: Set[str] = {note_id}
        current_ring: Set[str] = {note_id}
        collected_ids: List[str] = []

        for hop in range(max_hops):
            next_ring: Set[str] = set()
            for nid in current_ring:
                note = self.get_note(nid)
                if note is None:
                    continue
                for linked_id in note.L:
                    if linked_id not in visited:
                        visited.add(linked_id)
                        next_ring.add(linked_id)
                        collected_ids.append(linked_id)
            if not next_ring:
                break
            current_ring = next_ring

        return self._get_notes(collected_ids)

    def get_link_graph(self) -> Dict[str, Any]:
        """Export the full knowledge graph as node + edge lists.

        Returns
        -------
        dict with keys:
            nodes : list of {id, content_preview, keywords, q, degree}
            edges : list of {source, target, relation}
        """
        all_notes = self.list_notes()
        note_map = {n.id: n for n in all_notes}

        nodes = []
        for n in all_notes:
            nodes.append({
                "id": n.id[:8],
                "full_id": n.id,
                "content": n.c[:80],
                "keywords": n.K[:4],
                "q": round(n.q, 3),
                "degree": len(n.L),
            })

        edges = []
        seen_edges: Set[Tuple[str, str]] = set()
        for n in all_notes:
            for target_id in n.L:
                edge_key = (n.id, target_id) if n.id < target_id else (target_id, n.id)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    target_note = note_map.get(target_id)
                    if target_note:
                        shared = set(str(k).lower() for k in n.K) & \
                                 set(str(k).lower() for k in target_note.K)
                        rel = "extends" if len(shared) >= 3 else \
                              "same-topic" if len(shared) >= 2 else \
                              "semantic" if len(shared) >= 1 else "related"
                    else:
                        rel = "related"
                    edges.append({
                        "source": n.id[:8],
                        "target": target_id[:8],
                        "source_full": n.id,
                        "target_full": target_id,
                        "relation": rel,
                    })

        return {"nodes": nodes, "edges": edges}

    def get_link_statistics(self) -> Dict[str, Any]:
        """Compute graph structure statistics.

        Returns
        -------
        dict with keys: total_nodes, total_edges, avg_degree, density,
              isolated_nodes, max_degree, num_components
        """
        graph = self.get_link_graph()
        n_nodes = len(graph["nodes"])
        n_edges = len(graph["edges"])

        if n_nodes <= 1:
            return {"total_nodes": n_nodes, "total_edges": n_edges,
                    "avg_degree": 0.0, "density": 0.0,
                    "isolated_nodes": n_nodes, "max_degree": 0,
                    "num_components": n_nodes}

        max_deg = max(n["degree"] for n in graph["nodes"])
        avg_degree = (2.0 * n_edges) / n_nodes
        max_possible = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_possible if max_possible > 0 else 0.0
        isolated = sum(1 for n in graph["nodes"] if n["degree"] == 0)

        # BFS for connected components
        adj: Dict[str, Set[str]] = {}
        for e in graph["edges"]:
            adj.setdefault(e["source"], set()).add(e["target"])
            adj.setdefault(e["target"], set()).add(e["source"])

        visited: Set[str] = set()
        components = 0
        for nid in set(n["id"] for n in graph["nodes"]):
            if nid in visited:
                continue
            components += 1
            queue = [nid]
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                for nb in adj.get(cur, set()):
                    if nb not in visited:
                        queue.append(nb)

        return {"total_nodes": n_nodes, "total_edges": n_edges,
                "avg_degree": round(avg_degree, 3),
                "density": round(density, 5),
                "isolated_nodes": isolated,
                "max_degree": max_deg,
                "num_components": components}

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
