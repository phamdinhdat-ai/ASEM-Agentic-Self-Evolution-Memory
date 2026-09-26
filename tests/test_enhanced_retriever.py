"""Unit tests for EnhancedHybridRetriever."""

from __future__ import annotations

from datetime import datetime
import tempfile

import numpy as np
import pytest

from asem.enhanced_retriever import EnhancedHybridRetriever
from asem.memory_bank import MemoryBank
from asem.note import LinkRecord, Note


class _DummyEmbedBackend:
    def __init__(self, vector: np.ndarray):
        self._vector = vector

    def generate(self, prompt: str, **kwargs) -> str:
        return ""

    def embed(self, text: str) -> np.ndarray:
        return self._vector


def _create_note(note_id: str, vec: np.ndarray, q: float, links: list = None) -> Note:
    return Note(
        id=note_id,
        c=f"Content of {note_id}",
        t=datetime(2024, 1, 1),
        K=[note_id, "test"],
        G=["tag"],
        X=f"Description of {note_id}",
        e=vec,
        L=links or [],
        z=vec,
        q=q,
    )


def test_enhanced_retriever_multi_hop() -> None:
    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/bank.sqlite"
        bank = MemoryBank(db_path)

        v1 = np.asarray([1.0, 0.0], dtype=float)
        v2 = np.asarray([0.0, 1.0], dtype=float)

        # n1 linked to n2 via 'extends'
        n1 = _create_note("n1", v1, 0.8, links=[LinkRecord(target_id="n2", relation="extends")])
        n2 = _create_note("n2", v2, 0.5, links=[LinkRecord(target_id="n1", relation="extends")])

        bank.add(n1)
        bank.add(n2)

        backend = _DummyEmbedBackend(v1)
        retriever = EnhancedHybridRetriever(
            backend=backend,
            k1=2,
            k2=1,
            delta=0.1,
            lambda_weight=0.5,
            max_hops=1,
            enable_global_semantics=False,
        )

        results = retriever.retrieve("query", bank)
        # Should include n1 as top candidate + n2 via multi-hop traversal
        retrieved_ids = {n.id for n in results}
        assert "n1" in retrieved_ids
        assert "n2" in retrieved_ids

        bank.close()

