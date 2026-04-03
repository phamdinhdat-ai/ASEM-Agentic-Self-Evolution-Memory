"""HybridRetriever unit tests."""

from __future__ import annotations

from datetime import datetime
import tempfile

import numpy as np
import pytest

from asem.memory_bank import MemoryBank
from asem.note import Note
from asem.retriever import HybridRetriever


class _EmbedBackend:
    def __init__(self, vector: np.ndarray):
        self._vector = vector

    def generate(self, prompt: str, **kwargs) -> str:
        return ""

    def embed(self, text: str) -> np.ndarray:
        return self._vector


def _note(note_id: str, vec: np.ndarray, q: float) -> Note:
    return Note(
        id=note_id,
        c=note_id,
        t=datetime(2024, 1, 1),
        K=[note_id],
        G=["tag"],
        X=note_id,
        e=vec,
        L=[],
        z=vec,
        q=q,
    )


def test_phase_a_threshold() -> None:
    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/bank.sqlite"
        bank = MemoryBank(db_path)

        n1 = _note("n1", np.asarray([1.0, 0.0], dtype=float), 0.1)
        n2 = _note("n2", np.asarray([0.0, 1.0], dtype=float), 0.9)
        bank.add(n1)
        bank.add(n2)

        backend = _EmbedBackend(np.asarray([1.0, 0.0], dtype=float))
        retriever = HybridRetriever(backend=backend, k1=2, k2=1, delta=0.9, lambda_weight=0.5)

        results = retriever.retrieve("query", bank)
        assert len(results) == 1
        assert results[0].id == "n1"


def test_phase_b_rerank() -> None:
    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/bank.sqlite"
        bank = MemoryBank(db_path)

        n1 = _note("n1", np.asarray([1.0, 0.0], dtype=float), 0.1)
        n2 = _note("n2", np.asarray([0.5, 0.5], dtype=float), 0.9)
        bank.add(n1)
        bank.add(n2)

        backend = _EmbedBackend(np.asarray([1.0, 0.0], dtype=float))
        retriever = HybridRetriever(backend=backend, k1=2, k2=1, delta=0.0, lambda_weight=0.7)

        results = retriever.retrieve("query", bank)
        assert len(results) == 1
        assert results[0].id == "n2"
