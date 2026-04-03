"""Note and MemoryBank unit tests."""

from __future__ import annotations

from datetime import datetime
import tempfile

import numpy as np
import pytest

from asem.note import NoteConstructor
from asem.memory_bank import MemoryBank


class _FakeBackend:
    def generate(self, prompt: str, **kwargs) -> str:  # noqa: D401
        return '{"keywords": ["apple"], "tags": ["food"], "description": "User likes apples."}'

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)


def test_note_constructor_build() -> None:
    backend = _FakeBackend()
    prompt = "Extract note JSON. Content: {content}"
    constructor = NoteConstructor(backend=backend, prompt_template=prompt, q0=0.5)

    note = constructor.build("I like apples.", datetime(2024, 1, 1))

    assert note.K == ["apple"]
    assert note.G == ["food"]
    assert note.X == "User likes apples."
    assert note.L == []
    assert note.q == 0.5
    assert note.e.shape == (3,)
    assert note.z.shape == (3,)


def test_memory_bank_save_load() -> None:
    pytest.importorskip("faiss")

    backend = _FakeBackend()
    prompt = "Extract note JSON. Content: {content}"
    constructor = NoteConstructor(backend=backend, prompt_template=prompt, q0=0.5)
    note = constructor.build("I like apples.", datetime(2024, 1, 1))

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/bank.sqlite"
        bank = MemoryBank(db_path)
        bank.add(note)

        results = bank.ann_search(note.e, k=1)
        assert len(results) == 1
        assert results[0].id == note.id

        bank.update(note.id, {"q": 0.75})
        updated = bank.ann_search(note.e, k=1)[0]
        assert updated.q == 0.75

        save_path = f"{tmp}/bank_copy.sqlite"
        bank.save(save_path)
        restored = MemoryBank.load(save_path)
        restored_results = restored.ann_search(note.e, k=1)
        assert len(restored_results) == 1
        assert restored_results[0].id == note.id

        bank.delete(note.id)
        assert bank.ann_search(note.e, k=1) == []
