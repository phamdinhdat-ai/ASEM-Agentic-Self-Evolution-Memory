"""LinkEvolver unit tests."""

from __future__ import annotations

from datetime import datetime
import tempfile

import numpy as np
import pytest

from asem.link_evolver import LinkEvolver
from asem.memory_bank import MemoryBank
from asem.note import Note


class _QueueBackend:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, prompt: str, **kwargs) -> str:
        if not self._responses:
            return "[]"
        return self._responses.pop(0)

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)


def _note(note_id: str, desc: str) -> Note:
    return Note(
        id=note_id,
        c=desc,
        t=datetime(2024, 1, 1),
        K=[note_id],
        G=["tag"],
        X=desc,
        e=np.asarray([1.0, 0.0, 0.0], dtype=float),
        L=[],
        z=np.asarray([1.0, 0.0, 0.0], dtype=float),
        q=0.5,
    )


def test_link_and_evolve() -> None:
    pytest.importorskip("faiss")

    responses = [
        # link generation response
        '[{"source": "new", "target": "n1", "relation": "related"}]',
        # evolution response for neighbor
        '{"keywords": ["updated"], "tags": ["tag"], "description": "Updated."}',
    ]
    backend = _QueueBackend(responses)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/bank.sqlite"
        bank = MemoryBank(db_path)

        n1 = _note("n1", "Old")
        bank.add(n1)

        new_note = _note("new", "New")
        bank.add(new_note)

        evolver = LinkEvolver(
            backend=backend,
            link_prompt_template="{new_note} {neighbors}",
            evolve_prompt_template="{existing_note} {new_note}",
            k=1,
        )

        evolver.link_and_evolve(new_note, bank)

        updated_neighbor = bank.ann_search(new_note.e, k=1)[0]
        assert "new" in updated_neighbor.L
        assert updated_neighbor.K == ["updated"]
        assert updated_neighbor.X == "Updated."

        updated_new = bank.ann_search(new_note.e, k=2)[0]
        assert "n1" in updated_new.L or "new" in updated_new.L
