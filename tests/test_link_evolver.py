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

    # The current code uses batch evolution (P3_batch_evolution template)
    # and gates evolution on strong relations (extends, contradicts, causal).
    # The batch evolution template expects JSON with notes containing "id", "keywords",
    # "tags", "description".
    responses = [
        # link generation response — use "extends" (strong relation) to trigger evolution
        '[{"source": "new", "target": "n1", "relation": "extends"}]',
        # batch evolution response for neighbor
        '[{"id": "n1", "keywords": ["updated"], "tags": ["tag"], "description": "Updated."}]',
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

        updated_neighbor = bank.get_note("n1")
        assert updated_neighbor is not None
        assert any(l.target_id == "new" for l in updated_neighbor.L)
        # The LLM-identified relation type must be persisted on the edge
        assert any(
            l.target_id == "new" and l.relation == "extends"
            for l in updated_neighbor.L
        )
        assert updated_neighbor.K == ["updated"]
        assert updated_neighbor.X == "Updated."

        updated_new = bank.get_note("new")
        assert updated_new is not None
        assert any(l.target_id == "n1" for l in updated_new.L)
        assert any(
            l.target_id == "n1" and l.relation == "extends"
            for l in updated_new.L
        )

        bank.close()
