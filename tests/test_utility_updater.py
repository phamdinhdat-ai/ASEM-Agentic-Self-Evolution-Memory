"""UtilityUpdater unit tests."""

from __future__ import annotations

from datetime import datetime
import tempfile

import numpy as np
import pytest

from asem.memory_bank import MemoryBank
from asem.note import Note
from asem.utility_updater import UtilityUpdater


class _NoopBackend:
    def generate(self, prompt: str, **kwargs) -> str:
        return "summary"

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)


def _note(note_id: str, q: float) -> Note:
    return Note(
        id=note_id,
        c=note_id,
        t=datetime(2024, 1, 1),
        K=[note_id],
        G=["tag"],
        X=note_id,
        e=np.asarray([1.0, 0.0], dtype=float),
        L=[],
        z=np.asarray([1.0, 0.0], dtype=float),
        q=q,
    )


def test_ema_update_converges() -> None:
    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/bank.sqlite"
        bank = MemoryBank(db_path)

        note = _note("n1", 0.0)
        bank.add(note)

        updater = UtilityUpdater(
            backend=_NoopBackend(),
            alpha=0.1,
            q0=0.5,
            summary_prompt_template="{query} {answer} {reward}",
            note_constructor=None,
        )

        reward = 1.0
        for _ in range(10):
            current = bank.list_notes()[0]
            updater.update(reward, [current], bank)

        updated = bank.list_notes()[0]
        expected = 1.0 - (1.0 - 0.0) * (1 - 0.1) ** 10
        assert abs(updated.q - expected) < 1e-6
