"""Unit tests for SinglePassSessionIngestor."""

from __future__ import annotations

import json
import tempfile
import numpy as np
import pytest

from asem.memory_bank import MemoryBank
from asem.single_pass_ingest import SinglePassSessionIngestor


class _MockLLMBackend:
    def generate(self, prompt: str, **kwargs) -> str:
        return json.dumps([
            {
                "fact": "Caroline visited Hawaii on 7 May 2023.",
                "subject": "Caroline",
                "predicate": "visited",
                "object": "Hawaii",
                "entities": ["Caroline", "Hawaii"],
                "keywords": ["travel", "Hawaii"],
                "speaker": "Caroline"
            },
            {
                "fact": "Caroline bought a souvenir in Hawaii.",
                "subject": "Caroline",
                "predicate": "bought",
                "object": "souvenir",
                "entities": ["Caroline", "Hawaii"],
                "keywords": ["shopping"],
                "speaker": "Caroline"
            }
        ])

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)


def test_single_pass_ingestor() -> None:
    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/bank.sqlite"
        bank = MemoryBank(db_path)

        backend = _MockLLMBackend()
        ingestor = SinglePassSessionIngestor(backend=backend)

        turns = ["[Caroline] I went to Hawaii.", "[Caroline] I bought a souvenir there."]
        notes = ingestor.ingest_session(turns, bank, session_date="7 May 2023")

        assert len(notes) == 2
        assert bank.size() == 2
        assert notes[0].session_date == "7 May 2023"
        assert "Hawaii" in notes[0].entities

        bank.close()

