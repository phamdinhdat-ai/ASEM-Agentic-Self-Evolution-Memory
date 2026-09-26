"""Unit tests for ASEM-THG system registration."""

from __future__ import annotations

import tempfile
import numpy as np
import pytest

from eval.systems import BANK_FILE_NAMES, ALL_SYSTEMS, build_system
from eval.phase_runner import close_system


class _DummyBackend:
    def generate(self, prompt: str, **kwargs) -> str:
        return "[]"

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)


def test_asem_thg_system_registration() -> None:
    assert "ASEM-THG" in BANK_FILE_NAMES
    assert "ASEM-THG" in ALL_SYSTEMS
    assert BANK_FILE_NAMES["ASEM-THG"] == "asem_thg"


def test_build_asem_thg_system() -> None:
    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        backend = _DummyBackend()
        sys_obj = build_system("ASEM-THG", "configs/default.yaml", tmp, backend=backend)
        assert sys_obj is not None
        assert hasattr(sys_obj, "ingest_conversation")
        assert hasattr(sys_obj, "answer")
        close_system(sys_obj)
