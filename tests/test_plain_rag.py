"""PlainRAG — the attribution control for the graph.

PlainRAG must retrieve over the EXACT notes ASEM-THG ingested, with the graph
layer fully absent. These tests pin both halves of that contract, because a
control that silently differs from the treatment measures nothing:

  * it opens ASEM-THG's bank file (not a bank of its own), and
  * its retriever has link traversal DISABLED, so the graph edges ASEM-THG wrote
    into ``Note.L`` cannot reach the ranking through the base retriever's Phase C.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -m pytest tests/test_plain_rag.py -q
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import numpy as np
import pytest

from asem.memory_bank import MemoryBank
from asem.note import Note
from eval.baselines import PlainRAG
from eval.phase_runner import bank_dir, bank_file
from eval.systems import (
    ALL_SYSTEMS,
    BANK_FILE_NAMES,
    RETRIEVE_ONLY_SYSTEMS,
    build_plain_rag_system,
)


class _StubBackend:
    def generate(self, prompt: str, **kwargs) -> str:
        return "stub answer"

    def embed(self, text: str) -> np.ndarray:
        return np.random.RandomState(abs(hash(text)) % (2 ** 31)).rand(8).astype("float32")

    def embed_documents(self, texts):
        return [self.embed(t) for t in texts]


class _FixedBackend:
    """A 4-dim embedder so query vectors match the test notes' dimension."""

    def __init__(self, vec) -> None:
        self._v = np.asarray(vec, dtype="float32")

    def generate(self, prompt: str, **kwargs) -> str:
        return "stub answer"

    def embed(self, text: str) -> np.ndarray:
        return self._v

    def embed_documents(self, texts):
        return [self.embed(t) for t in texts]


def _note(nid: str, text: str, vec) -> Note:
    v = np.asarray(vec, dtype=float)
    return Note(
        id=nid, c=text, t=datetime(2022, 1, 1), K=[], G=[], X=text,
        e=v, L=[], z=v, q=0.5,
    )


# ---------------------------------------------------------------------------
# Bank aliasing
# ---------------------------------------------------------------------------

def test_plain_rag_is_registered_as_retrieve_only() -> None:
    assert "PlainRAG" in ALL_SYSTEMS
    assert "PlainRAG" in RETRIEVE_ONLY_SYSTEMS
    # It owns no bank of its own — it reuses ASEM-THG's.
    assert "PlainRAG" not in BANK_FILE_NAMES


def test_plain_rag_bank_path_resolves_to_asem_thg() -> None:
    """The whole control depends on this: same bank file, byte for byte."""
    for conv in ("locomo_0000", "locomo_0009"):
        thg = bank_file("root", "tag", "ASEM-THG", conv)
        rag = bank_file("root", "tag", "PlainRAG", conv)
        assert rag == thg, "PlainRAG must read ASEM-THG's bank, not its own"
        assert bank_dir("root", "tag", "PlainRAG", conv) == bank_dir(
            "root", "tag", "ASEM-THG", conv
        )


def test_plain_rag_does_not_create_a_second_bank_file(tmp_path) -> None:
    """Building PlainRAG must not write a ``plainrag.sqlite``."""
    pytest.importorskip("faiss")
    sys_obj = build_plain_rag_system(
        "configs/models/thg_v2_postfix_off.yaml", str(tmp_path), backend=_StubBackend()
    )
    assert os.path.exists(os.path.join(str(tmp_path), "asem_thg.sqlite"))
    assert not os.path.exists(os.path.join(str(tmp_path), "plainrag.sqlite"))
    sys_obj.memory_bank.close()


# ---------------------------------------------------------------------------
# The retriever must be graph-free
# ---------------------------------------------------------------------------

def test_plain_rag_retriever_has_link_traversal_disabled() -> None:
    pytest.importorskip("faiss")
    with tempfile.TemporaryDirectory() as tmp:
        sys_obj = build_plain_rag_system(
            "configs/models/thg_v2_postfix_off.yaml", tmp, backend=_StubBackend()
        )
        assert sys_obj.retriever.enable_link_traversal is False, (
            "link traversal would let ASEM-THG's graph edges (persisted in "
            "Note.L) leak into the control's ranking"
        )
        # And it is the base retriever, not the enhanced one.
        assert type(sys_obj.retriever).__name__ == "HybridRetriever"
        sys_obj.memory_bank.close()


def test_plain_rag_ranking_ignores_note_links(tmp_path) -> None:
    """Two banks identical except for Note.L must rank identically.

    This is the falsification: if the control's answer changes when only the
    graph edges change, the graph is not actually excluded.
    """
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    notes = [
        _note("a", "alpha", [1.0, 0.0, 0.0, 0.0]),
        _note("b", "beta", [0.9, 0.1, 0.0, 0.0]),
        _note("c", "gamma", [0.0, 1.0, 0.0, 0.0]),
    ]

    def build(with_links: bool) -> list:
        db = os.path.join(str(tmp_path), f"asem_thg_{int(with_links)}.sqlite")
        bank = MemoryBank(db)
        bank.clear()
        for n in notes:
            copy = _note(n.id, n.c, n.e)
            if with_links:
                from asem.note import LinkRecord

                copy.L = [LinkRecord(target_id="c", relation="same-entity")]
            bank.add(copy)
        return bank

    sys_obj = build_plain_rag_system(
        "configs/models/thg_v2_postfix_off.yaml", str(tmp_path), backend=_StubBackend()
    )
    # Query embedding must match the notes' 4 dims.
    sys_obj.retriever.backend = _FixedBackend(q)

    ids = []
    for with_links in (False, True):
        bank = build(with_links)
        sys_obj.memory_bank = bank
        ranked = sys_obj.retriever.retrieve("alpha", bank)
        ids.append([n.id for n in ranked])
        bank.close()

    assert ids[0] == ids[1], (
        f"ranking changed when only Note.L edges changed: {ids[0]} vs {ids[1]}"
    )
    sys_obj.memory_bank.close()
