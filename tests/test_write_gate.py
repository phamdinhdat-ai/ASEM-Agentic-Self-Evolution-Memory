"""NGMC write-gate unit tests.

The gate is the deterministic short-circuit for the S2 Memory Manager: it
must ADD clearly-novel turns, NOOP near-duplicates, and leave the ambiguous
band to the LLM — all without paying the LLM cost in the confident bands.
These tests verify the gate bands directly and that the pipeline only
consults the LLM inside the ambiguous band (and audits its verdict).
"""

from __future__ import annotations

import hashlib
from datetime import datetime
import tempfile

import numpy as np
import pytest

from asem.memory_bank import MemoryBank
from asem.memory_manager import MemoryManager, Op
from asem.note import Note, NoteConstructor
from asem.pipeline import ASEMPipeline
from asem.retriever import HybridRetriever
from asem.write_gate import WriteGate


# 6-dim vectors so they stay consistent with the 6-dim _vec() embeddings
A = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])      # canonical vector
B = np.asarray([0.6, 0.8, 0.0, 0.0, 0.0, 0.0])       # cos(A, B) = 0.6 -> ambiguous
ORTHO = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])   # cos(A, ORTHO) = 0.0 -> novel


def _vec(text: str) -> np.ndarray:
    """Deterministic content-derived unit vector (different text -> far apart)."""
    h = hashlib.md5(text.encode()).digest()
    v = np.asarray(list(h[:6]), dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


class _HashBackend:
    """Fake backend: embed derives from content; generate answers by prompt prefix."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.mm_op = "ADD"

    def generate(self, prompt: str, **kwargs) -> str:
        if prompt.startswith("NC:"):
            self.calls.append("NC")
            return '{"keywords": ["k"], "tags": ["t"], "description": "d"}'
        if prompt.startswith("MM:"):
            self.calls.append("MM")
            return f'{{"op": "{self.mm_op}"}}'
        return "{}"

    def embed(self, text: str) -> np.ndarray:
        return _vec(text)


class _MapBackend(_HashBackend):
    """Backend with a fixed vector map for exact content strings."""

    def __init__(self, embed_map):
        super().__init__()
        self._map = embed_map

    def embed(self, text: str) -> np.ndarray:
        if text in self._map:
            return self._map[text]
        return _vec(text)


class _CountingBackend(_HashBackend):
    def __init__(self) -> None:
        super().__init__()
        self.embed_calls = 0

    def embed(self, text: str) -> np.ndarray:
        self.embed_calls += 1
        return _vec(text)


class _NoopLink:
    def link_and_evolve(self, m_new, M):
        return None


def _note(nid: str, z: np.ndarray) -> Note:
    return Note(
        id=nid, c=nid, t=datetime(2024, 1, 1),
        K=[], G=[], X="", e=z, L=[], z=z, q=0.5,
    )


def _make_pipeline(tmp: str, backend, gate: WriteGate):
    nc = NoteConstructor(backend=backend, prompt_template="NC:{content}", q0=0.5)
    mm = MemoryManager(backend=backend, prompt_template="MM:{content} {memory}")
    retriever = HybridRetriever(backend=backend, k1=5, k2=2, delta=0.0, lambda_weight=0.5)
    bank = MemoryBank(f"{tmp}/bank.sqlite")
    pipeline = ASEMPipeline(
        memory_bank=bank,
        note_constructor=nc,
        memory_manager=mm,
        link_evolver=_NoopLink(),  # type: ignore[arg-type]
        retriever=retriever,
        answer_agent=None,  # not used by write_path
        utility_updater=None,  # not used by write_path
        write_gate=gate,
    )
    return pipeline, bank


# ---------------------------------------------------------------------------
# Direct gate bands
# ---------------------------------------------------------------------------

def test_gate_empty_bank_adds() -> None:
    gate = WriteGate(enabled=True, tau_high=0.45, tau_redund=0.92)
    op, max_sim = gate.propose(_note("n", A), [])
    assert op == Op.ADD
    assert max_sim == 0.0


def test_gate_novel_adds() -> None:
    gate = WriteGate(enabled=True, tau_high=0.45, tau_redund=0.92)
    op, max_sim = gate.propose(_note("n", A), [_note("x", ORTHO)])
    assert op == Op.ADD
    assert max_sim == pytest.approx(0.0)
    assert gate.stats["gate_add"] == 1


def test_gate_duplicate_noops() -> None:
    gate = WriteGate(enabled=True, tau_high=0.45, tau_redund=0.92)
    op, _ = gate.propose(_note("n", A), [_note("x", A)])
    assert op == Op.NOOP
    assert gate.stats["gate_noop"] == 1


def test_gate_ambiguous_returns_none() -> None:
    gate = WriteGate(enabled=True, tau_high=0.45, tau_redund=0.92)
    op, max_sim = gate.propose(_note("n", A), [_note("x", B)])  # cos = 0.6
    assert op is None
    assert max_sim == pytest.approx(0.6)
    assert gate.stats["ambiguous"] == 1


def test_gate_disabled_returns_none() -> None:
    gate = WriteGate(enabled=False)
    op, _ = gate.propose(_note("n", A), [_note("x", A)])
    assert op is None


def test_gate_audit_counts() -> None:
    gate = WriteGate()
    gate.propose(_note("n", A), [_note("x", ORTHO)])  # gate_add
    gate.propose(_note("n", A), [_note("x", A)])      # gate_noop
    gate.record_ambiguous_llm(Op.UPDATE)
    gate.record_ambiguous_llm(Op.ADD)
    s = gate.summary()
    assert s["gate_add"] == 1
    assert s["gate_noop"] == 1
    assert s["amb_update"] == 1
    assert s["amb_add"] == 1


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def test_pipeline_gate_skips_s2_llm() -> None:
    pytest.importorskip("faiss")
    # Controlled vectors: "apple pie" -> A, "quantum computing" -> ORTHO
    backend = _MapBackend({"apple pie": A, "quantum computing": ORTHO})
    gate = WriteGate(enabled=True, tau_high=0.45, tau_redund=0.92)
    with tempfile.TemporaryDirectory() as tmp:
        pipeline, bank = _make_pipeline(tmp, backend, gate)
        try:
            # empty bank -> gate ADD
            assert pipeline.write_path("apple pie", datetime(2024, 1, 1)) is not None
            assert bank.size() == 1

            # identical turn -> gate NOOP, nothing written, S2 LLM skipped
            assert pipeline.write_path("apple pie", datetime(2024, 1, 1)) is None
            assert bank.size() == 1

            # unrelated topic -> gate ADD, S2 LLM skipped
            assert pipeline.write_path("quantum computing", datetime(2024, 1, 1)) is not None
            assert bank.size() == 2

            # the S2 LLM must never have been consulted on any gated turn
            assert "MM" not in backend.calls
            assert gate.stats["gate_add"] >= 2
            assert gate.stats["gate_noop"] >= 1
        finally:
            bank.close()


def test_pipeline_ambiguous_band_consults_llm_and_audits() -> None:
    pytest.importorskip("faiss")
    backend = _MapBackend({"seed": A, "mid": B})
    gate = WriteGate(enabled=True, tau_high=0.45, tau_redund=0.92)
    with tempfile.TemporaryDirectory() as tmp:
        pipeline, bank = _make_pipeline(tmp, backend, gate)
        try:
            # seed a note (empty bank -> gate ADD)
            assert pipeline.write_path("seed", datetime(2024, 1, 1)) is not None

            # "mid" sits in the ambiguous band (cos 0.6) -> LLM must be consulted
            backend.mm_op = "UPDATE"
            assert pipeline.write_path("mid", datetime(2024, 1, 1)) is not None
            assert "MM" in backend.calls
            assert gate.stats["ambiguous"] == 1
            assert gate.stats["amb_update"] == 1
        finally:
            bank.close()


def test_pipeline_lazy_embedding_only_for_written_notes() -> None:
    pytest.importorskip("faiss")
    backend = _CountingBackend()
    gate = WriteGate(enabled=True, tau_high=0.45, tau_redund=0.92)
    with tempfile.TemporaryDirectory() as tmp:
        pipeline, bank = _make_pipeline(tmp, backend, gate)
        try:
            # first write: z (1) + completed e (1) = 2 embeds
            pipeline.write_path("apple pie", datetime(2024, 1, 1))
            assert backend.embed_calls == 2

            # duplicate turn: only z (1) is computed, e is skipped -> 3 total
            pipeline.write_path("apple pie", datetime(2024, 1, 1))
            assert backend.embed_calls == 3
            assert bank.size() == 1
        finally:
            bank.close()
