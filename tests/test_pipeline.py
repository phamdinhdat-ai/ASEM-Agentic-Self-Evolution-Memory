"""ASEM pipeline integration tests."""

from __future__ import annotations

from datetime import datetime
import tempfile

import numpy as np
import pytest

from asem.answer_agent import AnswerAgent
from asem.memory_bank import MemoryBank
from asem.memory_manager import MemoryManager
from asem.note import NoteConstructor
from asem.pipeline import ASEMPipeline
from asem.retriever import HybridRetriever
from asem.utility_updater import UtilityUpdater


class _TaggedBackend:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompt: str, **kwargs) -> str:
        if prompt.startswith("NC:"):
            self.calls.append("NC")
            return '{"keywords": ["k"], "tags": ["t"], "description": "d"}'
        if prompt.startswith("MM:"):
            self.calls.append("MM")
            return '{"op": "ADD"}'
        if prompt.startswith("AA:"):
            self.calls.append("AA")
            return '{"selected_ids": [], "answer": "ok"}'
        if prompt.startswith("SUM:"):
            self.calls.append("SUM")
            return "summary"
        return "{}"

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)


class _NoopLinkEvolver:
    def link_and_evolve(self, m_new, M):
        return None


def test_pipeline_five_turns() -> None:
    pytest.importorskip("faiss")

    backend = _TaggedBackend()

    note_constructor = NoteConstructor(
        backend=backend,
        prompt_template="NC:{content}",
        q0=0.5,
    )
    memory_manager = MemoryManager(
        backend=backend,
        prompt_template="MM:{content} {memory}",
    )
    retriever = HybridRetriever(
        backend=backend,
        k1=5,
        k2=2,
        delta=0.0,
        lambda_weight=0.5,
    )
    answer_agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
    )
    updater = UtilityUpdater(
        backend=backend,
        alpha=0.1,
        q0=0.5,
        summary_prompt_template="SUM:{query} {answer} {reward}",
        note_constructor=note_constructor,
    )

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        pipeline = ASEMPipeline(
            memory_bank=bank,
            note_constructor=note_constructor,
            memory_manager=memory_manager,
            link_evolver=_NoopLinkEvolver(),
            retriever=retriever,
            answer_agent=answer_agent,
            utility_updater=updater,
        )

        for i in range(5):
            answer = pipeline.run_turn(
                content=f"content {i}",
                query=f"query {i}",
                reward=1.0,
                timestamp=datetime(2024, 1, 1),
            )
            assert answer == "ok"

        assert len(bank.list_notes()) == 10
        assert backend.calls[:5] == ["NC", "MM", "AA", "SUM", "NC"]
