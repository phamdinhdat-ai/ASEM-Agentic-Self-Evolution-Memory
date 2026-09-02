"""Unit tests for Fast-ASEM (ASEM-v3) modules."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import tempfile
import numpy as np
import pytest

from asem.backends.base import InferenceBackend
from asem.config import ASEMConfig
from asem.fast_ingest import FastSessionIngestor, parse_session_datetime
from asem.memory_bank import MemoryBank
from asem.note import Note
from asem.retriever import HybridRetriever
from asem.answer_agent import AnswerAgent


class MockBackend(InferenceBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        # If extraction prompt
        if "expert memory extraction agent" in prompt:
            return json.dumps([
                {
                    "fact": "Caroline adopted a dog named Buddy on May 8, 2023.",
                    "entities": ["Caroline", "Buddy"],
                    "keywords": ["adopted", "dog", "buddy"],
                    "tags": ["pet"],
                    "speaker": "Caroline",
                },
                {
                    "fact": "Melanie recommended a veterinarian in Seattle.",
                    "entities": ["Melanie", "Seattle"],
                    "keywords": ["veterinarian", "seattle"],
                    "tags": ["recommendation"],
                    "speaker": "Melanie",
                }
            ])
        # Direct QA prompt
        if "conversational memory QA assistant" in prompt or "Memory Notes:" in prompt:
            return "Buddy the dog"
        return "mock response"

    def _embed(self, text: str) -> np.ndarray:
        # Deterministic pseudo-embedding based on string hash
        vec = np.zeros(8, dtype="float32")
        for i, ch in enumerate(text[:8]):
            vec[i] = (ord(ch) % 10) / 10.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def test_parse_session_datetime() -> None:
    dt, iso = parse_session_datetime("1:56 pm on 8 May, 2023")
    assert dt.year == 2023
    assert dt.month == 5
    assert dt.day == 8
    assert dt.hour == 13
    assert dt.minute == 56
    assert iso == "2023-05-08T13:56:00Z"

    dt2, iso2 = parse_session_datetime("10:30 am on 15 January, 2024")
    assert dt2.year == 2024
    assert dt2.month == 1
    assert dt2.day == 15
    assert dt2.hour == 10
    assert dt2.minute == 30
    assert iso2 == "2024-01-15T10:30:00Z"


def test_config_presets() -> None:
    cfg_fast = ASEMConfig.load("fast_eval")
    assert cfg_fast.preset == "fast_eval"
    assert cfg_fast.retriever.mode == "rrf"
    assert cfg_fast.answer.direct_mode is True

    cfg_sota = ASEMConfig.load("sota_benchmark")
    assert cfg_sota.preset == "sota_benchmark"
    assert cfg_sota.retriever.max_hops == 2
    assert cfg_sota.hyperparameters.k1 == 30

    cfg_deep = ASEMConfig.load("deep_evolution")
    assert cfg_deep.preset == "deep_evolution"
    assert cfg_deep.ingestion.mode == "turn_by_turn"


def test_fast_session_ingestor() -> None:
    backend = MockBackend()
    ingestor = FastSessionIngestor(backend=backend)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test_bank.sqlite"
        bank = MemoryBank(db_path)
        try:
            turns = [
                "[Caroline] Hey, I just adopted a cute dog named Buddy today!",
                "[Melanie] That is wonderful! You should take him to Dr. Smith in Seattle.",
            ]
            notes = ingestor.ingest_session(
                turns=turns,
                session_date_str="1:56 pm on 8 May, 2023",
                session_id="session_1",
                memory_bank=bank,
            )

            assert len(notes) == 2
            assert bank.size() == 2

            n1 = notes[0]
            assert "Buddy" in n1.entities
            assert n1.session_date == "1:56 pm on 8 May, 2023"
            assert n1.timestamp_iso == "2023-05-08T13:56:00Z"

            # Check graph linking: notes in same session should be linked
            assert len(n1.L) > 0

            # Test duplicate ingestion (gated NOOP)
            notes_dup = ingestor.ingest_session(
                turns=turns,
                session_date_str="1:56 pm on 8 May, 2023",
                session_id="session_1_dup",
                memory_bank=bank,
            )
            # Should not add duplicates
            assert bank.size() == 2
        finally:
            bank.close()


def test_hybrid_rrf_retriever() -> None:
    backend = MockBackend()
    retriever = HybridRetriever(
        backend=backend,
        k1=10,
        k2=3,
        delta=0.1,
        lambda_weight=0.35,
        use_rrf=True,
        use_bm25=True,
        use_entity_filter=True,
        use_temporal_boost=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test_retriever.sqlite"
        bank = MemoryBank(db_path)
        try:
            # Seed notes
            n1 = Note(
                id="n1",
                c="Caroline adopted a dog named Buddy in May 2023.",
                t=datetime(2023, 5, 8),
                K=["adopted", "dog", "buddy"],
                G=["pet"],
                X="Caroline adopted a dog named Buddy.",
                e=backend.embed("Caroline adopted a dog named Buddy in May 2023."),
                L=[],
                z=backend.embed("Caroline adopted a dog named Buddy."),
                q=0.8,
                session_date="8 May 2023",
                entities=["Caroline", "Buddy"],
            )
            n2 = Note(
                id="n2",
                c="Caroline traveled to Paris in July 2023.",
                t=datetime(2023, 7, 15),
                K=["paris", "travel", "vacation"],
                G=["travel"],
                X="Caroline traveled to Paris.",
                e=backend.embed("Caroline traveled to Paris in July 2023."),
                L=[],
                z=backend.embed("Caroline traveled to Paris."),
                q=0.5,
                session_date="15 July 2023",
                entities=["Caroline", "Paris"],
            )
            bank.add(n1)
            bank.add(n2)

            # Retrieve with entity & temporal query
            results = retriever.retrieve("What did Caroline adopt in May?", bank)
            assert len(results) >= 1
            assert results[0].id == "n1"
        finally:
            bank.close()


def test_answer_agent_direct_mode() -> None:
    backend = MockBackend()
    agent = AnswerAgent(
        backend=backend,
        prompt_template="dummy",
        baseline_prompt_template="Memory Notes:\n{context}\n\nQuestion: {query}\nAnswer:",
        direct_mode=True,
    )

    n1 = Note(
        id="n1",
        c="Caroline adopted a dog named Buddy.",
        t=datetime(2023, 5, 8),
        K=["dog"],
        G=["pet"],
        X="Caroline adopted a dog named Buddy.",
        e=backend.embed("Caroline adopted a dog named Buddy."),
        L=[],
        z=backend.embed("Caroline adopted a dog named Buddy."),
        q=0.8,
        session_date="8 May 2023",
        entities=["Caroline", "Buddy"],
    )

    ans = agent.direct_answer("What is Caroline's pet?", [n1])
    assert ans == "Buddy the dog"
