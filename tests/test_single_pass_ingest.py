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


class _RecordingBackend:
    """Captures the prompt so tests can assert the dialogue reaches the LLM."""

    def __init__(self) -> None:
        self.prompts: list = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return json.dumps([
            {
                "fact": "Caroline visited Hawaii on 7 May 2023.",
                "subject": "Caroline",
                "predicate": "visited",
                "object": "Hawaii",
                "entities": ["Caroline", "Hawaii"],
                "keywords": ["travel"],
                "speaker": "Caroline",
            }
        ])

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)


def test_single_pass_prompt_includes_dialogue() -> None:
    """Regression: the dialogue must be embedded in the extraction prompt.

    The template previously only carried ``{session_date}``; the ``dialogue``
    kwarg was silently dropped by ``str.format`` and the LLM saw no turns,
    returning ``[]`` -> 0 notes ingested.
    """
    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        backend = _RecordingBackend()
        ingestor = SinglePassSessionIngestor(backend=backend)

        turns = ["[Caroline] I went to Hawaii.", "[Caroline] I bought a souvenir there."]
        ingestor.ingest_session(turns, bank, session_date="7 May 2023")

        assert backend.prompts, "backend.generate must be called once per session"
        prompt = backend.prompts[0]
        assert "Hawaii" in prompt
        assert "I bought a souvenir there." in prompt

        bank.close()


def test_single_pass_empty_output_falls_back_to_turns() -> None:
    """An empty/refused LLM response must not silently yield 0 notes."""

    class _EmptyBackend:
        def generate(self, prompt: str, **kwargs) -> str:
            return "[]"

        def embed(self, text: str) -> np.ndarray:
            return np.asarray([1.0, 0.0], dtype=float)

    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        ingestor = SinglePassSessionIngestor(backend=_EmptyBackend())

        turns = ["[Caroline] I went to Hawaii.", "[Caroline] I bought a souvenir there."]
        notes = ingestor.ingest_session(turns, bank, session_date="7 May 2023")

        assert len(notes) == 2
        assert bank.size() == 2
        # The fallback strips the `[Speaker]` prefix off the fact text and keeps
        # the speaker in its own field, so the note is still readable evidence.
        assert notes[0].c == "I went to Hawaii."
        assert notes[0].speaker == "Caroline"

        bank.close()


class _HawaiiBackend:
    """One fact per session, always sharing the entity `Hawaii`."""

    def generate(self, prompt: str, **kwargs) -> str:
        return json.dumps([
            {
                "fact": "Caroline visited Hawaii.",
                "subject": "Caroline",
                "predicate": "visited",
                "object": "Hawaii",
                "entities": ["Caroline", "Hawaii"],
                "keywords": ["travel", "hawaii"],
                "tags": ["travel"],
                "speaker": "Caroline",
            }
        ])

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)


def test_links_are_persisted_bidirectionally_across_sessions() -> None:
    """A new note's mirror edge must land on peers that pre-date the ingest call.

    The retriever follows only outgoing links from a seed, so an edge that exists
    on just one endpoint is invisible whenever the seed is the older note. Peers
    from earlier sessions live only in the bank, so they must be written back.
    """
    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        ingestor = SinglePassSessionIngestor(backend=_HawaiiBackend())

        first = ingestor.ingest_session(
            ["[Caroline] I went to Hawaii."], bank, session_date="7 May 2023"
        )
        second = ingestor.ingest_session(
            ["[Caroline] I went to Hawaii again."], bank, session_date="8 May 2023"
        )

        earlier, later = first[0], second[0]

        # Forward edge on the newer note.
        assert any(link.target_id == earlier.id for link in later.L), later.L

        # Mirror edge, read back from the BANK (the peer was not recreated).
        stored = bank.get_note(earlier.id)
        assert stored is not None
        assert any(link.target_id == later.id for link in stored.L), stored.L

        bank.close()


class _MessyBackend:
    """Emits explicit JSON nulls and over-long fields on purpose."""

    def generate(self, prompt: str, **kwargs) -> str:
        return json.dumps([
            {
                "fact": " ".join(["Caroline talked about her week"] * 40),  # > cap
                "subject": "Caroline",
                "predicate": "talked",
                "object": "her week",
                "entities": None,                                   # explicit null
                "keywords": [f"kw{i}" for i in range(30)],          # > cap
                "tags": ["personal", "health"],
                "speaker": None,                                    # explicit null
            }
        ])

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)


def test_null_list_fields_tags_and_caps_are_handled() -> None:
    """Regression: `"entities": null` must not raise, and the write-path caps apply."""
    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        ingestor = SinglePassSessionIngestor(backend=_MessyBackend())

        notes = ingestor.ingest_session(
            ["[Caroline] I had a long week."], bank, session_date="7 May 2023"
        )

        assert len(notes) == 1
        note = notes[0]
        assert note.entities == []                      # null tolerated, no TypeError
        assert note.speaker is None
        assert note.G == ["personal", "health"]         # tags captured, not hardcoded
        assert len(note.K) == 12                        # cap_keywords
        assert len(note.c) <= 610                       # cap_description (~600 + ellipsis)

        bank.close()


class _SequenceBackend:
    """Returns a scripted response per call and records the call kwargs."""

    def __init__(self, responses, finish_reason=None) -> None:
        self._responses = list(responses)
        self.calls: list = []
        self.last_finish_reason = finish_reason

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)


def test_truncated_extraction_is_salvaged_not_fallen_back() -> None:
    """A response cut off mid-array must still produce real facts.

    Regression: the completion budget ran out mid-JSON, the whole response was
    discarded and the ingestor fell back to one note per dialogue line — no
    triplets, no entities, so the hyper-graph lost its entity/temporal edges.
    """
    pytest.importorskip("faiss")

    truncated = (
        '[{"fact": "Caroline visited Hawaii on 7 May 2023.", "subject": "Caroline", '
        '"predicate": "visited", "object": "Hawaii", "entities": ["Caroline", "Hawaii"], '
        '"keywords": ["travel"], "tags": ["travel"], "speaker": "Caroline"}, '
        '{"fact": "Caroline bought a souvenir.", "subject": "Caroline", '
        '"predicate": "bought", "object": "souvenir", "entities": ["Caroline"], '
        '"keywords": ["shopping"], "tags": ["event"], "speaker": "Caroline"}, '
        '{"fact": "Caroline liked the beach"'
    )

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        backend = _SequenceBackend([truncated], finish_reason="length")
        ingestor = SinglePassSessionIngestor(backend=backend)

        notes = ingestor.ingest_session(
            ["[Caroline] I went to Hawaii.", "[Caroline] Any turn."],
            bank,
            session_date="7 May 2023",
        )

        assert len(notes) == 2, [n.c for n in notes]
        assert notes[0].entities == ["Caroline", "Hawaii"]
        assert notes[1].G == ["event"]                 # tags from the extraction, not the fallback
        assert ingestor.last_stats["source"] == "llm"
        assert ingestor.last_stats["truncated"] is True
        # No retry was needed: the salvage recovered the complete objects.
        assert len(backend.calls) == 1

        bank.close()


def test_malformed_response_is_retried_then_succeeds() -> None:
    """`max_retries` must actually be used (it used to be dead config)."""
    pytest.importorskip("faiss")

    good = json.dumps([
        {
            "fact": "Caroline visited Hawaii on 7 May 2023.",
            "subject": "Caroline",
            "predicate": "visited",
            "object": "Hawaii",
            "entities": ["Caroline", "Hawaii"],
            "keywords": ["travel"],
            "tags": ["travel"],
            "speaker": "Caroline",
        }
    ])

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        backend = _SequenceBackend(["I'm sorry, I cannot help with that.", good])
        ingestor = SinglePassSessionIngestor(backend=backend, max_retries=1)

        notes = ingestor.ingest_session(
            ["[Caroline] I went to Hawaii."], bank, session_date="7 May 2023"
        )

        assert len(notes) == 1
        assert notes[0].c == "Caroline visited Hawaii on 7 May 2023."
        assert ingestor.last_stats["source"] == "llm_retry"
        assert len(backend.calls) == 2
        # The retry carries an explicit format-correction block.
        assert "FORMAT CORRECTION" in backend.calls[1]["prompt"]

        bank.close()


def test_no_retry_without_max_retries_still_falls_back() -> None:
    """Default (`max_retries=0`) keeps the per-turn fallback, one call total."""
    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        backend = _SequenceBackend(["still not json"])
        ingestor = SinglePassSessionIngestor(backend=backend)

        notes = ingestor.ingest_session(
            ["[Caroline] I went to Hawaii."], bank, session_date="7 May 2023"
        )

        assert len(backend.calls) == 1
        assert len(notes) == 1
        assert ingestor.last_stats["source"] == "fallback"
        assert notes[0].c == "I went to Hawaii."

        bank.close()


def test_response_format_is_passed_per_call() -> None:
    """Structured output must reach the backend, and only when configured."""
    pytest.importorskip("faiss")

    payload = json.dumps([
        {
            "fact": "Caroline visited Hawaii.",
            "subject": "Caroline",
            "predicate": "visited",
            "object": "Hawaii",
            "entities": ["Hawaii"],
            "keywords": ["travel"],
            "tags": ["travel"],
            "speaker": "Caroline",
        }
    ])

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        backend = _SequenceBackend([payload])
        schema = {"type": "json_object"}
        ingestor = SinglePassSessionIngestor(backend=backend, response_format=schema)

        ingestor.ingest_session(["[Caroline] I went to Hawaii."], bank, session_date="7 May 2023")

        assert backend.calls[0]["kwargs"]["response_format"] == schema

        bank.close()


def test_no_response_format_is_sent_when_unset() -> None:
    pytest.importorskip("faiss")

    payload = json.dumps([
        {"fact": "Caroline visited Hawaii.", "entities": [], "keywords": [], "tags": []}
    ])

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        backend = _SequenceBackend([payload])
        ingestor = SinglePassSessionIngestor(backend=backend)

        ingestor.ingest_session(["[Caroline] I went to Hawaii."], bank, session_date="7 May 2023")

        assert "response_format" not in backend.calls[0]["kwargs"]

        bank.close()


def test_fact_array_schema_is_a_valid_openai_schema() -> None:
    from asem.single_pass_ingest import build_fact_array_schema, json_schema_response_format

    schema = build_fact_array_schema()
    assert schema["type"] == "array"
    assert schema["items"]["required"] == ["fact"]

    wrapped = json_schema_response_format()
    assert wrapped["type"] == "json_schema"
    assert wrapped["json_schema"]["schema"] == schema

