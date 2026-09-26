"""Note and MemoryBank unit tests."""

from __future__ import annotations

from datetime import datetime
import gc
import tempfile
import time

import numpy as np
import pytest

from asem.note import (
    MAX_KEYWORDS,
    LinkRecord,
    Note,
    NoteConstructor,
    cap_description,
    cap_keywords,
)
from asem.memory_bank import MemoryBank


class _FakeBackend:
    def generate(self, prompt: str, **kwargs) -> str:  # noqa: D401
        return '{"keywords": ["apple"], "tags": ["food"], "description": "User likes apples."}'

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)


def test_note_constructor_build() -> None:
    backend = _FakeBackend()
    prompt = "Extract note JSON. Content: {content}"
    constructor = NoteConstructor(backend=backend, prompt_template=prompt, q0=0.5)

    note = constructor.build("I like apples.", datetime(2024, 1, 1))

    assert note.K == ["apple"]
    assert note.G == ["food"]
    assert note.X == "User likes apples."
    assert note.L == []
    assert note.q == 0.5
    assert note.e.shape == (3,)
    assert note.z.shape == (3,)


def test_memory_bank_save_load() -> None:
    pytest.importorskip("faiss")

    backend = _FakeBackend()
    prompt = "Extract note JSON. Content: {content}"
    constructor = NoteConstructor(backend=backend, prompt_template=prompt, q0=0.5)
    note = constructor.build("I like apples.", datetime(2024, 1, 1))

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/bank.sqlite"
        bank = MemoryBank(db_path)
        bank.add(note)

        results = bank.ann_search(note.e, k=1)
        assert len(results) == 1
        assert results[0].id == note.id

        bank.update(note.id, {"q": 0.75})
        updated = bank.ann_search(note.e, k=1)[0]
        assert updated.q == 0.75

        save_path = f"{tmp}/bank_copy.sqlite"
        bank.save(save_path)
        restored = MemoryBank.load(save_path)
        restored_results = restored.ann_search(note.e, k=1)
        assert len(restored_results) == 1
        assert restored_results[0].id == note.id

        bank.delete(note.id)
        assert bank.ann_search(note.e, k=1) == []

        restored.close()
        bank.close()
        gc.collect()
        time.sleep(0.05)  # allow Windows to release file handles


# ---------------------------------------------------------------------------
# entities=None regression
#
# ``ASEMPipeline._merge_update`` used to pass `entities=merged_entities or None`
# when an UPDATE merged two notes that had no entities. ``Note.entities`` is a
# list field, so that None was persisted as the JSON literal "null" and came
# back as None, crashing every later mutation:
#
#   link_evolver._apply_links -> MemoryBank.update -> Note.to_dict
#   TypeError: 'NoneType' object is not iterable
#
# which aborted the whole ASEM ingestion of a conversation.
# ---------------------------------------------------------------------------

def _plain_note(note_id: str = "n1", entities=None) -> Note:
    return Note(
        id=note_id,
        c="Caroline moved to Berlin.",
        t=datetime(2023, 5, 8, 13, 56),
        K=["caroline", "berlin"],
        G=["moving"],
        X="Caroline moved to Berlin.",
        e=np.asarray([1.0, 0.0, 0.0], dtype=float),
        L=[],
        z=np.asarray([1.0, 0.0, 0.0], dtype=float),
        q=0.5,
        entities=entities,
    )


def test_note_to_dict_tolerates_none_entities() -> None:
    assert _plain_note(entities=None).to_dict()["entities"] == []


def test_note_from_dict_tolerates_null_entities() -> None:
    payload = _plain_note().to_dict()
    payload["entities"] = None            # what legacy rows decoded to
    assert Note.from_dict(payload).entities == []


def test_memory_bank_never_persists_null_entities() -> None:
    pytest.importorskip("faiss")
    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        note = _plain_note(entities=None)
        bank.add(note)

        raw = bank._conn.execute(
            "SELECT entities FROM notes WHERE id = ?", (note.id,)
        ).fetchone()[0]
        assert raw == "[]"
        assert bank.get_note(note.id).entities == []

        bank.close()
        gc.collect()
        time.sleep(0.05)


def test_memory_bank_update_survives_legacy_null_entities_row() -> None:
    """Exact reproduction of the ASEM v1 ingestion crash."""
    pytest.importorskip("faiss")
    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        note = _plain_note()
        bank.add(note)

        # Simulate a bank written by the buggy code: entities stored as JSON null.
        bank._conn.execute(
            "UPDATE notes SET entities = 'null' WHERE id = ?", (note.id,)
        )
        bank._conn.commit()
        assert bank.get_note(note.id).entities == []   # decoded to a list

        # This is the call that raised TypeError inside link_evolver._apply_links.
        bank.update(note.id, {"L": [LinkRecord(target_id="other", relation="semantic")]})

        after = bank.get_note(note.id)
        assert after.entities == []
        assert [lr.target_id for lr in after.L] == ["other"]
        assert after.to_dict()["entities"] == []

        bank.close()
        gc.collect()
        time.sleep(0.05)


def test_merge_update_keeps_entities_a_list() -> None:
    from asem.pipeline import ASEMPipeline

    merged = ASEMPipeline._merge_update(
        _plain_note("t1", entities=[]), _plain_note("n2", entities=[])
    )
    assert merged.entities == []
    assert merged.to_dict()["entities"] == []

    # and the merge still unions entities when both sides have them
    merged2 = ASEMPipeline._merge_update(
        _plain_note("t1", entities=["Caroline"]),
        _plain_note("n2", entities=["Berlin", "Caroline"]),
    )
    assert merged2.entities == ["Caroline", "Berlin"]


# ---------------------------------------------------------------------------
# Explicit JSON null on list fields
#
# Same bug class as above: an LLM that answers `"keywords": null` used to blow
# up the whole extraction with "NoneType object is not iterable".
# ---------------------------------------------------------------------------

def test_parse_note_fields_tolerates_null_list_fields() -> None:
    constructor = NoteConstructor(
        backend=_FakeBackend(),
        prompt_template="Extract note JSON. Content: {content}",
        q0=0.5,
    )
    K, G, X = constructor._parse_note_fields(
        '{"keywords": null, "tags": null, "entities": null, "description": "Kept."}'
    )
    assert K == []
    assert G == []
    assert X == "Kept."


def test_parse_batch_list_tolerates_null_list_fields() -> None:
    parsed = NoteConstructor._parse_batch_list(
        [
            {"keywords": None, "tags": None, "description": "d1"},
            {"keywords": ["a"], "tags": None, "description": "d2"},
        ],
        expected_count=2,
    )
    assert parsed == [([], [], "d1"), (["a"], [], "d2")]


# ---------------------------------------------------------------------------
# Bounds on the stored attributes (the write path that grows them)
# ---------------------------------------------------------------------------

def test_cap_description_leaves_short_text_untouched() -> None:
    text = "Calvin got advice from a producer to stay true to himself."

    assert cap_description(text) == text


def test_cap_description_cuts_on_a_sentence_boundary() -> None:
    """A run-on description must not end mid-clause when a sentence fits."""
    text = ("First fact sentence about the session. " * 4).strip() + " " + "X" * 400

    capped = cap_description(text, limit=120)

    assert len(capped) <= 130
    assert capped.endswith("."), f"should cut after a sentence, got {capped[-40:]!r}"


def test_cap_description_falls_back_to_a_word_boundary() -> None:
    capped = cap_description("word " * 400, limit=100)

    assert len(capped) <= 105
    assert capped.endswith("…")


def test_cap_keywords_dedupes_case_insensitively_and_caps() -> None:
    keywords = ["Alex", "alex", "Google", ""] + [f"kw{i}" for i in range(30)]

    capped = cap_keywords(keywords)

    assert capped[:2] == ["Alex", "Google"]
    assert len(capped) == MAX_KEYWORDS
    assert "" not in capped
