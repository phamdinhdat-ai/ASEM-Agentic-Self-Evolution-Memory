"""Unit tests for output validation, retry handling, and LinkRecord persistence.

Covers:
* asem/llm_validator.py — validators + LLMRetryHandler
* asem/note.py — LinkRecord backward-compatible (de)serialization
* asem/memory_bank.py — round-trip of relation-typed links through SQLite
* data/prompts/ — every template renders with its expected placeholders
"""

from __future__ import annotations

import re
from datetime import datetime
import gc
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

from asem.llm_validator import (
    LLMRetryHandler,
    ValidationResult,
    validate_batch_notes,
    validate_distil_response,
    validate_link_array,
    validate_memory_ops,
    validate_note_fields,
    validate_summary,
)
from asem.note import LEGACY_LINK_RELATION, LinkRecord, Note, NoteConstructor
from asem.memory_bank import MemoryBank

PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "prompts"
)

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def test_validate_note_fields_ok() -> None:
    data = {"keywords": ["apple"], "tags": ["food"], "description": "Likes apples."}
    result = validate_note_fields(data)
    assert result.valid
    assert result.parsed is data


def test_validate_note_fields_fails_on_bad_shape() -> None:
    assert not validate_note_fields("not-a-dict").valid
    assert not validate_note_fields({"keywords": "oops", "tags": [], "description": "x"}).valid
    assert not validate_note_fields({"keywords": [], "tags": [], "description": 42}).valid
    assert not validate_note_fields({"keywords": [], "tags": []}).valid  # missing description


def test_validate_link_array_ok_and_cleans_relation() -> None:
    data = [
        {"source": "n1", "target": "e1", "relation": "Extends"},  # mixed case → lower
        {"source": "n1", "target": "e2", "relation": "semantic"},
    ]
    result = validate_link_array(data, valid_source_id="n1",
                                 valid_target_ids={"e1", "e2"})
    assert result.valid
    assert result.parsed[0]["relation"] == "extends"


def test_validate_link_array_unknown_relation_strict() -> None:
    data = [{"source": "n1", "target": "e1", "relation": "similar"}]
    result = validate_link_array(data, valid_source_id="n1", valid_target_ids={"e1"},
                                 allow_unknown_relations=False)
    assert not result.valid
    assert any("invalid relation" in e for e in result.errors)


def test_validate_link_array_unknown_relation_soft_remap() -> None:
    data = [{"source": "n1", "target": "e1", "relation": "similar"}]
    result = validate_link_array(data, valid_source_id="n1", valid_target_ids={"e1"},
                                 allow_unknown_relations=True)
    assert result.valid
    assert result.parsed[0]["relation"] == "semantic"


def test_validate_link_array_bad_ids_fail() -> None:
    data = [{"source": "wrong", "target": "ghost", "relation": "extends"}]
    result = validate_link_array(data, valid_source_id="n1", valid_target_ids={"e1"})
    assert not result.valid
    assert len(result.errors) == 2


def test_validate_link_array_self_link_allowed_by_validator() -> None:
    # The validator does not forbid self-links — the caller's prompt does.
    data = [{"source": "n1", "target": "n1", "relation": "same-topic"}]
    result = validate_link_array(data, valid_source_id="n1", valid_target_ids={"n1"})
    assert result.valid


def test_validate_memory_ops() -> None:
    ok = [
        {"index": 0, "op": "ADD", "target_id": None},
        {"index": 1, "op": "UPDATE", "target_id": "m1"},
        {"index": 2, "op": "NOOP", "target_id": None},
    ]
    assert validate_memory_ops(ok).valid

    bad = [
        {"index": 0, "op": "MERGE", "target_id": None},          # invalid op
        {"index": 1, "op": "UPDATE", "target_id": None},          # UPDATE needs target
        {"index": "x", "op": "ADD", "target_id": None},           # non-int index
    ]
    result = validate_memory_ops(bad)
    assert not result.valid
    assert len(result.errors) >= 3


def test_validate_batch_notes_count_mismatch() -> None:
    data = [{"keywords": [], "tags": [], "description": "a"}]
    result = validate_batch_notes(data, expected_count=3)
    assert not result.valid
    assert "expected exactly 3" in result.errors[0]


def test_validate_batch_notes_missing_fields() -> None:
    result = validate_batch_notes([{"keywords": [], "tags": []}], expected_count=1)
    assert not result.valid


def test_validate_batch_notes_require_content_flag() -> None:
    # An entry with the right schema but empty/missing content would pass the
    # base validator yet be silently dropped during embedding — only the
    # extraction flag catches it.
    bad = [{"content": "", "keywords": [], "tags": [], "description": "d"}]
    assert validate_batch_notes(bad).valid
    result = validate_batch_notes(bad, require_content=True)
    assert not result.valid
    assert any("content" in e for e in result.errors)

    ok = [{"content": "hi", "keywords": [], "tags": [], "description": "d"}]
    assert validate_batch_notes(ok, require_content=True).valid

    # Evolution-style entries (no content key) must still pass without the flag
    evolve = [{"id": "n1", "keywords": ["k"], "tags": ["t"], "description": "d"}]
    assert validate_batch_notes(evolve).valid


def test_validate_distil_response() -> None:
    assert validate_distil_response(
        {"selected_ids": ["n1"], "answer": "Alex works at Google."}
    ).valid
    assert not validate_distil_response({"selected_ids": "n1", "answer": "x"}).valid
    assert not validate_distil_response({"selected_ids": [], "answer": 3}).valid


def test_validate_summary() -> None:
    assert validate_summary("  Alex has a dog.  ").parsed == "Alex has a dog."
    assert not validate_summary("").valid
    assert not validate_summary(42).valid


# ---------------------------------------------------------------------------
# LLMRetryHandler
# ---------------------------------------------------------------------------

class _ScriptedGenerator:
    """Returns a canned response per call, in order."""

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


def _parse_json(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def test_retry_succeeds_first_attempt() -> None:
    gen = _ScriptedGenerator(['{"keywords": ["a"], "tags": ["t"], "description": "d"}'])
    handler = LLMRetryHandler(gen.generate, max_retries=2)
    parsed, attempt = handler.invoke(
        "template", parse_fn=_parse_json, validate_fn=validate_note_fields)
    assert attempt == 0
    assert parsed["keywords"] == ["a"]
    assert len(gen.calls) == 1


def test_retry_recovers_after_parse_failure() -> None:
    gen = _ScriptedGenerator([
        "Sorry, here is my answer: sure thing!",          # unparseable
        '{"keywords": ["a"], "tags": ["t"], "description": "d"}',
    ])
    handler = LLMRetryHandler(gen.generate, max_retries=2)
    parsed, attempt = handler.invoke(
        "template", parse_fn=_parse_json, validate_fn=validate_note_fields)
    assert attempt == 1
    assert parsed["keywords"] == ["a"]
    # The retry prompt must quote the bad output and ask for correction
    assert "FORMAT CORRECTION" in gen.calls[1]
    assert "sure thing" in gen.calls[1]


def test_retry_recovers_after_validation_failure() -> None:
    gen = _ScriptedGenerator([
        '{"keywords": "oops", "tags": [], "description": "d"}',  # schema violation
        '{"keywords": ["a"], "tags": ["t"], "description": "d"}',
    ])
    handler = LLMRetryHandler(gen.generate, max_retries=2)
    parsed, attempt = handler.invoke(
        "template", parse_fn=_parse_json, validate_fn=validate_note_fields)
    assert attempt == 1
    # The retry prompt names the exact error
    assert "must be a JSON array" in gen.calls[1]


def test_retry_exhausts_and_returns_last_parse() -> None:
    gen = _ScriptedGenerator([
        '{"keywords": "bad", "tags": [], "description": "d"}',
        '{"keywords": "still-bad", "tags": [], "description": "d"}',
        '{"keywords": "still-bad", "tags": [], "description": "d"}',
    ])
    handler = LLMRetryHandler(gen.generate, max_retries=2)
    parsed, attempt = handler.invoke(
        "template", parse_fn=_parse_json, validate_fn=validate_note_fields)
    assert attempt == 2
    assert parsed["keywords"] == "still-bad"  # best-effort salvage
    assert len(gen.calls) == 3


def test_retry_disabled_single_shot() -> None:
    gen = _ScriptedGenerator(['not json at all'])
    handler = LLMRetryHandler(gen.generate, max_retries=0)
    parsed, attempt = handler.invoke(
        "template", parse_fn=_parse_json, validate_fn=validate_note_fields)
    assert parsed is None
    assert attempt == 0
    assert len(gen.calls) == 1


# ---------------------------------------------------------------------------
# LinkRecord backward compatibility + serialization
# ---------------------------------------------------------------------------

def test_linkrecord_from_variants() -> None:
    assert LinkRecord.from_dict("n1") == LinkRecord("n1", LEGACY_LINK_RELATION)
    assert LinkRecord.from_dict({"target_id": "n1", "relation": "extends"}) == \
        LinkRecord("n1", "extends")
    assert LinkRecord.from_dict({"id": "n1"}) == LinkRecord("n1", LEGACY_LINK_RELATION)
    rec = LinkRecord("n1", "causal")
    assert LinkRecord.from_dict(rec) is rec  # idempotent


def test_note_l_round_trip_with_relation_types() -> None:
    note = Note(
        id="m1", c="content", t=datetime(2024, 1, 1),
        K=["a"], G=["g"], X="desc",
        e=np.zeros(4), L=[LinkRecord("n2", "extends"), LinkRecord("n3", "linked")],
        z=np.zeros(4), q=0.5,
    )
    payload = note.to_dict()
    assert payload["L"] == [
        {"target_id": "n2", "relation": "extends"},
        {"target_id": "n3", "relation": "linked"},
    ]
    restored = Note.from_dict(payload)
    assert restored.L == note.L


def test_note_l_accepts_legacy_flat_ids() -> None:
    payload = {
        "id": "m1", "c": "content", "t": "2024-01-01T00:00:00",
        "K": [], "G": [], "X": "desc", "L": ["n1", "n2"], "q": 0.5,
    }
    note = Note.from_dict(payload)
    assert [l.target_id for l in note.L] == ["n1", "n2"]
    assert all(l.relation == LEGACY_LINK_RELATION for l in note.L)


class _FakeBackend:
    def generate(self, prompt: str, **kwargs) -> str:
        return '{"keywords": ["apple"], "tags": ["food"], "description": "User likes apples."}'

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)


def test_memory_bank_persists_relation_typed_links() -> None:
    pytest.importorskip("faiss")
    backend = _FakeBackend()
    nc = NoteConstructor(backend=backend, prompt_template="NC:{content}", q0=0.5)

    a = nc.build("Alice likes apples.", datetime(2024, 1, 1))
    b = nc.build("Bob likes bananas.", datetime(2024, 1, 2))
    a.L.append(LinkRecord(b.id, "extends"))

    tmp = tempfile.mkdtemp()
    try:
        path = f"{tmp}/bank.sqlite"
        bank = MemoryBank(path)
        bank.add(a)
        bank.add(b)
        bank.update(a.id, {"L": [l.to_dict() for l in a.L]})

        # Reload from disk — relation type must survive the SQLite round trip
        save_path = f"{tmp}/copy.sqlite"
        bank.save(save_path)
        restored = MemoryBank.load(save_path)

        graph = restored.get_link_graph()
        edge = next(
            e for e in graph["edges"]
            if {e["source_full"], e["target_full"]} == {a.id, b.id}
        )
        assert edge["relation"] == "extends"

        connected = restored.get_connected_notes(a.id)
        assert any(n.id == b.id for n in connected)

        bank.close()
        restored.close()
        gc.collect()
        time.sleep(0.05)  # allow Windows to release file handles
    finally:
        # Best-effort cleanup: Windows AV scanners occasionally hold
        # sqlite temp files; assertions above are what matters.
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Prompt templates render with the expected placeholders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,placeholders", [
    ("P1_note_construction.txt", ["content"]),
    ("P2_link_generation.txt", ["new_note", "neighbors"]),
    ("P3_memory_evolution.txt", ["existing_note", "new_note"]),
    ("P4_batch_note_extraction.txt", ["dialogue"]),
    ("P5_batch_memory_ops.txt", ["new_notes", "existing_memory"]),
    ("P6_batch_link_generation.txt", ["new_notes", "neighbors"]),
    ("P1_batch_note_construction.txt", ["turns_text"]),
    ("P3_batch_evolution.txt", ["existing_notes", "new_note"]),
    ("P_memory_manager.txt", ["content", "memory"]),
    ("P_distil.txt", ["query", "candidates"]),
    ("P_summary.txt", ["query", "answer", "reward"]),
])
def test_prompt_template_renders(filename: str, placeholders: List[str]) -> None:
    """Every prompt file must be a valid str.format template.

    A stray unescaped brace, or a placeholder the caller does not fill,
    would raise here — this catches the small-model prompt edits
    breaking at runtime.
    """
    path = PROMPTS_DIR / filename
    assert path.exists(), f"missing prompt file {path}"
    template = path.read_text(encoding="utf-8")

    # All placeholders in the file must be expected (typo guard)
    found = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", template))
    assert found == set(placeholders), f"{filename}: unexpected placeholders {found}"

    # Rendering must succeed (escaped braces + all kwargs present)
    rendered = template.format(**{p: "X" for p in placeholders})
    assert rendered  # non-empty


def test_note_constructor_batch_default_template() -> None:
    """build_batch with batch_prompt_template=None loads the file template."""
    class _BatchBackend(_FakeBackend):
        def generate(self, prompt: str, **kwargs) -> str:
            return json.dumps([
                {"keywords": ["apple"], "tags": ["food"],
                 "description": "User likes apples."},
                {"keywords": ["banana"], "tags": ["food"],
                 "description": "User likes bananas."},
            ])

    nc = NoteConstructor(backend=_BatchBackend(), prompt_template="NC:{content}", q0=0.5)
    notes = nc.build_batch(["turn one", "turn two"], datetime(2024, 1, 1))
    assert len(notes) == 2
    assert notes[0].K == ["apple"]
    assert notes[1].K == ["banana"]
