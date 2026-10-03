"""BatchIngestor JSON-extraction tests (truncation tolerance)."""

from __future__ import annotations

from asem.batch_ingestion import _extract_json


def test_extract_full_array() -> None:
    raw = '[{"source": "a", "target": "b", "relation": "extends"}]'
    data = _extract_json(raw, expect_array=True)
    assert isinstance(data, list)
    assert data[0]["relation"] == "extends"


def test_extract_fenced_array() -> None:
    raw = '```json\n[{"source": "a", "target": "b", "relation": "causal"}]\n```'
    data = _extract_json(raw, expect_array=True)
    assert isinstance(data, list)
    assert len(data) == 1


def test_salvage_truncated_array() -> None:
    """Model hit its output cap mid-array — every complete object before the
    cut must be recovered instead of returning zero links."""
    raw = (
        '[{"source": "a", "target": "b", "relation": "extends"}, '
        '{"source": "c", "target": "d", "relation": "causal"}, '
        '{"source": "e", "target": "f", "relation": "semantic"}, '
        '{"source": "g"'          # <- cut off mid-object
    )
    data = _extract_json(raw, expect_array=True)
    assert isinstance(data, list)
    assert len(data) == 3
    assert data[1] == {"source": "c", "target": "d", "relation": "causal"}


def test_salvage_truncated_mid_string_returns_none() -> None:
    """No complete object exists before the cut — must not fabricate."""
    raw = '[{"source": "a", "target": "b", "relation": "exte'
    assert _extract_json(raw, expect_array=True) is None


def test_garbage_returns_none() -> None:
    assert _extract_json("sorry, here is my answer", expect_array=True) is None


def test_salvage_truncated_array_with_inner_arrays() -> None:
    """The last complete object must survive when the objects contain arrays.

    Regression: the bracket window (`rfind("]")`) hit the *inner* array of an
    entry whose own closing brace came after it, chopped the response there and
    returned None — so a truncated extraction lost every fact instead of only
    the unfinished one.
    """
    raw = (
        '[{"fact": "a", "entities": ["Caroline", "Hawaii"]}, '
        '{"fact": "b", "entities": ["Caroline"]}, '
        '{"fact": "c", "entities": ["Car'
    )
    data = _extract_json(raw, expect_array=True)
    assert isinstance(data, list)
    assert [item["fact"] for item in data] == ["a", "b"]


def test_trailing_comma_is_repaired() -> None:
    raw = '[{"fact": "a", "tags": ["x"],},]'
    assert _extract_json(raw, expect_array=True) == [{"fact": "a", "tags": ["x"]}]


def test_truncated_object_without_any_complete_entry_returns_none() -> None:
    raw = '[{"fact": "only one, and it never clo'
    assert _extract_json(raw, expect_array=True) is None


def test_prose_around_a_fenced_truncated_array() -> None:
    """The fence and the prose must go, and both complete objects must survive."""
    raw = "Here are the facts:\n```json\n[{\"fact\": \"a\"}, {\"fact\": \"b\"}"
    data = _extract_json(raw, expect_array=True)
    assert isinstance(data, list)
    assert [item["fact"] for item in data] == ["a", "b"]


def test_object_extraction_still_works() -> None:
    raw = 'Sure!\n```json\n{"op": "ADD", "target_id": null}\n```'
    assert _extract_json(raw, expect_array=False) == {"op": "ADD", "target_id": None}
