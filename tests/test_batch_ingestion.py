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
