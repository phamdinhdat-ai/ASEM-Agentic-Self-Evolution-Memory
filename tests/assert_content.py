"""Shared content-assertion helpers for the W10 context-builder tests and dumps.

WHY THIS MODULE EXISTS
----------------------
`_render_notes_block` wraps long lines: `_wrap(text, indent=...)` inserts a
newline plus an indent MID-SENTENCE. A raw substring assertion such as

    assert note.X in rendered          # WRONG

is therefore False for any note long enough to wrap — it tests the WRAPPER's
line breaking, not whether the fact survived. It passes only when the fixture
happens to be short enough not to wrap, which is how this class of bug hides:
the assertion is green for the wrong reason, and it silently stops testing
anything the moment someone edits a fixture to be longer.

The same trap applies to whitespace the renderer normalises, to the `…` clip
marker, and to any multi-word marker that can straddle a line break.

THE RULE (team standing rule): every assertion must name the artifact it was
derived from, and any assertion about rendered CONTENT must compare on
whitespace-normalised text via the helpers below. Never compare raw substrings
against a rendered block.

USAGE
-----
    from tests.assert_content import assert_contains, assert_absent

    assert_contains(rendered, note.X)        # wrap-safe
    assert_absent(rendered, "via unknown")   # wrap-safe

`normalize_ws` collapses ALL runs of whitespace (including the newline+indent a
wrap inserts) to a single space and strips the ends, so a wrapped sentence
compares equal to its unwrapped form. `flat()` is the same thing for callers
that need the normalised string rather than a boolean.
"""

from __future__ import annotations

__all__ = [
    "normalize_ws",
    "flat",
    "contains",
    "assert_contains",
    "assert_absent",
    "assert_contains_all",
]


def normalize_ws(text: str) -> str:
    """Collapse every whitespace run to one space; strip the ends.

    Turning a wrapped render into a single-line string is what makes a
    content assertion independent of the wrapper's line-breaking decisions.
    """
    return " ".join(str(text or "").split())


def flat(text: str) -> str:
    """Alias for :func:`normalize_ws`, for readability at call sites."""
    return normalize_ws(text)


def contains(haystack: str, needle: str) -> bool:
    """Wrap-safe containment test for rendered content."""
    return normalize_ws(needle) in normalize_ws(haystack)


def assert_contains(haystack: str, needle: str, *, label: str = "") -> None:
    """Assert ``needle`` appears in ``haystack``, ignoring line wrapping.

    ``label`` names the ARTIFACT the expectation came from (e.g. "real 365-note
    bank", "captured pre-W10 render"), which is the provenance the standing rule
    requires. Pass it whenever the expectation is not obviously self-describing.
    """
    if not contains(haystack, needle):
        where = f" [{label}]" if label else ""
        raise AssertionError(
            f"{needle[:80]!r} not found in rendered content{where}. "
            "Compared whitespace-normalised, so this is a real absence, not a "
            "line-wrap artifact."
        )


def assert_absent(haystack: str, needle: str, *, label: str = "") -> None:
    """Assert ``needle`` does NOT appear, ignoring line wrapping.

    This is the safety-critical direction: a placeholder leaking into the answer
    context is a silently wrong output, so the check must not be defeated by the
    renderer inserting a newline inside the placeholder text.
    """
    if contains(haystack, needle):
        where = f" [{label}]" if label else ""
        raise AssertionError(
            f"{needle[:80]!r} unexpectedly present in rendered content{where}."
        )


def assert_contains_all(haystack: str, needles, *, label: str = "") -> None:
    """Convenience: every needle must be present."""
    for needle in needles:
        assert_contains(haystack, needle, label=label)
