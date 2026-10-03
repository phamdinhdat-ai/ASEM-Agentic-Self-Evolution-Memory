"""Frozen-bank migration safety for GB1's `triplets` column.

The Lead's condition on widening WS1 into ``asem/memory_bank.py``: an existing
frozen bank must gain the nullable column and NOTHING else — same row count,
same field values, `triplets == []` everywhere. These tests copy a REAL frozen
LoCoMo bank to a temp path (the migration is a write, so read-only is not
enough) and assert it.

If no frozen bank is present on the machine the migration test is SKIPPED rather
than silently passing; a synthetic bank is always exercised so the mechanism is
covered even without the corpus.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
    C:\\Users\\Dat Pham\\anaconda3\\envs\\memory-r1\\python.exe -m pytest tests/test_bank_migration.py -q
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from asem.memory_bank import MemoryBank
from asem.note import Note, TripletRecord

REPO_ROOT = Path(__file__).resolve().parent.parent

# A real frozen ASEM-THG bank. The `ds_thg` family is the one the analysis
# cites as bank-compatibility-critical (LoCoMo `ds_fixed`).
FROZEN_CANDIDATES = [
    REPO_ROOT / "static/memory_banks/locomo10/ds_thg/ASEM-THG/locomo_0000/asem_thg.sqlite",
    REPO_ROOT / "static/memory_banks/locomo10/ds_thg/ASEM-THG/locomo_0001/asem_thg.sqlite",
    REPO_ROOT / "static/memory_banks/locomo10/ckey_deepseek/ASEM-THG/locomo_0000/asem_thg.sqlite",
]


def _find_frozen_bank() -> Path | None:
    for candidate in FROZEN_CANDIDATES:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _column_names(db_path: Path) -> set:
    conn = sqlite3.connect(str(db_path))
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()}
    finally:
        conn.close()


def _raw_rows(db_path: Path, columns: list) -> list:
    """Every pre-existing column, as raw values, ordered deterministically."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        sql = f"SELECT {','.join(columns)} FROM notes ORDER BY id"
        return [tuple(row[c] for c in columns) for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


# Columns a pre-GB1 bank is known to have; used as the byte-identity comparison
# basis (deliberately excludes `triplets`, which is the new column).
_LEGACY_COLUMNS = [
    "id", "c", "t", "K", "G", "X", "e", "L", "z", "q",
    "session_id", "session_date", "timestamp_iso", "entities", "speaker",
]


def test_frozen_bank_migration_preserves_every_row() -> None:
    """A real frozen bank opens, gains only `triplets`, and is otherwise identical.

    This is the evidence that GB1 does not invalidate any frozen bank:
      (a) it opens at all,
      (b) `COUNT(*)` is unchanged,
      (c) every legacy field is byte-identical before/after, and
      (d) `triplets` exists and is NULL for all pre-existing rows.
    """
    frozen = _find_frozen_bank()
    if frozen is None:
        pytest.skip("no frozen ASEM-THG bank on this machine")

    with tempfile.TemporaryDirectory() as tmp:
        copied = Path(tmp) / "asem_thg.sqlite"
        shutil.copy2(frozen, copied)

        # ---- pre-migration snapshot ------------------------------------
        cols_before = _column_names(copied)
        assert "triplets" not in cols_before, "fixture bank unexpectedly already migrated"
        rows_before = _raw_rows(copied, _LEGACY_COLUMNS)
        count_before = len(rows_before)
        assert count_before > 0, "frozen bank has no notes; pick another fixture"

        # ---- migration happens on open --------------------------------
        bank = MemoryBank(str(copied))
        try:
            # (b) row count unchanged
            assert bank.size() == count_before, (
                f"migration changed the row count: {bank.size()} != {count_before}"
            )
        finally:
            bank.close()

        # (d) the new column exists and is NULL for every pre-existing row
        cols_after = _column_names(copied)
        assert "triplets" in cols_after, "migration did not add the triplets column"
        assert cols_before <= cols_after, "migration dropped a pre-existing column"

        conn = sqlite3.connect(str(copied))
        try:
            non_null = conn.execute(
                "SELECT COUNT(*) FROM notes WHERE triplets IS NOT NULL"
            ).fetchone()[0]
        finally:
            conn.close()
        assert non_null == 0, (
            f"{non_null} pre-existing rows gained a non-NULL triplets value; "
            "an additive migration must leave every legacy row untouched"
        )

        # (c) every legacy field is byte-identical
        rows_after = _raw_rows(copied, _LEGACY_COLUMNS)
        assert rows_after == rows_before, "a legacy field changed during migration"

        # Every note loads with an empty triplet list, i.e. today's semantics.
        bank = MemoryBank(str(copied))
        try:
            for note in bank.list_notes():
                assert note.triplets == [], (
                    f"legacy note {note.id} came back with {note.triplets!r}"
                )
        finally:
            bank.close()


def test_legacy_row_without_column_decodes_to_empty_triplets(tmp_path=None) -> None:
    """A database whose `notes` table predates the column must still load.

    The migration loop is what handles this for a real bank; this test pins the
    `_row_to_note` guard independently, so a future change that reads
    `row["triplets"]` unguarded fails here.
    """
    tmp = Path(tempfile.mkdtemp())
    db = tmp / "legacy.sqlite"

    # Hand-build a PRE-GB1 schema (no triplets column) and one row.
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE notes (
            id TEXT PRIMARY KEY, c TEXT, t TEXT, K TEXT, G TEXT, X TEXT, e TEXT,
            L TEXT, z TEXT, q REAL, session_id TEXT, session_date TEXT,
            timestamp_iso TEXT, entities TEXT, speaker TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO notes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "legacy-1", "a fact", "2022-03-01T00:00:00", "[]", "[]", "x",
            json.dumps([0.0, 0.0]), "[]", json.dumps([0.0, 0.0]), 0.5,
            "session_1", "2022-03-01", "2022-03-01T00:00:00Z", "[]", None,
        ),
    )
    conn.commit()
    conn.close()

    bank = MemoryBank(str(db))
    try:
        note = bank.get_notes_by_ids(["legacy-1"])[0]
        assert note.triplets == []
        # …and the migration added the column without inventing a value.
        assert "triplets" in _column_names(db)
    finally:
        bank.close()


def test_triplets_round_trip_through_sqlite() -> None:
    """GB1's whole point: the triplet must survive the process boundary.

    Before the `triplets` column existed this assertion FAILED — `Note.triplets`
    came back as `[]` after a `add_many` -> `get_notes_by_ids` round trip, so
    `hydrate()` could rebuild no versioning index and a freshly ingested note
    could never supersede a pre-existing one.
    """
    tmp = Path(tempfile.mkdtemp())
    bank = MemoryBank(str(tmp / "gb1.sqlite"))
    try:
        note = Note(
            id="n1", c="Caroline lives in Boston", t=__import__("datetime").datetime(2022, 3, 1),
            K=[], G=[], X="Caroline lives in Boston",
            e=np.asarray([0.1, 0.2, 0.3]), L=[], z=np.asarray([0.1, 0.2, 0.3]), q=0.5,
            session_id="session_1", session_date="2022-03-01",
            timestamp_iso="2022-03-01T00:00:00Z", entities=["Caroline"],
            triplets=[TripletRecord("Caroline", "lives in", "Boston", "2022-03-01T00:00:00Z")],
        )
        bank.add_many([note])

        back = bank.get_notes_by_ids(["n1"])[0]
        assert len(back.triplets) == 1, "triplet did not survive the SQLite round trip"
        assert back.triplets[0].subject == "Caroline"
        assert back.triplets[0].predicate == "lives in"
        assert back.triplets[0].object == "Boston"
        assert back.triplets[0].recorded_at == "2022-03-01T00:00:00Z"
        assert back.triplets == note.triplets
    finally:
        bank.close()


def test_note_without_triplets_serialises_byte_identically() -> None:
    """A note with no triplet must not gain a `triplets` key (frozen-bank contract)."""
    note = Note(
        id="n1", c="a fact", t=__import__("datetime").datetime(2022, 3, 1),
        K=[], G=[], X="x", e=None, L=[], z=np.asarray([0.0]), q=0.5,
    )
    assert "triplets" not in note.to_dict()
    restored = Note.from_dict(note.to_dict())
    assert restored.triplets == []
    # An explicit null payload from an older writer must also be tolerated.
    payload = dict(note.to_dict(), triplets=None)
    assert Note.from_dict(payload).triplets == []
