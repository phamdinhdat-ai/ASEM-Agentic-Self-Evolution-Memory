"""Regression tests for central temporal parsing.

These pin the contract that made the LongMemEval date-arithmetic failures
possible: every LongMemEval session date used to fail to parse and fall through
to ``datetime.now()``, so each note was stamped with the day the benchmark ran.
That gave the answer prompt a confident but wrong anchor ("recorded on
3 October 2026" for a March-2022 session), which is why questions like "how many
days between X and Y" produced 445 days where the gold was 14.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from asem.temporal import parse_session_datetime


# ---------------------------------------------------------------------------
# LongMemEval: "YYYY/MM/DD (Day) HH:MM" — the format that used to fail 100%
# ---------------------------------------------------------------------------

def test_longmemeval_slash_date_with_weekday_and_time() -> None:
    dt, iso = parse_session_datetime("2023/02/20 (Mon) 12:15")
    assert iso == "2023-02-20T12:15:00Z"
    assert (dt.year, dt.month, dt.day) == (2023, 2, 20)


def test_longmemeval_question_date_format() -> None:
    """`question_date` uses the same shape and must parse too."""
    _dt, iso = parse_session_datetime("2023/04/10 (Mon) 23:07")
    assert iso == "2023-04-10T23:07:00Z"


def test_longmemeval_single_digit_month_and_day() -> None:
    _dt, iso = parse_session_datetime("2023/1/3 (Tue) 9:05")
    assert iso == "2023-01-03T09:05:00Z"


def test_longmemeval_date_without_time() -> None:
    _dt, iso = parse_session_datetime("2023/02/20 (Mon)")
    assert iso == "2023-02-20T00:00:00Z"


def test_longmemeval_lowercase_and_spaced_weekday() -> None:
    for raw in ("2023/02/20 (mon) 12:15", "2023/02/20 ( Mon ) 12:15"):
        _dt, iso = parse_session_datetime(raw)
        assert iso == "2023-02-20T12:15:00Z", raw


def test_longmemeval_never_returns_now() -> None:
    """The specific regression: a real date must not become 'today'."""
    now = datetime.now(timezone.utc)
    _dt, iso = parse_session_datetime("2022/03/02 (Wed) 22:25")
    assert iso == "2022-03-02T22:25:00Z"
    assert not (now - _dt) < timedelta(days=3), "parsed date should not be near 'now'"


def test_out_of_range_components_are_not_dates() -> None:
    """`2023/13/45` is not a real date; it must be flagged, not become today."""
    _dt, iso = parse_session_datetime("2023/13/45 (Mon) 99:99")
    assert iso is None


# ---------------------------------------------------------------------------
# LoCoMo and generic formats must keep working (regression guard)
# ---------------------------------------------------------------------------

def test_locomo_time_on_date_format() -> None:
    """LoCoMo's "1:56 pm on 8 May, 2023" must be unaffected by the new pre-pass."""
    _dt, iso = parse_session_datetime("1:56 pm on 8 May, 2023")
    assert iso == "2023-05-08T13:56:00Z"


def test_locomo_session_header_is_unwrapped() -> None:
    _dt, iso = parse_session_datetime("[Session 1 — 1:56 pm on 8 May, 2023]")
    assert iso == "2023-05-08T13:56:00Z"


def test_plain_and_iso_formats() -> None:
    assert parse_session_datetime("8 May 2023")[1] == "2023-05-08T00:00:00Z"
    assert parse_session_datetime("2023-05-08")[1] == "2023-05-08T00:00:00Z"
    assert parse_session_datetime("2023-05-08T13:56:00Z")[1] == "2023-05-08T13:56:00Z"


# ---------------------------------------------------------------------------
# The "unknown" contract
# ---------------------------------------------------------------------------

def test_empty_input_legitimately_means_now() -> None:
    """No date supplied is a real case: keep now() AND a valid ISO string."""
    _dt, iso = parse_session_datetime(None)
    assert iso is not None
    _dt2, iso2 = parse_session_datetime("   ")
    assert iso2 is not None


def test_unparseable_input_is_flagged_with_none_iso() -> None:
    """A non-empty non-date must be SIGNALLED, never silently turned into today.

    Returning a valid-looking `now()` ISO string is what let the LongMemEval
    mis-parse survive a whole benchmark run unnoticed.
    """
    _dt, iso = parse_session_datetime("not a date at all")
    assert iso is None, "unparseable input must not masquerade as a valid date"


def test_unparseable_input_does_not_raise() -> None:
    for raw in ("", None, "   ", "garbage", "2023/13/45 (Mon) 99:99"):
        dt, _iso = parse_session_datetime(raw)
        assert isinstance(dt, datetime)


def test_every_oracle_session_date_parses() -> None:
    """The measured defect: 948/948 LongMemEval session dates failed to parse.

    Skips when the dataset is not present so the suite stays runnable without it.
    """
    import json
    import os

    path = os.path.join("datasets", "longmemeval", "longmemeval_oracle.json")
    if not os.path.exists(path):
        import pytest

        pytest.skip("longmemeval_oracle.json not present")

    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    checked = 0
    failures = []
    for record in records:
        for raw in list(record.get("haystack_dates") or []) + [record.get("question_date")]:
            if not raw:
                continue
            checked += 1
            _dt, iso = parse_session_datetime(raw)
            if iso is None:
                failures.append(raw)

    assert checked > 0, "expected session dates in the oracle split"
    assert not failures, f"{len(failures)}/{checked} session dates failed to parse: {failures[:5]}"
