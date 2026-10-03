"""Central temporal parsing and normalization utilities for ASEM.

Provides robust conversion of diverse timestamp formats (human-readable,
ISO-8601, and dialogue session headers) into timezone-aware datetime objects
and ISO-8601 strings.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Optional, Tuple

from .logging_utils import get_logger

_log = get_logger("temporal")

# ---------------------------------------------------------------------------
# LongMemEval session dates
# ---------------------------------------------------------------------------
# LongMemEval stamps every session as ``2023/02/20 (Mon) 12:15`` — an
# ISO-ish slash date with a parenthesised weekday wedged in the middle.
# ``datetime.strptime`` cannot consume the ``(Mon)`` group, and none of the
# formats in the list below happened to accept the slash-separated date either,
# so EVERY LongMemEval session fell through to the ``datetime.now()`` fallback.
# Measured on ``longmemeval_oracle.json``: 948/948 session dates (100%)
# mis-parsed, which stamped every note in the bank with today's date. The
# answer prompt then read "recorded on 3 October 2026" for a March-2022
# session, so date arithmetic had no valid anchor and produced answers like
# "445 days" where the gold was 14 (the true anchors, ``2/15`` and ``3/1``,
# are stated verbatim in the evidence turns).
#
# The weekday is redundant with the date, so the pre-pass simply drops it and
# normalises the separators; the existing ``%Y-%m-%d`` paths then apply.
_LME_WEEKDAY_RE = re.compile(r"\(\s*[A-Za-z]{3,9}\s*\)")
# "2023/02/20" or "2023-02-20" or "2023.02.20", optionally followed by a time.
_LME_DATE_RE = re.compile(
    r"(?P<y>\d{4})[/\-.](?P<m>\d{1,2})[/\-.](?P<d>\d{1,2})"
    r"(?:[T\s]+(?P<H>\d{1,2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?\s*(?P<ampm>[AaPp]\.?[Mm]\.?)?)?"
)


def _parse_longmemeval_style(raw: str) -> Optional[datetime]:
    """Parse ``2023/02/20 (Mon) 12:15``-style stamps, or return ``None``.

    Accepts the weekday in parentheses anywhere in the string, ``/``, ``-`` or
    ``.`` separators, an optional 12-hour suffix, and an optional 24-hour time.
    """
    cleaned = _LME_WEEKDAY_RE.sub(" ", raw).strip()
    match = _LME_DATE_RE.search(cleaned)
    if not match:
        return None

    year = int(match.group("y"))
    month = int(match.group("m"))
    day = int(match.group("d"))
    hour = int(match.group("H") or 0)
    minute = int(match.group("M") or 0)
    second = int(match.group("S") or 0)

    ampm = (match.group("ampm") or "").replace(".", "").lower()
    if ampm.startswith("p") and hour < 12:
        hour += 12
    elif ampm.startswith("a") and hour == 12:
        hour = 0

    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        # Out-of-range components (2023/13/45) — not a real date.
        return None


def parse_session_datetime(
    timestamp_str: Optional[str],
) -> Tuple[datetime, Optional[str]]:
    """Convert human-readable timestamps or ISO strings to (datetime, ISO-8601 string).

    Supports:
        - LongMemEval timestamps: "2023/02/20 (Mon) 12:15" -> (2023-02-20 12:15:00, "2023-02-20T12:15:00Z")
        - LoCoMo timestamps: "1:56 pm on 8 May, 2023" -> (2023-05-08 13:56:00, "2023-05-08T13:56:00Z")
        - Session headers: "[Session 1 — 1:56 pm on 8 May, 2023]" or "Session 1 - 8 May 2023"
        - Standard dates: "8 May 2023", "May 8, 2023", "2023-05-08"
        - ISO strings: "2023-05-08T13:56:00Z", "2023-05-08T13:56:00+00:00"

    Returns:
        (datetime_obj, iso_8601_str)

    Contract for unparseable input:
        The second element is the ISO string, or ``None`` when the input could
        not be interpreted as a date. The first element is still a ``datetime``
        so existing callers keep working, but a ``None`` ISO string is the
        signal that the value is a **placeholder, not evidence**. Callers that
        render dates (``asem.answer_agent._human_date``) use it to say
        "date unknown" instead of publishing a confidently wrong date.

        Empty input legitimately means "now" (no date was supplied at all), so
        it keeps the original ``now()`` + ISO behaviour and is NOT flagged.
        A non-empty input that fails to parse is flagged with a warning,
        because silently returning ``now()`` is what corrupted LongMemEval.
    """
    if not timestamp_str or not str(timestamp_str).strip():
        now = datetime.now(timezone.utc)
        return now, now.strftime("%Y-%m-%dT%H:%M:%SZ")

    raw = str(timestamp_str).strip()

    # If raw is wrapped in brackets e.g. "[Session 1 — 1:56 pm on 8 May, 2023]"
    header_num, header_date = extract_session_header(raw)
    if header_date:
        raw = header_date.strip()

    # 0. LongMemEval slash-dates with a parenthesised weekday (see module docstring).
    lme_dt = _parse_longmemeval_style(raw)
    if lme_dt is not None:
        return lme_dt, lme_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Try direct ISO format
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt, dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        pass

    # 2. Try split by " on " (e.g., "1:56 pm on 8 May, 2023")
    try:
        parts = raw.lower().split(" on ")
        if len(parts) == 2:
            time_part = parts[0].strip()
            date_part = parts[1].strip().replace(",", "")
            datetime_str = f"{date_part} {time_part}"
            for fmt in [
                "%d %B %Y %I:%M %p",
                "%d %b %Y %I:%M %p",
                "%B %d %Y %I:%M %p",
                "%b %d %Y %I:%M %p",
                "%Y-%m-%d %I:%M %p",
            ]:
                try:
                    dt = datetime.strptime(datetime_str, fmt).replace(tzinfo=timezone.utc)
                    return dt, dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    continue
    except Exception:
        pass

    # 3. Try date-only or date-first formats (e.g. "8 May 2023", "May 8 2023", "8 May, 2023")
    cleaned_date = raw.replace(",", "").strip()
    for fmt in [
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%B %Y",
        "%b %Y",
        "%Y",
    ]:
        try:
            dt = datetime.strptime(cleaned_date, fmt).replace(tzinfo=timezone.utc)
            return dt, dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue

    # Nothing matched. Return a placeholder datetime so positional callers keep
    # working, but flag it with a ``None`` ISO string.
    #
    # This used to return a valid-looking ``now()`` ISO stamp, which is exactly
    # how the LongMemEval mis-parse went unnoticed for a whole benchmark run:
    # every note received a plausible "3 October 2026" date and the answer
    # prompt presented it as evidence. A ``None`` ISO string makes the failure
    # visible to ``_human_date`` (renders "date unknown") and to the audit
    # script, instead of silently poisoning date arithmetic.
    _log.warning(
        "Unparseable session date {!r} — treating as unknown (no date evidence). "
        "Add the format to parse_session_datetime if it is a real corpus format.",
        str(timestamp_str)[:80],
    )
    return datetime.now(timezone.utc), None


def extract_session_header(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract session number and date string from a turn or session header.

    Examples:
        "[Session 1 — 1:56 pm on 8 May, 2023]" -> (1, "1:56 pm on 8 May, 2023")
        "[Session 2 - 2023-05-09]"             -> (2, "2023-05-09")
        "session_3: 8 May 2023"                -> (3, "8 May 2023")
    """
    if not text:
        return None, None

    # Pattern: [Session N — <date>] or [Session N - <date>] or [Session N: <date>]
    m = re.search(
        r"\[?(?:Session|session)[_\s]+(\d+)\s*(?:[—–\-:]|\s+on\s+)\s*([^\]\n]+)\]?",
        text,
    )
    if m:
        try:
            sess_num = int(m.group(1))
        except ValueError:
            sess_num = None
        date_str = m.group(2).strip().rstrip("]")
        return sess_num, date_str

    # Simpler pattern: [Session N] with no date
    m_no_date = re.search(r"\[?(?:Session|session)[_\s]+(\d+)\]?", text)
    if m_no_date:
        try:
            return int(m_no_date.group(1)), None
        except ValueError:
            return None, None

    return None, None

