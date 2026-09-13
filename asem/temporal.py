"""Central temporal parsing and normalization utilities for ASEM.

Provides robust conversion of diverse timestamp formats (human-readable,
ISO-8601, and dialogue session headers) into timezone-aware datetime objects
and ISO-8601 strings.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Optional, Tuple


def parse_session_datetime(
    timestamp_str: Optional[str],
) -> Tuple[datetime, Optional[str]]:
    """Convert human-readable timestamps or ISO strings to (datetime, ISO-8601 string).

    Supports:
        - LoCoMo timestamps: "1:56 pm on 8 May, 2023" -> (2023-05-08 13:56:00, "2023-05-08T13:56:00Z")
        - Session headers: "[Session 1 — 1:56 pm on 8 May, 2023]" or "Session 1 - 8 May 2023"
        - Standard dates: "8 May 2023", "May 8, 2023", "2023-05-08"
        - ISO strings: "2023-05-08T13:56:00Z", "2023-05-08T13:56:00+00:00"

    Returns:
        (datetime_obj, iso_8601_str)
    """
    if not timestamp_str or not str(timestamp_str).strip():
        now = datetime.now(timezone.utc)
        return now, now.strftime("%Y-%m-%dT%H:%M:%SZ")

    raw = str(timestamp_str).strip()

    # If raw is wrapped in brackets e.g. "[Session 1 — 1:56 pm on 8 May, 2023]"
    header_num, header_date = extract_session_header(raw)
    if header_date:
        raw = header_date.strip()

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

    # Fallback to current UTC
    now = datetime.now(timezone.utc)
    return now, now.strftime("%Y-%m-%dT%H:%M:%SZ")


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

