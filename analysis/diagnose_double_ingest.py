"""Decisive test: does the lazy `answer()` re-ingest REPLACE or ADD to the bank?

Owned by WS2. Runs ONE LongMemEval question and samples the bank at three
points, printing the note count and the session-date composition each time:

    T0  after the explicit `_ingest_question` (dated sessions)
    T1  after the lazy re-ingest triggered inside `answer()` (flat, undated)
    T2  after the answer completes

It calls the SAME public API the runner calls, in the SAME order, so the result
is directly attributable to the runner's call sequence. Nothing is patched.

Usage:
    python analysis/diagnose_double_ingest.py --idx 0
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_ROOT, ".env"), override=False)

from eval.longmemeval import (  # noqa: E402
    build_sessions,
    group_by_session,
    history_text,
    load_longmemeval,
    sessions_for_group,
)
from eval.phase_runner import build_backend_from_config, close_system, ingest_system  # noqa: E402
from eval.systems import build_system  # noqa: E402

#: Overwritten from --db-dir in main(). Deliberately defaults to a SCRATCH dir
#: so this diagnostic can never touch the bank a long run is building.
BANK = ""


def snapshot_bank(tag: str) -> Dict[str, Any]:
    """Note count + date composition of the LIVE bank, via a fresh connection."""
    if not os.path.exists(BANK):
        return {"tag": tag, "n": 0, "error": "no bank file"}
    c = sqlite3.connect(f"file:{BANK}?mode=ro", uri=True, timeout=30)
    try:
        rows = list(c.execute("SELECT timestamp_iso, session_id, session_date FROM notes"))
    finally:
        c.close()
    now_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {
        "tag": tag,
        "n": len(rows),
        "n_null_timestamp_iso": sum(1 for r in rows if not r[0]),
        "n_timestamp_today": sum(1 for r in rows if (r[0] or "").startswith(now_day)),
        "n_with_session_id": sum(1 for r in rows if r[1]),
        "session_ids": sorted({str(r[1]) for r in rows if r[1]}),
        "n_with_session_date": sum(1 for r in rows if r[2]),
        "distinct_session_dates": sorted({str(r[2]) for r in rows if r[2]})[:8],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--config", default="configs/models/deepseek_v41_flash.yaml")
    ap.add_argument("--db-dir", default="analysis/_scratch_bank",
                    help="SCRATCH bank dir; never point this at a live run's bank.")
    ap.add_argument("--input", default="datasets/longmemeval/longmemeval_oracle.json")
    ap.add_argument("--out", default="analysis/_double_ingest_probe.json")
    args = ap.parse_args()

    global BANK
    db_dir = os.path.join(_ROOT, args.db_dir)
    os.makedirs(db_dir, exist_ok=True)
    BANK = os.path.join(db_dir, "asem_thg.sqlite")
    print(f"scratch bank = {BANK}")

    items = load_longmemeval(os.path.join(_ROOT, args.input))
    item = items[args.idx]
    groups = group_by_session([item])
    group = groups[0]
    sessions = sessions_for_group(group)
    history = group[0]["history"]

    print(f"question idx={args.idx} id={item.question_id}")
    print(f"  haystack sessions={len(sessions)} turns={sum(len(s.turns) for s in sessions)}")
    print(f"  history list type={type(history).__name__} len={len(history)} "
          f"first_item_type={type(history[0]).__name__} first={history[0]!r}")

    backend = build_backend_from_config(os.path.join(_ROOT, args.config))
    system = build_system("ASEM-THG", os.path.join(_ROOT, args.config),
                          db_dir, backend=backend)
    out: Dict[str, Any] = {
        "idx": args.idx,
        "question_id": item.question_id,
        "n_sessions": len(sessions),
        "n_turns": sum(len(s.turns) for s in sessions),
        "history_len": len(history),
        "history_first_item": str(history[0]),
        "history_is_flat_str_list": all(isinstance(h, str) for h in history),
        "stages": [],
    }
    try:
        system.reset()
        out["stages"].append(snapshot_bank("T0_after_reset"))

        # This is exactly what scripts/run_longmemeval_eval.py:442 does.
        ingest_system(system, "ASEM-THG", list(sessions))
        out["stages"].append(snapshot_bank("T1_after_explicit_ingest_question"))
        out["_ingested_flag_after_explicit"] = bool(getattr(system, "_ingested", None))

        t0 = time.time()
        ans = system.answer(group[0]["query"], history)
        out["answer_seconds"] = round(time.time() - t0, 2)
        out["answer"] = (ans or "")[:300]
        out["stages"].append(snapshot_bank("T2_after_answer_lazy_path"))
        out["_ingested_flag_after_answer"] = bool(getattr(system, "_ingested", None))
    finally:
        close_system(system)

    for s in out["stages"]:
        print(f"  [{s['tag']}] n={s.get('n')} null_ts={s.get('n_null_timestamp_iso')} "
              f"today={s.get('n_timestamp_today')} with_sid={s.get('n_with_session_id')} "
              f"sids={s.get('session_ids')} dates={s.get('distinct_session_dates')}")

    n1 = out["stages"][1].get("n", 0)
    n2 = out["stages"][2].get("n", 0)
    out["ratio_after_answer_over_after_explicit"] = round(n2 / n1, 4) if n1 else None
    out["verdict"] = (
        "REPLACE (dated notes discarded)" if n2 and n1 and n2 <= n1
        else ("ADD (bank inflated by undated layer)" if n2 > n1 else "inconclusive")
    )
    print(f"  RATIO {n2}/{n1} = {out['ratio_after_answer_over_after_explicit']} -> {out['verdict']}")

    with open(os.path.join(_ROOT, args.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
