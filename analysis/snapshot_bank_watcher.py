"""Snapshot the LongMemEval ASEM-THG bank once per answered question (READ-ONLY).

Why this exists
---------------
`ASEMTHGSystem.reset()` (eval/systems.py:698-700) calls
`pipeline.memory_bank.clear()` before every LongMemEval question, so
`<db_dir>/asem_thg.sqlite` holds only the LAST question's graph. A per-question
graph-quality table therefore cannot be recovered after the run — it has to be
sampled while the run is in flight.

This watcher is deliberately OUTSIDE the pipeline: it only observes
`data/benchmarks/results/longmemeval/<tag>/preds/*__ASEM-THG.jsonl` (the
resumable prediction file the runner flushes after every question) and, when a
new row appears, copies the live bank file with SQLite's own backup API so the
copy is consistent even while the writer holds the DB.

Usage
-----
    python analysis/snapshot_bank_watcher.py \
        --tag gb_full --db-dir data/benchmarks/longmemeval_banks \
        --out-dir data/benchmarks/longmemeval_banks/gb_full_snapshots
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple


def _preds_path(tag: str, results_root: str) -> Optional[str]:
    pattern = os.path.join(results_root, tag, "preds", f"{tag}__*__ASEM-THG.jsonl")
    found = sorted(glob.glob(pattern))
    return found[0] if found else None


def _read_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A partially-flushed final line; ignore it.
                    continue
    except OSError:
        pass
    return rows


def _snapshot(src: str, dst: str) -> bool:
    """Copy a live SQLite DB consistently (safe while another process writes)."""
    if not os.path.exists(src):
        return False
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        source = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30)
        try:
            target = sqlite3.connect(tmp)
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        os.replace(tmp, dst)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  !! snapshot failed for {src}: {exc}", file=sys.stderr, flush=True)
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def _n_notes(path: str) -> int:
    try:
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return int(c.execute("SELECT COUNT(*) FROM notes").fetchone()[0])
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        return -1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="gb_full")
    p.add_argument("--db-dir", default="data/benchmarks/longmemeval_banks")
    p.add_argument("--results-root", default="data/benchmarks/results/longmemeval")
    p.add_argument("--out-dir", default="data/benchmarks/longmemeval_banks/gb_full_snapshots")
    p.add_argument("--poll", type=float, default=5.0)
    p.add_argument("--idle-timeout", type=float, default=1800.0,
                   help="Stop after this many seconds with no new prediction row.")
    p.add_argument("--max-questions", type=int, default=0)
    p.add_argument("--once", action="store_true", help="Take at most one snapshot then exit")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    index_path = os.path.join(args.out_dir, "_index.json")
    index: Dict[str, Any] = {"tag": args.tag, "snapshots": [], "double_ingest": []}
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as fh:
                index = json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass

    bank = os.path.join(args.db_dir, "asem_thg.sqlite")
    seen: Set[int] = {int(s["idx"]) for s in index.get("snapshots", [])}
    preds: Optional[str] = None
    last_new = time.time()
    printed_header = False

    print(f"watching tag={args.tag} bank={bank} -> {args.out_dir}", flush=True)

    while True:
        if preds is None:
            preds = _preds_path(args.tag, args.results_root)
            if preds is None:
                if time.time() - last_new > args.idle_timeout:
                    print("no prediction file appeared; giving up", flush=True)
                    return 1
                time.sleep(args.poll)
                continue
            if not printed_header:
                print(f"preds = {preds}", flush=True)
                printed_header = True

        rows = _read_rows(preds)
        new_rows = [r for r in rows if int(r.get("idx", -1)) not in seen]
        for row in new_rows:
            idx = int(row["idx"])
            qid = str(row.get("question_id", f"q{idx}"))
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", qid)

            # RACE, and how it is handled honestly.
            #
            # The runner flushes the prediction row and then immediately calls
            # `reset()` for the next question, which CLEARS the bank. Between
            # those two events the bank still holds this question's graph, but
            # the window is short and this watcher polls from outside the
            # process, so a capture can land AFTER the clear and read 0 notes.
            #
            # We therefore try a few times in quick succession and keep the
            # FIRST non-empty read, recording how many attempts it took. A
            # capture that never sees a non-empty bank is recorded with
            # `n_notes: 0` and `race: "missed"` rather than being silently
            # dropped, so the index never pretends to have data it lacks.
            n = -1
            ok = False
            attempts = 0
            for attempt in range(3):
                attempts = attempt + 1
                dst = os.path.join(args.out_dir, f"q_{idx:04d}_{safe}.sqlite")
                ok = _snapshot(bank, dst)
                n = _n_notes(dst) if ok else -1
                if n > 0:
                    break
                time.sleep(0.35)

            entry = {
                "idx": idx,
                "question_id": qid,
                "question_type": row.get("question_type"),
                "snapshot": os.path.join(args.out_dir, f"q_{idx:04d}_{safe}.sqlite") if ok else None,
                "n_notes": n,
                "capture_attempts": attempts,
                "race": "ok" if n > 0 else "missed",
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            index.setdefault("snapshots", []).append(entry)
            seen.add(idx)
            last_new = time.time()
            flag = "" if n > 0 else "   <- RACE: bank already cleared, no graph captured"
            print(f"  [q{idx}] {qid} -> {n} notes{flag}", flush=True)
            with open(index_path, "w", encoding="utf-8") as fh:
                json.dump(index, fh, indent=2)
            if args.once or (args.max_questions and len(seen) >= args.max_questions):
                print("done", flush=True)
                return 0

        if time.time() - last_new > args.idle_timeout:
            print(f"idle for {args.idle_timeout}s with {len(seen)} snapshots; exiting", flush=True)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
