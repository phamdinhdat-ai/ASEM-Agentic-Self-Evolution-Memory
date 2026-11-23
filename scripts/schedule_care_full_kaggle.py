"""Overnight scheduler for five private Kaggle GPU shards (two active slots)."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "build/kaggle/care-relation-full"
OUTPUT = ROOT / "build/care/relation_full"
STATE = ROOT / "build/care/relation_full_scheduler_status.json"
OWNER = "phamdinhdatai"
ACTIVE = {"QUEUED", "RUNNING", "INITIALIZING", "PENDING", "PREPARING"}


def _save(value: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def _command(args: list[str], *, timeout: int = 120) -> tuple[int, str]:
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                errors="replace", timeout=timeout)
        return result.returncode, (result.stdout + "\n" + result.stderr).strip()
    except Exception as exc:
        return -1, type(exc).__name__ + ": " + str(exc)[:300]


def _ref(index: int) -> str:
    return f"{OWNER}/asem-care-relation-full-locomo-shard-{index}"


def _status(index: int) -> tuple[str | None, str]:
    code, output = _command(["kaggle", "kernels", "status", _ref(index)], timeout=90)
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
    return (match.group(1) if code == 0 and match else None), output[:500]


def _download(index: int) -> tuple[bool, str]:
    destination = OUTPUT / f"shard-{index}"
    pattern = rf"care_relation_full_s{index}/(?:status\.json|results/report\.json|pilot\.log)"
    code, output = _command(["kaggle", "kernels", "output", _ref(index),
                             "-p", str(destination), "--file-pattern", pattern,
                             "--quiet"], timeout=300)
    base = destination / f"care_relation_full_s{index}"
    status_path = base / "status.json"
    report_path = base / "results/report.json"
    if code != 0 or not status_path.exists() or not report_path.exists():
        return False, output[:500] or "Missing status/report after download"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return status.get("state") == "complete", json.dumps(status)[:500]


def _initial() -> dict:
    return {"protocol": "care-relation-full-scheduler-v1",
            "submitted": [0, 1], "completed": [], "failed": {},
            "pending": [2, 3, 4], "last_status": {},
            "started_at": datetime.now(timezone.utc).isoformat(),
            "state": "running"}


def step(state: dict) -> dict:
    active = 0
    for index in list(state["submitted"]):
        if index in state["completed"] or str(index) in state["failed"]:
            continue
        status, detail = _status(index)
        state["last_status"][str(index)] = status or "UNKNOWN"
        if status == "COMPLETE":
            success, download_detail = _download(index)
            if success:
                state["completed"].append(index)
                state["completed"].sort()
            else:
                local_status = OUTPUT / f"shard-{index}" / f"care_relation_full_s{index}" / "status.json"
                if local_status.exists() and json.loads(local_status.read_text(
                        encoding="utf-8")).get("state") == "failed":
                    state["failed"][str(index)] = download_detail
                else:
                    state.setdefault("last_download_error", {})[str(index)] = download_detail
        elif status in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
            state["failed"][str(index)] = status + ": " + detail
        else:
            # An unknown status is counted as active to avoid overbooking GPU.
            active += 1
    while active < 2 and state["pending"]:
        index = state["pending"][0]
        code, output = _command(["kaggle", "kernels", "push", "-p",
                                 str(PACKAGE / f"shard-{index}")], timeout=180)
        if code == 0 and "successfully pushed" in output:
            state["pending"].pop(0)
            state["submitted"].append(index)
            state["submitted"].sort()
            state["last_status"][str(index)] = "SUBMITTED"
            active += 1
        else:
            state["last_push_error"] = output[:500]
            break
    if len(state["completed"]) == 5:
        code, output = _command([sys.executable, "-m", "scripts.merge_care_full_locomo",
                                 "--root", str(OUTPUT)], timeout=300)
        if code == 0:
            state["state"] = "complete"
            state["merged_report"] = str(ROOT / "build/care/relation_full_merged.json")
        else:
            state["state"] = "merge_failed"
            state["merge_error"] = output[-1500:]
    elif state["failed"]:
        state["state"] = "shard_failed"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save(state)
    print(json.dumps({"state": state["state"], "submitted": state["submitted"],
                      "completed": state["completed"], "pending": state["pending"],
                      "last_status": state["last_status"],
                      "failed": state["failed"]}), flush=True)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=90)
    parser.add_argument("--max-hours", type=int, default=30)
    args = parser.parse_args()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else _initial()
    deadline = datetime.now(timezone.utc) + timedelta(hours=args.max_hours)
    while True:
        state = step(state)
        if args.once or state["state"] in {"complete", "merge_failed"}:
            break
        if datetime.now(timezone.utc) >= deadline:
            state["state"] = "scheduler_timeout"
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save(state)
            break
        time.sleep(args.poll_seconds)
