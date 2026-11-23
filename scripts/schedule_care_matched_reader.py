"""Complete five private CARE matched-reader Kaggle shards, two at a time."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from scripts.run_care_full_locomo import SHARDS


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "build/care/matched_reader_scheduler_status.json"
PACKAGE = ROOT / "build/kaggle/care-matched-reader"
OUTPUT = ROOT / "build/care/matched_reader"


def command(args: list[str], *, timeout: int = 300) -> tuple[int, str]:
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                errors="replace", timeout=timeout)
        return result.returncode, (result.stdout + "\n" + result.stderr).strip()
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {str(exc)[:300]}"


def reference(shard: int) -> str:
    return f"phamdinhdatai/asem-care-matched-reader-shard-{shard}"


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def fetch(shard: int) -> tuple[bool, str]:
    destination = OUTPUT / f"shard-{shard}"
    name = f"care_matched_reader_s{shard}"
    pattern = rf"{name}/(?:status\.json|results/report\.json|pilot\.log)"
    code, detail = command(["kaggle", "kernels", "output", reference(shard),
                            "-p", str(destination), "--file-pattern", pattern,
                            "--quiet"], timeout=300)
    base = destination / name
    status_path, report_path = base / "status.json", base / "results/report.json"
    if code or not status_path.exists() or not report_path.exists():
        return False, detail[:500] or "Missing shard output"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if status.get("state") != "complete" or report["manifest"]["shard_index"] != shard:
        return False, "Incomplete shard or wrong manifest"
    if {row["sample_id"] for row in report["rows"]} != set(SHARDS[shard]):
        return False, "Shard conversation IDs mismatch"
    if len(report["rows"]) != status["n_rows"]:
        return False, "Shard question count mismatch"
    return True, f"{len(report['rows'])} questions complete"


def step(state: dict) -> dict:
    for shard in list(state["submitted"]):
        if shard in state["completed"]:
            continue
        code, detail = command(["kaggle", "kernels", "status", reference(shard)], timeout=90)
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", detail)
        status = match.group(1) if code == 0 and match else "UNKNOWN"
        state["last_status"][str(shard)] = status
        if status == "COMPLETE":
            okay, message = fetch(shard)
            if okay:
                state["completed"].append(shard)
                state["completed"].sort()
            else:
                state["failed"][str(shard)] = message
        elif status in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
            state["failed"][str(shard)] = detail[:500]

    active = [shard for shard in state["submitted"] if shard not in state["completed"]
              and str(shard) not in state["failed"]]
    while len(active) < 2 and state["pending"] and not state["failed"]:
        shard = state["pending"][0]
        code, detail = command(["kaggle", "kernels", "push", "-p",
                                str(PACKAGE / f"shard-{shard}")], timeout=180)
        if code or "successfully pushed" not in detail:
            state["failed"][str(shard)] = detail[:500]
            break
        state["pending"].pop(0)
        state["submitted"].append(shard)
        active.append(shard)
        state["last_status"][str(shard)] = "SUBMITTED"
    if state["failed"]:
        state["state"] = "failed"
    elif len(state["completed"]) == 5:
        code, detail = command([sys.executable, "-m",
                                "scripts.summarize_care_matched_reader"], timeout=300)
        state["state"] = "complete" if code == 0 else "merge_failed"
        if code:
            state["merge_error"] = detail[-1000:]
    else:
        state["state"] = "running"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    save(state)
    print(json.dumps({key: state[key] for key in
                      ("state", "submitted", "completed", "pending", "failed",
                       "last_status")}), flush=True)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
    else:
        state = {"protocol": "care-matched-reader-scheduler-v1", "state": "running",
                 "submitted": [0, 1], "completed": [], "pending": [2, 3, 4],
                 "failed": {}, "last_status": {},
                 "started_at": datetime.now(timezone.utc).isoformat()}
    while True:
        state = step(state)
        if args.once or state["state"] in {"failed", "complete", "merge_failed"}:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
