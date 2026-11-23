"""Run the Qwen3-4B pilot and cohort after LongMemEval terminates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "build/care/longmemeval_scheduler_status.json"
STATE = ROOT / "build/care/second_backbone_4b_scheduler_status.json"
PACKAGE = ROOT / "build/kaggle"
OUTPUT = ROOT / "build/care/second_backbone_4b"


def command(args: list[str], *, timeout: int = 300) -> tuple[int, str]:
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                errors="replace", timeout=timeout)
        return result.returncode, (result.stdout + "\n" + result.stderr).strip()
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {str(exc)[:300]}"


def reference(pilot: bool) -> str:
    return "phamdinhdatai/asem-care-second-backbone-4b" + ("-pilot" if pilot else "")


def status(pilot: bool) -> str:
    code, detail = command(["kaggle", "kernels", "status", reference(pilot)], timeout=90)
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", detail)
    return match.group(1) if code == 0 and match else "UNKNOWN"


def fetch(pilot: bool) -> tuple[bool, str]:
    destination = OUTPUT / ("pilot" if pilot else "full")
    name = "care_second_backbone_4b" + ("_pilot" if pilot else "")
    pattern = rf"{name}/(?:status\.json|results/report\.json|pilot\.log|vllm\.log)"
    code, detail = command(["kaggle", "kernels", "output", reference(pilot),
                            "-p", str(destination), "--file-pattern", pattern,
                            "--quiet"], timeout=300)
    base = destination / name
    status_path, report_path = base / "status.json", base / "results/report.json"
    if code or not status_path.exists() or not report_path.exists():
        return False, detail[:500] or "Missing output"
    run_status = json.loads(status_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = 2 if pilot else 50
    if run_status.get("state") != "complete" or len(report["rows"]) != expected:
        return False, f"Incomplete run: {run_status.get('state')}, {len(report['rows'])} rows"
    if report["manifest"]["model"]["repo"] != "Qwen/Qwen3-4B":
        return False, "Wrong backbone"
    return True, f"{expected} questions complete"


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def step(state: dict) -> dict:
    prior_state = (json.loads(PRIOR.read_text(encoding="utf-8"))["state"]
                   if PRIOR.exists() else "missing")
    state["prior_state"] = prior_state
    if prior_state in {"complete", "failed", "merge_failed"} and not state["failed"]:
        if not state["pilot_submitted"]:
            code, detail = command(["kaggle", "kernels", "push", "-p",
                                    str(PACKAGE / "care-second-backbone-4b-pilot")], timeout=180)
            if code or "successfully pushed" not in detail:
                state["failed"]["pilot"] = detail[:500]
            else:
                state["pilot_submitted"] = True
                state["pilot_status"] = "SUBMITTED"
        elif not state["pilot_complete"]:
            state["pilot_status"] = status(True)
            if state["pilot_status"] == "COMPLETE":
                okay, detail = fetch(True)
                if okay:
                    state["pilot_complete"] = True
                else:
                    state["failed"]["pilot"] = detail
            elif state["pilot_status"] in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                state["failed"]["pilot"] = state["pilot_status"]

        if state["pilot_complete"] and not state["full_submitted"]:
            code, detail = command(["kaggle", "kernels", "push", "-p",
                                    str(PACKAGE / "care-second-backbone-4b")], timeout=180)
            if code or "successfully pushed" not in detail:
                state["failed"]["full"] = detail[:500]
            else:
                state["full_submitted"] = True
                state["full_status"] = "SUBMITTED"
        elif state["full_submitted"] and not state["full_complete"]:
            state["full_status"] = status(False)
            if state["full_status"] == "COMPLETE":
                okay, detail = fetch(False)
                if okay:
                    state["full_complete"] = True
                else:
                    state["failed"]["full"] = detail
            elif state["full_status"] in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                state["failed"]["full"] = state["full_status"]

    if state["failed"]:
        state["state"] = "failed"
    elif state["full_complete"]:
        code, detail = command([sys.executable, "-m", "scripts.summarize_care_second_backbone"])
        state["state"] = "complete" if code == 0 else "merge_failed"
        if code:
            state["merge_error"] = detail[-1000:]
    else:
        state["state"] = "running"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    save(state)
    print(json.dumps({key: state[key] for key in
                      ("state", "prior_state", "pilot_submitted", "pilot_status",
                       "pilot_complete", "full_submitted", "full_status",
                       "full_complete", "failed")}), flush=True)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
    else:
        state = {"protocol": "care-second-backbone-4b-scheduler-v1",
                 "state": "running", "prior_state": "missing",
                 "pilot_submitted": False, "pilot_status": "NOT_SUBMITTED",
                 "pilot_complete": False, "full_submitted": False,
                 "full_status": "NOT_SUBMITTED", "full_complete": False,
                 "failed": {}, "started_at": datetime.now(timezone.utc).isoformat()}
    while True:
        state = step(state)
        if args.once or state["state"] in {"failed", "complete", "merge_failed"}:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
