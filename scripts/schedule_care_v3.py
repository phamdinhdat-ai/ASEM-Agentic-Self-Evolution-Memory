"""Queue, fetch and validate the protocol-v3 LongMemEval-S shards (docs/CARE_protocol_v3_20261118.md).

Kaggle allows two concurrent GPU sessions, so shards s0..s4 are pushed as slots free up. Each finished
shard is downloaded and checked against the protocol's gate G4 before it counts:
  * event/row count equals the expected shard size, ids distinct;
  * manifest: protocol, benchmark, dataset SHA-256, shard index, worker code SHA-256 as pinned;
  * model-call count within the cap;
  * every declared arm present on every row.
A failed shard stops further submissions and is reported; nothing is retried silently.
When all five pass, scripts.summarize_care_v3 runs.

    python -m scripts.schedule_care_v3 --poll-seconds 300
"""

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
STATE = ROOT / "build/care/v3_scheduler_status.json"
PACKAGE = ROOT / "build/kaggle/care-v3"
OUTPUT = ROOT / "kaggle_outputs/care_v3"
INGEST = ROOT / "build/care/ingested_v3"
EXPECTED = [98, 110, 100, 97, 95]
DATASET_SHA = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
CODE_SHA = "837e0b259a638ed11ebad075eebf8b1d75fee6191920e35f5126983749afea7d"
MAX_CALLS = 2000
MAX_ACTIVE = 2
TERMINAL_BAD = {"ERROR", "FAILED", "CANCELLED", "CANCELED"}


def command(args: list[str], *, timeout: int = 300) -> tuple[int, str]:
    try:
        r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, errors="replace", timeout=timeout)
        return r.returncode, (r.stdout + "\n" + r.stderr).strip()
    except Exception as exc:  # network or CLI failure must not kill the scheduler
        return -1, f"{type(exc).__name__}: {str(exc)[:300]}"


def reference(shard: int) -> str:
    return f"phamdinhdatai/asem-care-v3-lmes-s{shard}"


def kernel_status(shard: int) -> str:
    code, detail = command(["kaggle", "kernels", "status", reference(shard)], timeout=90)
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", detail)
    return m.group(1) if code == 0 and m else "UNKNOWN"


def fetch_and_validate(shard: int) -> tuple[bool, str]:
    dest = OUTPUT / f"s{shard}"
    name = f"care_v3_lmes_s{shard}"
    pattern = rf"{name}/(?:status\.json|results/.*|pilot\.log|vllm\.log)"
    code, detail = command(["kaggle", "kernels", "output", reference(shard), "-p", str(dest),
                            "--file-pattern", pattern, "--quiet"], timeout=900)
    if code:
        return False, detail[:400] or "missing output"
    return validate_dir(shard, dest / name)


def validate_dir(shard: int, base: Path) -> tuple[bool, str]:
    """Gate G4 for one finished kernel directory (``base`` holds status.json and results/)."""
    report_path, status_path, calls_path = base / "results/report.json", base / "status.json", base / "results/calls.jsonl"
    if not report_path.exists() or not status_path.exists():
        return False, "missing output"
    if json.loads(status_path.read_text(encoding="utf-8")).get("state") != "complete":
        return False, "worker state is not complete"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    man, rows = report["manifest"], report["rows"]
    problems = []
    if len(rows) != EXPECTED[shard]:
        problems.append(f"{len(rows)} rows, expected {EXPECTED[shard]}")
    if len({r["id"] for r in rows}) != len(rows):
        problems.append("duplicate ids")
    if man.get("benchmark") != "longmemeval" or man.get("shard_index") != shard:
        problems.append("wrong benchmark or shard index")
    if man.get("dataset_sha256") != DATASET_SHA:
        problems.append("dataset sha mismatch")
    if man.get("code_sha256") != CODE_SHA:
        problems.append("code sha mismatch")
    if man.get("gold_in_model_context") is not False:
        problems.append("gold_in_model_context not False")
    arms = set(man.get("arms", []))
    if any(set(r["systems"]) != arms for r in rows):
        problems.append("a row misses an arm")
    n_calls = sum(1 for _ in calls_path.open(encoding="utf-8")) if calls_path.exists() else -1
    if not 0 < n_calls <= MAX_CALLS:
        problems.append(f"call count {n_calls}")
    return (not problems), ("; ".join(problems) if problems else f"{len(rows)} questions, {n_calls} calls, hashes match")


def absorb_ingested(state: dict) -> None:
    """Shards run elsewhere and validated by scripts.ingest_remote_results leave a marker; count them as done."""
    for marker in sorted(INGEST.glob("s*")) if INGEST.exists() else []:
        shard = int(marker.name[1:])
        if shard not in state["completed"]:
            state["completed"] = sorted(state["completed"] + [shard])
            state["notes"][str(shard)] = "ingested from a remote run"
        if shard in state["pending"]:
            state["pending"].remove(shard)


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE)


def step(state: dict) -> dict:
    absorb_ingested(state)
    for shard in [s for s in state["submitted"] if s not in state["completed"] and str(s) not in state["failed"]]:
        cur = kernel_status(shard)
        state["last_status"][str(shard)] = cur
        if cur == "COMPLETE":
            ok, detail = fetch_and_validate(shard)
            state["notes"][str(shard)] = detail
            if ok:
                state["completed"] = sorted(state["completed"] + [shard])
            else:
                state["failed"][str(shard)] = detail
        elif cur in TERMINAL_BAD:
            state["failed"][str(shard)] = cur
    active = [s for s in state["submitted"] if s not in state["completed"] and str(s) not in state["failed"]]
    while state["pending"] and len(active) < MAX_ACTIVE and not state["failed"]:
        shard = state["pending"][0]
        code, detail = command(["kaggle", "kernels", "push", "-p", str(PACKAGE / f"s{shard}")], timeout=240)
        if "successfully pushed" in detail:
            state["pending"].pop(0)
            state["submitted"].append(shard)
            state["last_status"][str(shard)] = "SUBMITTED"
            active.append(shard)
        elif "Maximum batch GPU session count" in detail or code == -1:
            state["notes"]["push"] = detail[-200:]   # slot not free yet, or transient network error
            break
        else:
            state["failed"][f"push-s{shard}"] = detail[-400:]
            break
    if state["failed"]:
        state["state"] = "failed"
    elif len(state["completed"]) == 5:
        code, detail = command([sys.executable, "-m", "scripts.summarize_care_v3"], timeout=1800)
        state["state"] = "complete" if code == 0 else "summary_failed"
        if code:
            state["summary_error"] = detail[-800:]
    else:
        state["state"] = "running"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    save(state)
    print(json.dumps({k: state[k] for k in ("state", "submitted", "completed", "pending", "failed")}), flush=True)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--skip", type=int, nargs="*", default=[], help="shard numbers to take out of the queue (run elsewhere)")
    args = ap.parse_args()
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
    else:   # s0 and s1 were pushed by hand on 21/11/2026 before this scheduler existed
        state = {"protocol": "care-v3-scheduler-v1", "submitted": [0, 1], "completed": [], "pending": [2, 3, 4],
                 "failed": {}, "last_status": {"0": "SUBMITTED", "1": "SUBMITTED"}, "notes": {},
                 "started_at": datetime.now(timezone.utc).isoformat()}
    for shard in args.skip:
        if shard in state["pending"]:
            state["pending"].remove(shard)
    while True:
        state = step(state)
        if args.once or state["state"] in {"complete", "failed", "summary_failed"}:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
