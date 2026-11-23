"""Run the full paired CARE LoCoMo evaluation with the configured API backend.

The five shards checkpoint every prediction. Re-running this command resumes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse

from scripts.run_care_full_locomo import SHARDS, selected_questions
from scripts.run_care_locomo import _save_json


ROOT = Path(__file__).resolve().parents[1]


def _load_env(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def _preflight(base_url: str, api_key: str, model: str) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Reply with exactly OK."}],
        temperature=0, max_tokens=20)
    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError("Model preflight returned no text")


def run(*, dataset: Path, output: Path, env_file: Path,
        max_new_calls: int = 2500) -> dict:
    config = _load_env(env_file)
    base_url = os.environ.get("OPENAI_BASE_URL") or config.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY") or config.get("OPENAI_API_KEY", "")
    model = os.environ.get("LLM_MODEL") or config.get("LLM_MODEL", "")
    if not base_url or not api_key or not model:
        raise ValueError("OPENAI_BASE_URL, OPENAI_API_KEY and LLM_MODEL are required")

    payload = dataset.read_bytes()
    records = json.loads(payload)
    if len(records) != len(SHARDS) * 2:
        raise ValueError("Expected the complete ten-conversation LoCoMo source")
    counts = [len(selected_questions(records, i)) for i in range(len(SHARDS))]
    identity = {"dataset_sha256": hashlib.sha256(payload).hexdigest(),
                "model": model, "endpoint_host": urlparse(base_url).hostname,
                "question_counts": counts, "max_new_calls_per_shard": max_new_calls}
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise ValueError("Refusing to resume a different API or dataset run")
    status = {"state": "preflight", "identity": identity, "completed_shards": []}
    _save_json(status_path, status)
    try:
        _preflight(base_url, api_key, model)
    except Exception as exc:
        code = getattr(exc, "status_code", None)
        status.update(state="blocked", error=type(exc).__name__,
                      http_status=code, detail=str(exc)[:500])
        _save_json(status_path, status)
        raise

    env = os.environ.copy()
    env.update(OPENAI_BASE_URL=base_url, OPENAI_API_KEY=api_key, LLM_MODEL=model)
    for index in range(len(SHARDS)):
        base = output / f"shard-{index}" / f"care_relation_full_s{index}"
        shard_status = base / "status.json"
        if shard_status.exists() and json.loads(
                shard_status.read_text(encoding="utf-8")).get("state") == "complete":
            status["completed_shards"].append(index)
            continue
        base.mkdir(parents=True, exist_ok=True)
        status.update(state="running", current_shard=index)
        _save_json(status_path, status)
        _save_json(shard_status, {"state": "running", "shard": index})
        command = [sys.executable, "-m", "scripts.run_care_full_locomo",
                   "--dataset", str(dataset), "--output", str(base / "results"),
                   "--shard-index", str(index), "--max-new-calls", str(max_new_calls),
                   "--base-url", base_url, "--model", model,
                   "--no-chat-template-kwargs"]
        with (base / "worker.log").open("a", encoding="utf-8") as log:
            finished = subprocess.run(command, cwd=ROOT, env=env,
                                      stdout=log, stderr=subprocess.STDOUT)
        if finished.returncode:
            _save_json(shard_status, {"state": "failed", "shard": index,
                                      "exit_code": finished.returncode})
            status.update(state="failed", exit_code=finished.returncode)
            _save_json(status_path, status)
            raise RuntimeError(f"Shard {index} failed; see {base / 'worker.log'}")
        _save_json(shard_status, {"state": "complete", "shard": index})
        status["completed_shards"].append(index)
        _save_json(status_path, status)

    merged = output.parent / (output.name + "_merged.json")
    command = [sys.executable, "-m", "scripts.merge_care_full_locomo",
               "--dataset", str(dataset), "--root", str(output),
               "--output", str(merged)]
    with (output / "merge.log").open("w", encoding="utf-8") as log:
        finished = subprocess.run(command, cwd=ROOT, stdout=log,
                                  stderr=subprocess.STDOUT)
    if finished.returncode:
        status["state"] = "merge_failed"
        _save_json(status_path, status)
        raise RuntimeError(f"Merge failed; see {output / 'merge.log'}")
    status.update(state="complete", merged_report=str(merged))
    status.pop("current_shard", None)
    _save_json(status_path, status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "datasets/locomo/locomo10.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "build/care/relation_full_api")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--max-new-calls", type=int, default=2500)
    args = parser.parse_args()
    result = run(dataset=args.dataset.resolve(), output=args.output.resolve(),
                 env_file=args.env_file.resolve(), max_new_calls=args.max_new_calls)
    print(json.dumps(result, indent=2, ensure_ascii=False))
