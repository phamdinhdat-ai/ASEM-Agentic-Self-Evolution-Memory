"""Run a bounded CARE versus BM25 LoCoMo pilot on Kaggle T4 x2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from scripts.run_kaggle_vllm import create_runtime, server_command, wait_server

ROOT = Path(__file__).resolve().parents[1]


def _save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _download_locked(lock: dict, cache: Path, name: str = "locomo10.json") -> Path:
    spec = lock["sources"][name]
    target = cache / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != spec["sha256"]:
        temporary = target.with_suffix(".part")
        urllib.request.urlretrieve(spec["url"], temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("LoCoMo source checksum mismatch")
        temporary.replace(target)
    return target


def run(*, output: str = "/kaggle/working/care_sourcefirst",
        cache: str = "/kaggle/temp/care_sourcefirst", per_category: int = 4,
        max_new_calls: int = 100, hf_secret_label: str = "HF_TOKEN",
        start_per_category: int = 0, categories: tuple[int, ...] = (1, 2, 3, 4),
        worker_module: str = "scripts.run_care_locomo",
        worker_extra_args: tuple[str, ...] = (),
        server_context_tokens: int = 8192, gpu_memory_utilization: str | None = None,
        extra_packages: tuple[str, ...] = (),
        dataset_name: str = "locomo10.json") -> dict:
    output_path, cache_path = Path(output), Path(cache)
    output_path.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / "kaggle/vllm-experiment-lock.json").read_text(encoding="utf-8"))
    started, process = time.monotonic(), None
    _save(output_path / "status.json", {"state": "installing", "per_category": per_category,
        "start_per_category": start_per_category, "categories": categories})
    try:
        if not os.environ.get("HF_TOKEN"):
            try:
                from kaggle_secrets import UserSecretsClient
                token = UserSecretsClient().get_secret(hf_secret_label)
                if token:
                    os.environ["HF_TOKEN"] = token
            except Exception:
                pass  # Qwen3-8B is public; anonymous download is valid.
        runtime = create_runtime(cache_path / "venv", output_path / "environment")
        # The upstream LoCoMo F1 uses NLTK's PorterStemmer. Keep the scorer
        # separate from the model-serving dependencies.
        with (output_path / "nltk-install.log").open("w", encoding="utf-8") as install_log:
            subprocess.run([sys.executable, "-m", "pip", "--python", str(runtime),
                "install", "--no-cache-dir", "nltk==3.9.1", *extra_packages], check=True,
                stdout=install_log, stderr=subprocess.STDOUT)
        dataset = _download_locked(lock, cache_path / "datasets", dataset_name)
        os.environ["HF_HOME"] = str(cache_path / "huggingface")
        command = server_command(runtime, lock, context_tokens=server_context_tokens,
                                 utilization=gpu_memory_utilization)
        _save(output_path / "model.json", {"model": lock["model"],
            "server_command": command, "thinking": False,
            "context_tokens": server_context_tokens})
        _save(output_path / "status.json", {"state": "starting_model"})
        with (output_path / "vllm.log").open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_server(process, "http://127.0.0.1:8000", timeout=900,
                        model_name=lock["model"].get("served_name", "asem-atlas-qwen3-8b"))
            _save(output_path / "status.json", {"state": "running_pilot"})
            worker = [str(runtime), "-m", worker_module, "--dataset", str(dataset),
                      "--output", str(output_path / "results"), "--per-category", str(per_category),
                      "--start-per-category", str(start_per_category),
                      "--categories", *(str(category) for category in categories),
                      "--max-new-calls", str(max_new_calls), *worker_extra_args]
            with (output_path / "pilot.log").open("w", encoding="utf-8") as pilot_log:
                finished = subprocess.run(worker, cwd=ROOT, env=os.environ.copy(),
                    stdout=pilot_log, stderr=subprocess.STDOUT, timeout=3 * 60 * 60)
            if finished.returncode:
                raise RuntimeError(f"CARE pilot worker exited {finished.returncode}; see pilot.log")
        report = json.loads((output_path / "results/report.json").read_text(encoding="utf-8"))
        status = {"state": "complete", "aggregate": report["aggregate"],
                  "n_rows": len(report["rows"]), "seconds": time.monotonic() - started,
                  "output": str(output_path / "results")}
        _save(output_path / "status.json", status)
        print(json.dumps(status, indent=2), flush=True)
        return status
    except Exception as exc:
        _save(output_path / "status.json", {"state": "failed", "error": type(exc).__name__,
            "detail": str(exc)[:500], "seconds": time.monotonic() - started})
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


if __name__ == "__main__":
    run()
