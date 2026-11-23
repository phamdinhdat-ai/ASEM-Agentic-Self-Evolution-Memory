"""Full LoCoMo QA paired evaluation, sharded by whole conversation.

Gold answers and evidence IDs are used only after each prediction for scoring.
Each answer is checkpointed as an append-only event for robust overnight runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

from asem.care import CareBridge
from asem.care.relation_bridge import CareRelationBridge
from scripts.run_care_locomo import _save_json, score_locomo, turns_for


SHARDS = (
    ("conv-42", "conv-30"),
    ("conv-43", "conv-44"),
    ("conv-48", "conv-47"),
    ("conv-50", "conv-41"),
    ("conv-26", "conv-49"),
)
ARMS = ("bridge_v2", "relation_temporal")
CODE_FILES = (
    "asem/care/bridge.py", "asem/care/sourcefirst.py",
    "asem/care/temporal_bridge.py", "asem/care/temporal_relations.py",
    "asem/care/relation_bridge.py", "asem/care/relation_proof.py",
    "scripts/run_care_full_locomo.py", "scripts/run_care_locomo.py",
    "kaggle/vllm-experiment-lock.json",
)


class ModelBackendError(RuntimeError):
    """Stop the run without checkpointing a prediction when the API fails."""


def selected_questions(dataset: list[dict[str, Any]], shard_index: int) -> list[dict[str, Any]]:
    if not 0 <= shard_index < len(SHARDS):
        raise ValueError("shard_index must be 0..4")
    samples = {str(sample["sample_id"]): sample for sample in dataset}
    expected = set().union(*(set(group) for group in SHARDS))
    if set(samples) != expected or sum(map(len, SHARDS)) != len(expected):
        raise ValueError("LoCoMo conversation IDs do not match locked full-run shards")
    selected = []
    for sample_id in SHARDS[shard_index]:
        for ordinal, qa in enumerate(samples[sample_id]["qa"]):
            if int(qa["category"]) not in (1, 2, 3, 4, 5):
                raise ValueError("Unexpected LoCoMo QA category")
            selected.append({"id": f"{sample_id}:{ordinal}", "sample_id": sample_id,
                "category": int(qa["category"]), "question": str(qa["question"]),
                "answer": str(qa.get("answer", "")),
                "evidence": list(qa.get("evidence") or [])})
    return selected


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _aggregate(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    values = [row["systems"][arm] for row in rows if arm in row["systems"]]
    answerable = [row["systems"][arm] for row in rows
                  if row["category"] != 5 and arm in row["systems"]]
    adversarial = [row["systems"][arm] for row in rows
                   if row["category"] == 5 and arm in row["systems"]]
    proof_status = Counter(part["status"] for value in values
                           for part in value.get("relation_proof_trace", []))
    return {"n": len(values),
        "mean_score": sum(value["f1"] for value in values) / len(values) if values else None,
        "answerable_mean_f1": sum(value["f1"] for value in answerable) / len(answerable)
                               if answerable else None,
        "adversarial_correct": sum(value["f1"] == 1 for value in adversarial),
        "adversarial_n": len(adversarial),
        "evidence_recall": sum(value["evidence_recall"] or 0 for value in answerable)
                           / len(answerable) if answerable else None,
        "calls": sum(value["new_calls"] for value in values),
        "answer_seconds": sum(value["seconds"] for value in values),
        "escalations": sum(bool(value.get("escalated")) for value in values),
        "errors": sum(value.get("error") is not None for value in values),
        "relation_proof_status": dict(proof_status)}


def run(dataset_path: Path, output: Path, backend: Any, *, shard_index: int,
        max_new_calls: int = 2500, model_info: dict[str, Any] | None = None) -> dict[str, Any]:
    dataset_path, output = Path(dataset_path), Path(output)
    payload = dataset_path.read_bytes()
    dataset = json.loads(payload)
    questions = selected_questions(dataset, shard_index)
    code_root = Path(__file__).resolve().parents[1]
    lock = json.loads((code_root / "kaggle/vllm-experiment-lock.json").read_text(encoding="utf-8"))
    manifest = {"protocol": "care-relation-full-locomo-v1",
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "model": model_info or lock["model"], "temperature": 0.0,
        "thinking": False if model_info is None else None,
        "max_reader_tokens": 224, "model_context_tokens": 8192,
        "code_sha256": {name: hashlib.sha256((code_root / name).read_bytes()).hexdigest()
                        for name in CODE_FILES},
        "shard_index": shard_index, "shard_samples": list(SHARDS[shard_index]),
        "question_ids": [item["id"] for item in questions],
        "arms": list(ARMS), "scorer": "LoCoMo token-F1 and category-5 refusal semantics",
        "full_dataset_adapter": "raw text turns with session dates and available image captions",
        "gold_in_model_context": False, "official_multimodal_run": False}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Refusing to resume a different full-run manifest")
    _save_json(manifest_path, manifest)
    events_path = output / "events.jsonl"
    events = _load_events(events_path)
    done = {(event["id"], event["arm"]) for event in events}
    if len(done) != len(events):
        raise ValueError("Duplicate checkpoint event")
    samples = {sample["sample_id"]: sample for sample in dataset}
    memories: dict[str, tuple[CareBridge, CareRelationBridge]] = {}
    for item in questions:
        sample_id = item["sample_id"]
        if sample_id not in memories:
            turns = turns_for(samples[sample_id]["conversation"])
            memories[sample_id] = CareBridge(turns), CareRelationBridge(turns)
        bridge, relation = memories[sample_id]
        arm_order = (ARMS if int(hashlib.sha256(item["id"].encode()).hexdigest(), 16) % 2
                     else ARMS[::-1])
        for arm in arm_order:
            if (item["id"], arm) in done:
                continue
            if len(getattr(backend, "calls", [])) >= max_new_calls:
                raise RuntimeError("CARE full-run model-call cap reached")
            started = time.monotonic()
            before = len(getattr(backend, "calls", []))
            try:
                result = (bridge if arm == "bridge_v2" else relation).answer(
                    item["question"], backend)
                error = None
            except Exception as exc:
                if isinstance(exc, ModelBackendError) or "model-call cap" in str(exc):
                    raise
                result = {"answer": "", "source_ids": [], "bundle_ids": [],
                          "candidate_ids": []}
                error = type(exc).__name__ + ": " + str(exc)[:180]
            result.pop("raw_reader_response", None)
            gold_evidence = set(item["evidence"])
            value = {**result, "error": error,
                "f1": score_locomo(result["answer"], item["answer"], item["category"]),
                "evidence_recall": (len(gold_evidence & set(result["bundle_ids"]))
                                    / len(gold_evidence) if gold_evidence else None),
                "new_calls": len(getattr(backend, "calls", [])) - before,
                "seconds": time.monotonic() - started}
            event = {"id": item["id"], "arm": arm, "value": value}
            _append_jsonl(events_path, event)
            events.append(event)
            done.add((item["id"], arm))
            if len(events) % 25 == 0:
                print(json.dumps({"shard": shard_index, "completed_predictions": len(events),
                                  "expected_predictions": len(questions) * len(ARMS)}), flush=True)
    by_id = {item["id"]: {key: item[key] for key in
             ("id", "sample_id", "category", "question")} | {"systems": {}}
             for item in questions}
    for event in events:
        by_id[event["id"]]["systems"][event["arm"]] = event["value"]
    rows = [by_id[item["id"]] for item in questions]
    if any(set(row["systems"]) != set(ARMS) for row in rows):
        raise RuntimeError("Incomplete paired shard")
    aggregate = {arm: _aggregate(rows, arm) for arm in ARMS}
    by_category = {str(category): {arm: _aggregate(
        [row for row in rows if row["category"] == category], arm) for arm in ARMS}
        for category in (1, 2, 3, 4, 5)}
    deltas = [row["systems"]["relation_temporal"]["f1"]
              - row["systems"]["bridge_v2"]["f1"] for row in rows]
    paired = {"mean_delta": sum(deltas) / len(deltas),
              "wins": sum(delta > 0 for delta in deltas),
              "losses": sum(delta < 0 for delta in deltas),
              "ties": sum(delta == 0 for delta in deltas)}
    report = {"manifest": manifest, "aggregate": aggregate,
              "by_category": by_category, "paired": paired, "rows": rows}
    _save_json(output / "rows.json", rows)
    _save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--max-new-calls", type=int, default=2500)
    # Accepted for compatibility with the shared Kaggle launcher.
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--start-per-category", type=int, default=0)
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="asem-atlas-qwen3-8b")
    parser.add_argument("--no-chat-template-kwargs", action="store_true")
    args = parser.parse_args()
    from openai import APIError, OpenAI

    class Counted:
        def __init__(self):
            self.client = OpenAI(base_url=args.base_url,
                                 api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                                 timeout=180)
            self.log_path = args.output / "calls.jsonl"
            self.calls = _load_events(self.log_path)

        def generate(self, prompt: str, **kwargs: Any) -> str:
            if len(self.calls) >= args.max_new_calls:
                raise RuntimeError("CARE full-run model-call cap reached")
            request = {"model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": float(kwargs.get("temperature", 0.0)),
                "max_tokens": int(kwargs.get("max_tokens", 224))}
            if not args.no_chat_template_kwargs:
                request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            try:
                response = self.client.chat.completions.create(**request)
            except APIError as exc:
                raise ModelBackendError(f"Model API failed: {type(exc).__name__}: {exc}") from exc
            choice = response.choices[0]
            answer = choice.message.content or ""
            receipt = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "finish_reason": choice.finish_reason,
                       "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                       "completion_tokens": getattr(response.usage, "completion_tokens", None)}
            sequence = len(self.calls) + 1
            _save_json(args.output / "raw_calls" / f"{sequence:05d}.json",
                       {"prompt": prompt, "response": answer, **receipt})
            _append_jsonl(self.log_path, receipt)
            self.calls.append(receipt)
            return answer

    backend = Counted()
    local_default = (args.model == "asem-atlas-qwen3-8b" and
                     args.base_url == "http://127.0.0.1:8000/v1")
    model_info = None if local_default else {
        "repo": args.model, "provider": "openai-compatible",
        "endpoint_host": urlparse(args.base_url).hostname}
    report = run(args.dataset, args.output, backend, shard_index=args.shard_index,
                 max_new_calls=args.max_new_calls, model_info=model_info)
    _save_json(args.output / "calls.json", backend.calls)
    print(json.dumps({"shard": args.shard_index, "aggregate": report["aggregate"],
                      "paired": report["paired"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
