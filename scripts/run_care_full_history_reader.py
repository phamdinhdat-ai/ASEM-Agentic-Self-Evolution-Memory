"""One-pass CARE reader over every original turn in a 40K model window.

This is a fixed, category-balanced 50-question full-history sensitivity
analysis, not the complete 1,986-question experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

from asem.care import CareBridge
from scripts.run_care_context_baselines import MODEL
from scripts.run_care_full_locomo import (_append_jsonl, _load_events,
                                          selected_questions, SHARDS)
from scripts.run_care_locomo import _save_json, score_locomo, turns_for


WINDOW = 40960
ANSWER_CAP = 224
EVIDENCE_CHARS = 500000


def select_audit_questions(dataset: list[dict], shard_index: int,
                           per_sample: int = 5) -> list[dict]:
    rows = selected_questions(dataset, shard_index)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["sample_id"], row["category"])].append(row)
    selected = []
    for sample_id in SHARDS[shard_index]:
        per_conversation = []
        for category in range(1, 6):
            group = groups[(sample_id, category)]
            group.sort(key=lambda row: hashlib.sha256(
                ("care-full-history-40k-v1:" + row["id"]).encode()).hexdigest())
        position = 0
        while len(per_conversation) < per_sample:
            changed = False
            for category in range(1, 6):
                group = groups[(sample_id, category)]
                if len(group) > position and len(per_conversation) < per_sample:
                    per_conversation.append(group[position])
                    changed = True
            if not changed:
                raise ValueError(f"Too few questions for {sample_id}")
            position += 1
        selected.extend(per_conversation)
    return selected


def token_count(tokenizer, prompt: str) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, return_dict=True,
        add_generation_prompt=True, enable_thinking=False)
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return len(ids)


def run(dataset_path: Path, output: Path, *, shard_index: int,
        question_limit: int = 0, max_new_calls: int = 20) -> dict:
    from openai import OpenAI
    from transformers import AutoTokenizer

    dataset_bytes = dataset_path.read_bytes()
    dataset = json.loads(dataset_bytes)
    questions = select_audit_questions(dataset, shard_index)
    if question_limit:
        questions = questions[:question_limit]
    tokenizer = AutoTokenizer.from_pretrained(MODEL["repo"],
                                               revision=MODEL["revision"], use_fast=True)
    manifest = {"protocol": "care-full-history-40k-v1",
                "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "model": MODEL, "shard_index": shard_index,
                "shard_samples": list(SHARDS[shard_index]),
                "question_ids": [row["id"] for row in questions],
                "selection": "five SHA-256 selected questions per conversation, round-robin categories",
                "question_limit": question_limit,
                "window_tokens": WINDOW, "answer_cap": ANSWER_CAP,
                "reader": "CareBridge._reader with all original turns and no repair",
                "gold_in_model_context": False}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Refusing to resume a different full-history manifest")
    _save_json(manifest_path, manifest)
    events_path = output / "events.jsonl"
    events = _load_events(events_path)
    done = {event["id"] for event in events}
    if len(done) != len(events):
        raise ValueError("Duplicate checkpoint event")
    samples = {sample["sample_id"]: sample for sample in dataset}
    bridges = {}
    client = OpenAI(base_url="http://127.0.0.1:8000/v1",
                    api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                    timeout=300, max_retries=2)
    new_calls = 0

    class CheckedBackend:
        def __init__(self):
            self.receipt = None

        def generate(self, prompt: str, **kwargs):
            nonlocal new_calls
            if new_calls >= max_new_calls:
                raise RuntimeError("Full-history model call cap reached")
            local_tokens = token_count(tokenizer, prompt)
            if local_tokens + ANSWER_CAP > WINDOW:
                raise ValueError(f"Full original history exceeds {WINDOW}-token window")
            response = client.chat.completions.create(
                model="asem-atlas-qwen3-8b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=ANSWER_CAP,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            self.receipt = {"prompt_tokens_local": local_tokens,
                            "prompt_tokens_server": getattr(response.usage, "prompt_tokens", None),
                            "completion_tokens": getattr(response.usage, "completion_tokens", None),
                            "finish_reason": response.choices[0].finish_reason,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            if self.receipt["finish_reason"] not in ("stop", "length"):
                raise RuntimeError(f"Unexpected finish reason: {self.receipt['finish_reason']}")
            new_calls += 1
            return response.choices[0].message.content or ""

    backend = CheckedBackend()
    for item in questions:
        if item["id"] in done:
            continue
        sid = item["sample_id"]
        if sid not in bridges:
            bridges[sid] = CareBridge(turns_for(samples[sid]["conversation"]),
                                      evidence_chars=EVIDENCE_CHARS)
        bridge = bridges[sid]
        started = time.monotonic()
        result = bridge._reader(item["question"], bridge.turns, backend)
        if len(result["bundle_ids"]) != len(bridge.turns):
            raise ValueError(f"Full history unexpectedly truncated: {item['id']}")
        gold_sources = set(item["evidence"])
        event = {"id": item["id"], "sample_id": sid, "category": item["category"],
                 "question": item["question"], "answer": result["answer"],
                 "source_ids": result["source_ids"], "invalid_reason": result["invalid_reason"],
                 "score": score_locomo(result["answer"], item["answer"], item["category"]),
                 "evidence_recall": (len(gold_sources & set(result["bundle_ids"]))
                                     / len(gold_sources) if gold_sources else None),
                 "turns_visible": len(result["bundle_ids"]),
                 "seconds": time.monotonic() - started, **backend.receipt}
        _append_jsonl(events_path, event)
        events.append(event)
        done.add(item["id"])
        print(json.dumps({"shard": shard_index, "done": len(events),
                          "expected": len(questions), "id": item["id"],
                          "prompt_tokens": event["prompt_tokens_server"]}), flush=True)
    by_id = {event["id"]: event for event in events}
    rows = [by_id[item["id"]] for item in questions]
    answerable = [row for row in rows if row["category"] != 5]
    unsupported = [row for row in rows if row["category"] == 5]
    report = {"manifest": manifest,
              "aggregate": {"n": len(rows),
                            "score": sum(row["score"] for row in rows) / len(rows),
                            "answerable_f1": sum(row["score"] for row in answerable) / len(answerable),
                            "refusal_correct": sum(row["score"] == 1 for row in unsupported),
                            "refusal_n": len(unsupported),
                            "prompt_tokens": sum(row["prompt_tokens_server"] for row in rows),
                            "completion_tokens": sum(row["completion_tokens"] for row in rows)},
              "rows": rows}
    _save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--max-new-calls", type=int, default=20)
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--start-per-category", type=int, default=0)
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    report = run(args.dataset, args.output, shard_index=args.shard_index,
                 question_limit=args.question_limit, max_new_calls=args.max_new_calls)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
