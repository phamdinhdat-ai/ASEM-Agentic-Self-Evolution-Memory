"""Qwen3-8B NoMemory and context-window FullContext on CARE's locked LoCoMo shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from scripts.run_care_full_locomo import (
    ModelBackendError, SHARDS, _append_jsonl, _load_events, selected_questions,
)
from scripts.run_care_locomo import _save_json, score_locomo, turns_for


ARMS = ("no_memory", "full_context_8k")
MODEL = {"repo": "Qwen/Qwen3-8B",
         "revision": "b968826d9c46dd6066d109eabc6255188de91218"}
WINDOW = 8192
MAX_ANSWER_TOKENS = 224
SAFETY_TOKENS = 32
NO_MEMORY_PROMPT = (
    "Answer the following question as concisely as possible. "
    "Give only the answer (a few words or one sentence). "
    "If you are not sure, provide your best guess rather than asking for more information.\n\n"
    "Question: {query}\n\nAnswer:"
)
FULL_CONTEXT_PROMPT = (
    "Each conversation line is prefixed with the absolute session date in brackets. "
    "When the question asks for a time or date, resolve relative expressions "
    "against that session date and give the exact absolute date.\n\n"
    "Use the conversation excerpts below to answer the question. "
    "Reply with only the answer — a few words or one sentence, no explanation.\n\n"
    "Conversation:\n{context}\n\nQuestion: {query}\n\nAnswer:"
)
OMISSION = "[Earlier conversation turns omitted to fit the 8K model window]"


def token_count(tokenizer: Any, prompt: str) -> int:
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    return len(tokens["input_ids"] if isinstance(tokens, dict) else tokens)


def turn_text(turn: Any) -> str:
    return f"[{turn.date}] {turn.speaker}: {turn.text}"


def fit_context(tokenizer: Any, turns: list[Any], questions: list[str]) -> tuple[str, dict]:
    """Keep the first five dated turns and the longest recent suffix that fits."""
    lines = [turn_text(turn) for turn in turns]
    head = min(5, len(lines))
    longest_question = max(questions, key=lambda q: token_count(tokenizer,
        NO_MEMORY_PROMPT.format(query=q)))

    def make(suffix_size: int) -> str:
        if head + suffix_size >= len(lines):
            return "\n".join(lines)
        suffix = lines[len(lines) - suffix_size:] if suffix_size else []
        return "\n".join([*lines[:head], OMISSION, *suffix])

    budget = WINDOW - MAX_ANSWER_TOKENS - SAFETY_TOKENS
    low, high = 0, len(lines) - head
    while low < high:
        middle = (low + high + 1) // 2
        prompt = FULL_CONTEXT_PROMPT.format(context=make(middle),
                                             query=longest_question)
        if token_count(tokenizer, prompt) <= budget:
            low = middle
        else:
            high = middle - 1
    context = make(low)
    if token_count(tokenizer, FULL_CONTEXT_PROMPT.format(
            context=context, query=longest_question)) > budget:
        raise ValueError("First five conversation turns do not fit 8K window")
    return context, {
        "total_turns": len(lines), "included_turns": min(len(lines), head + low),
        "head_turns": head, "recent_turns": low,
        "truncated": head + low < len(lines),
        "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
        "longest_question_prompt_tokens": token_count(
            tokenizer, FULL_CONTEXT_PROMPT.format(context=context,
                                                   query=longest_question)),
    }


def summarize(rows: list[dict], arm: str) -> dict:
    values = [row["systems"][arm] for row in rows]
    answerable = [row["systems"][arm] for row in rows if row["category"] != 5]
    adversarial = [row["systems"][arm] for row in rows if row["category"] == 5]
    return {
        "n": len(values),
        "mean_score": sum(v["score"] for v in values) / len(values) if values else None,
        "answerable_mean_f1": (sum(v["score"] for v in answerable) / len(answerable)
                               if answerable else None),
        "adversarial_correct": sum(v["score"] == 1 for v in adversarial),
        "adversarial_n": len(adversarial),
        "errors": sum(v["error"] is not None for v in values),
        "calls": sum(v["new_calls"] for v in values),
        "seconds": sum(v["seconds"] for v in values),
    }


def run(dataset_path: Path, output: Path, *, shard_index: int, max_new_calls: int,
        question_limit: int = 0) -> dict:
    from openai import APIError, OpenAI
    from transformers import AutoTokenizer

    dataset_path, output = Path(dataset_path), Path(output)
    dataset_bytes = dataset_path.read_bytes()
    dataset = json.loads(dataset_bytes)
    questions = selected_questions(dataset, shard_index)
    if question_limit:
        questions = questions[:question_limit]
    samples = {sample["sample_id"]: sample for sample in dataset}
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL["repo"], revision=MODEL["revision"], use_fast=True)
    contexts = {}
    coverage = {}
    for sample_id in SHARDS[shard_index]:
        items = [item for item in questions if item["sample_id"] == sample_id]
        if not items:
            continue
        turns = turns_for(samples[sample_id]["conversation"])
        contexts[sample_id], coverage[sample_id] = fit_context(
            tokenizer, turns, [item["question"] for item in items])
    manifest = {
        "protocol": "care-context-baselines-locomo-v1",
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "model": MODEL, "shard_index": shard_index,
        "shard_samples": list(SHARDS[shard_index]),
        "question_limit": question_limit,
        "question_ids": [item["id"] for item in questions],
        "arms": list(ARMS),
        "window_tokens": WINDOW, "max_answer_tokens": MAX_ANSWER_TOKENS,
        "safety_tokens": SAFETY_TOKENS, "full_context_policy": "first 5 turns plus longest recent suffix",
        "no_memory_prompt_sha256": hashlib.sha256(NO_MEMORY_PROMPT.encode()).hexdigest(),
        "full_context_prompt_sha256": hashlib.sha256(FULL_CONTEXT_PROMPT.encode()).hexdigest(),
        "coverage": coverage,
        "scorer": "LoCoMo token-F1 and category-5 refusal semantics",
        "data_adapter": "text turns plus available image captions, with session dates",
        "gold_in_model_context": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(
            manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Refusing to resume a different baseline manifest")
    _save_json(manifest_path, manifest)
    events_path = output / "events.jsonl"
    events = _load_events(events_path)
    done = {(event["id"], event["arm"]) for event in events}
    if len(done) != len(events):
        raise ValueError("Duplicate checkpoint event")
    client = OpenAI(base_url="http://127.0.0.1:8000/v1",
                    api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                    timeout=180, max_retries=2)
    calls = 0
    for item in questions:
        for arm in ARMS:
            if (item["id"], arm) in done:
                continue
            if calls >= max_new_calls:
                raise RuntimeError("Baseline model-call cap reached")
            prompt = (NO_MEMORY_PROMPT.format(query=item["question"])
                      if arm == "no_memory" else
                      FULL_CONTEXT_PROMPT.format(
                          context=contexts[item["sample_id"]], query=item["question"]))
            prompt_tokens = token_count(tokenizer, prompt)
            if prompt_tokens + MAX_ANSWER_TOKENS > WINDOW:
                raise ValueError(f"Prompt exceeds model window: {item['id']} {arm}")
            started = time.monotonic()
            try:
                response = client.chat.completions.create(
                    model="asem-atlas-qwen3-8b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=MAX_ANSWER_TOKENS,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            except APIError as exc:
                raise ModelBackendError(f"Model API failed: {type(exc).__name__}: {exc}") from exc
            choice = response.choices[0]
            answer = choice.message.content or ""
            if choice.finish_reason not in ("stop", "length"):
                raise RuntimeError(f"Unexpected finish reason: {choice.finish_reason}")
            value = {
                "answer": answer, "error": None,
                "score": score_locomo(answer, item["answer"], item["category"]),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_tokens_local": prompt_tokens,
                "prompt_tokens_server": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "finish_reason": choice.finish_reason,
                "new_calls": 1, "seconds": time.monotonic() - started,
            }
            _append_jsonl(events_path, {"id": item["id"], "arm": arm, "value": value})
            events.append({"id": item["id"], "arm": arm, "value": value})
            done.add((item["id"], arm))
            calls += 1
            if len(events) % 25 == 0:
                print(json.dumps({"shard": shard_index, "completed_predictions": len(events),
                                  "expected_predictions": len(questions) * len(ARMS)}),
                      flush=True)
    by_id = {item["id"]: {
        "id": item["id"], "sample_id": item["sample_id"],
        "category": item["category"], "question": item["question"], "systems": {},
    } for item in questions}
    for event in events:
        by_id[event["id"]]["systems"][event["arm"]] = event["value"]
    rows = [by_id[item["id"]] for item in questions]
    if any(set(row["systems"]) != set(ARMS) for row in rows):
        raise RuntimeError("Incomplete paired baseline shard")
    report = {
        "manifest": manifest,
        "aggregate": {arm: summarize(rows, arm) for arm in ARMS},
        "by_category": {str(category): {
            arm: summarize([row for row in rows if row["category"] == category], arm)
            for arm in ARMS} for category in range(1, 6)},
        "rows": rows,
    }
    _save_json(output / "rows.json", rows)
    _save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--max-new-calls", type=int, default=1000)
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--start-per-category", type=int, default=0)
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    report = run(args.dataset, args.output, shard_index=args.shard_index,
                 max_new_calls=args.max_new_calls, question_limit=args.question_limit)
    print(json.dumps({"shard": args.shard_index, "aggregate": report["aggregate"],
                      "coverage": report["manifest"]["coverage"]}), flush=True)


if __name__ == "__main__":
    main()
