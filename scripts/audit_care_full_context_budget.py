"""Tokenize untruncated LoCoMo histories under the locked Qwen3-8B prompt."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from transformers import AutoTokenizer

from scripts import run_care_context_baselines as context_baselines
from scripts.run_care_context_baselines import (
    FULL_CONTEXT_PROMPT, MAX_ANSWER_TOKENS, MODEL, NO_MEMORY_PROMPT,
    SAFETY_TOKENS, turn_text,
)
from scripts.run_care_locomo import turns_for


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/locomo/locomo10.json"
OUTPUT = ROOT / "analysis/ASEM_CARE_full_context_budget_20261113.json"
EXPECTED_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"


def token_count(tokenizer, prompt: str) -> int:
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    return len(tokens["input_ids"] if isinstance(tokens, Mapping) else tokens)


def main() -> None:
    payload = DATASET.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_SHA256
    dataset = json.loads(payload)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL["repo"], revision=MODEL["revision"], use_fast=True, local_files_only=True)
    baseline = json.loads((ROOT / "build/care/context_baselines_merged.json")
                          .read_text(encoding="utf-8"))
    no_memory_mismatches = [row["id"] for row in baseline["rows"]
        if token_count(tokenizer, NO_MEMORY_PROMPT.format(query=row["question"]))
        != row["systems"]["no_memory"]["prompt_tokens_server"]]
    assert not no_memory_mismatches, no_memory_mismatches[:5]
    context_baselines.token_count = token_count
    rows = []
    for conversation in dataset:
        turns = turns_for(conversation["conversation"])
        questions = [str(qa["question"]) for qa in conversation["qa"]]
        _, reconstructed_coverage = context_baselines.fit_context(tokenizer, turns, questions)
        recorded_coverage = baseline["context_coverage"][conversation["sample_id"]]
        assert reconstructed_coverage == recorded_coverage, conversation["sample_id"]
        longest = max(questions, key=lambda q: token_count(
            tokenizer, NO_MEMORY_PROMPT.format(query=q)))
        context = "\n".join(turn_text(turn) for turn in turns)
        prompt_tokens = token_count(tokenizer, FULL_CONTEXT_PROMPT.format(
            context=context, query=longest))
        rows.append({"sample_id": conversation["sample_id"],
            "turns": len(turns), "questions": len(questions),
            "full_prompt_tokens": prompt_tokens,
            "required_window_with_reserve": prompt_tokens + MAX_ANSWER_TOKENS + SAFETY_TOKENS})
    result = {"protocol": "care-full-context-budget-audit-v1",
        "dataset_sha256": EXPECTED_SHA256, "model": MODEL,
        "selection": "per-conversation question longest under NoMemory tokenizer policy",
        "tokenizer_validation": "all 1986 reconstructed NoMemory prompt counts match server receipts",
        "context_validation": "all 10 reconstructed fixed contexts match recorded SHA-256 and metadata",
        "reserve_tokens": MAX_ANSWER_TOKENS + SAFETY_TOKENS,
        "rows": sorted(rows, key=lambda r: r["sample_id"])}
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
