"""Paired CARE-Bridge development evaluation with a shared Qwen3-8B backend.

The selected 20 questions are disjoint from CARE's first two Kaggle samples.
Gold answers and evidence IDs enter only the post-generation scorer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from asem.care import CareBridge, CareSourceFirst
from scripts.run_care_locomo import _save_json, score_locomo, select_questions, turns_for


ARMS = ("proof_bm25", "bridge", "care_v1")


def run(dataset_path: Path, output: Path, backend: Any, *, per_category: int = 4,
        start_per_category: int = 13, categories: tuple[int, ...] = (1, 2, 3, 4, 5),
        max_new_calls: int = 160) -> dict[str, Any]:
    dataset_path, output = Path(dataset_path), Path(output)
    payload = dataset_path.read_bytes()
    dataset = json.loads(payload)
    questions = select_questions(dataset, per_category=per_category, categories=categories,
                                 start_per_category=start_per_category)
    conversations = {conversation["sample_id"]: conversation for conversation in dataset}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": "care-bridge-locomo-validation-v2",
        "data_sha256": hashlib.sha256(payload).hexdigest(),
        "selected_ids": [item["id"] for item in questions],
        "per_category": per_category, "start_per_category": start_per_category,
        "categories": list(categories), "arms": list(ARMS),
        "scorer": "LoCoMo task_eval/evaluation.py token-F1 semantics",
        "official_full_run": False, "gold_in_model_context": False,
        "note": "Validation sample at offset 13; Bridge v2 was designed using offset 9 only."}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Refusing to resume a different CARE-Bridge manifest")
    _save_json(manifest_path, manifest)
    rows_path = output / "rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    done = {(row["id"], arm) for row in rows for arm in row["systems"]}
    memories = {}
    for item in questions:
        if all((item["id"], arm) in done for arm in ARMS):
            continue
        sample = item["sample_id"]
        if sample not in memories:
            turns = turns_for(conversations[sample]["conversation"])
            memories[sample] = (CareBridge(turns),
                                CareSourceFirst(turns, candidate_limit=18, evidence_chars=10000))
        bridge, legacy = memories[sample]
        row = next((record for record in rows if record["id"] == item["id"]), None)
        if row is None:
            row = {key: item[key] for key in ("id", "sample_id", "category", "question")}
            row["systems"] = {}
            rows.append(row)
        for arm in ARMS:
            if (item["id"], arm) in done:
                continue
            before = len(getattr(backend, "calls", []))
            if before >= max_new_calls:
                _save_json(rows_path, rows)
                raise RuntimeError("CARE-Bridge model-call cap reached")
            started = time.monotonic()
            try:
                if arm == "care_v1":
                    result = legacy.answer(item["question"], backend, compile_mode="risk")
                else:
                    result = bridge.answer(item["question"], backend, mode=arm)
                error = None
            except Exception as exc:
                if "model-call cap" in str(exc):
                    _save_json(rows_path, rows)
                    raise
                result = {"answer": "", "source_ids": [], "bundle_ids": [], "candidate_ids": []}
                error = type(exc).__name__ + ": " + str(exc)[:120]
            result.pop("raw_reader_response", None)
            gold_evidence = set(item["evidence"])
            row["systems"][arm] = {**result, "error": error,
                "f1": score_locomo(result["answer"], item["answer"], item["category"]),
                "evidence_recall": (len(gold_evidence & set(result["bundle_ids"])) / len(gold_evidence)
                                    if gold_evidence else None),
                "new_calls": len(getattr(backend, "calls", [])) - before,
                "seconds": time.monotonic() - started}
            _save_json(rows_path, rows)
            print(json.dumps({"id": item["id"], "arm": arm,
                "f1": row["systems"][arm]["f1"], "error": error,
                "calls": row["systems"][arm]["new_calls"]}), flush=True)
    aggregate = {}
    for arm in ARMS:
        values = [row["systems"][arm] for row in rows if arm in row["systems"]]
        answerable = [row["systems"][arm] for row in rows
                      if row["category"] != 5 and arm in row["systems"]]
        refusals = [row["systems"][arm] for row in rows
                    if row["category"] == 5 and arm in row["systems"]]
        aggregate[arm] = {"n": len(values),
            "mean_score": sum(value["f1"] for value in values) / len(values) if values else 0,
            "answerable_mean_f1": (sum(value["f1"] for value in answerable) / len(answerable)
                                   if answerable else None),
            "answerable_mean_evidence_recall": (sum(value["evidence_recall"] or 0 for value in answerable)
                                               / len(answerable) if answerable else None),
            "adversarial_correct": sum(value["f1"] == 1 for value in refusals),
            "adversarial_n": len(refusals),
            "calls": sum(value["new_calls"] for value in values),
            "answer_seconds": sum(value["seconds"] for value in values),
            "errors": sum(value["error"] is not None for value in values),
            "escalations": sum(bool(value.get("escalated")) for value in values)}
    report = {"manifest": manifest, "aggregate": aggregate, "rows": rows}
    _save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--start-per-category", type=int, default=13)
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-new-calls", type=int, default=160)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="asem-atlas-qwen3-8b")
    args = parser.parse_args()
    from openai import OpenAI

    class Counted:
        def __init__(self):
            self.client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180)
            self.calls: list[dict[str, Any]] = []

        def generate(self, prompt: str, **kwargs: Any) -> str:
            if len(self.calls) >= args.max_new_calls:
                raise RuntimeError("CARE-Bridge model-call cap reached")
            response = self.client.chat.completions.create(model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(kwargs.get("temperature", 0.0)),
                max_tokens=int(kwargs.get("max_tokens", 224)),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            choice = response.choices[0]
            answer = choice.message.content or ""
            receipt = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None)}
            self.calls.append(receipt)
            _save_json(args.output / "raw_calls" / f"{len(self.calls):05d}.json",
                {"prompt": prompt, "response": answer, **receipt})
            return answer

    backend = Counted()
    report = run(args.dataset, args.output, backend, per_category=args.per_category,
                 start_per_category=args.start_per_category, categories=tuple(args.categories),
                 max_new_calls=args.max_new_calls)
    _save_json(args.output / "calls.json", backend.calls)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
