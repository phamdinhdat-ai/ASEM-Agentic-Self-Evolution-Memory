"""Paired CARE-Bridge v2 / temporal evaluation with a shared Qwen3-8B backend.

The selected questions are disjoint from prior CARE offset 0-20 samples.
Gold answers and evidence IDs enter only the post-generation scorer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from asem.care import CareBridge
from asem.care.temporal_bridge import CareTemporalBridge
from scripts.run_care_locomo import _save_json, score_locomo, select_questions, turns_for


ARMS = ("bridge_v2", "temporal")
TEMPORAL_QUERY = re.compile(
    r"\b(?:last year|one year ago|before|after|during|favorite|favourite|quit|"
    r"started|stopped|previously|at the time|no longer)\b", re.I)


def select_paired_questions(dataset: list[dict[str, Any]], *, per_category: int,
                            start_per_category: int, categories: tuple[int, ...],
                            challenge_count: int = 10) -> list[dict[str, Any]]:
    primary = select_questions(dataset, per_category=per_category, categories=categories,
                               start_per_category=start_per_category)
    for item in primary:
        item["slice"] = "primary"
    prior = {item["id"] for item in select_questions(dataset,
             per_category=start_per_category, start_per_category=0, categories=categories)}
    used = prior | {item["id"] for item in primary}
    challenge = []
    for conversation in dataset:
        for ordinal, qa in enumerate(conversation["qa"]):
            question_id = f"{conversation['sample_id']}:{ordinal}"
            if (int(qa["category"]) != 5 or question_id in used
                    or not TEMPORAL_QUERY.search(str(qa["question"]))):
                continue
            item = {"id": question_id, "sample_id": conversation["sample_id"],
                    "category": 5, "question": str(qa["question"]),
                    "answer": str(qa.get("answer", "")),
                    "evidence": list(qa.get("evidence") or []),
                    "slice": "temporal_adversarial"}
            rank = hashlib.sha256(("care-temporal-challenge-v1:" + question_id).encode()).hexdigest()
            challenge.append((rank, item))
    if len(challenge) < challenge_count:
        raise ValueError("Insufficient unseen temporal adversarial questions")
    return primary + [item for _, item in sorted(challenge)[:challenge_count]]


def run(dataset_path: Path, output: Path, backend: Any, *, per_category: int = 6,
        start_per_category: int = 21, categories: tuple[int, ...] = (1, 2, 3, 4, 5),
        max_new_calls: int = 180) -> dict[str, Any]:
    dataset_path, output = Path(dataset_path), Path(output)
    payload = dataset_path.read_bytes()
    dataset = json.loads(payload)
    questions = select_paired_questions(dataset, per_category=per_category,
        categories=categories, start_per_category=start_per_category)
    conversations = {conversation["sample_id"]: conversation for conversation in dataset}
    output.mkdir(parents=True, exist_ok=True)
    code_root = Path(__file__).resolve().parents[1]
    code_files = ("asem/care/bridge.py", "asem/care/temporal_bridge.py",
                  "asem/care/temporal_relations.py", "scripts/run_care_temporal_locomo.py")
    manifest = {"protocol": "care-temporal-paired-v1",
        "data_sha256": hashlib.sha256(payload).hexdigest(),
        "code_sha256": {name: hashlib.sha256((code_root / name).read_bytes()).hexdigest()
                        for name in code_files},
        "selected_ids": [item["id"] for item in questions],
        "selected_slices": {item["id"]: item["slice"] for item in questions},
        "temporal_query_pattern": TEMPORAL_QUERY.pattern,
        "per_category": per_category, "start_per_category": start_per_category,
        "categories": list(categories), "arms": list(ARMS),
        "scorer": "LoCoMo task_eval/evaluation.py token-F1 semantics",
        "official_full_run": False, "gold_in_model_context": False,
        "note": "Fresh hash offsets 21-26 plus 10 unseen category-5 temporal lexical questions; paired v2 reader."}
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
            memories[sample] = (CareBridge(turns), CareTemporalBridge(turns))
        bridge, temporal = memories[sample]
        row = next((record for record in rows if record["id"] == item["id"]), None)
        if row is None:
            row = {key: item[key] for key in ("id", "sample_id", "category", "question", "slice")}
            row["systems"] = {}
            rows.append(row)
        for arm in (ARMS if int(hashlib.sha256(item["id"].encode()).hexdigest(), 16) % 2
                    else ARMS[::-1]):
            if (item["id"], arm) in done:
                continue
            before = len(getattr(backend, "calls", []))
            if before >= max_new_calls:
                _save_json(rows_path, rows)
                raise RuntimeError("CARE-Bridge model-call cap reached")
            started = time.monotonic()
            try:
                result = (bridge if arm == "bridge_v2" else temporal).answer(
                    item["question"], backend)
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
            "escalations": sum(bool(value.get("escalated")) for value in values),
            "guard_incompatible": sum(str(value.get("invalid_reason", "")).startswith("temporal_")
                                      or str((value.get("first_pass") or {}).get("invalid_reason", ""))
                                      .startswith("temporal_") for value in values)}
    by_slice = {slice_name: {arm: {
        "n": sum(slice_name == row["slice"] and arm in row["systems"] for row in rows),
        "mean_score": (sum(row["systems"][arm]["f1"] for row in rows
                           if row["slice"] == slice_name and arm in row["systems"])
                       / max(1, sum(row["slice"] == slice_name and arm in row["systems"]
                                    for row in rows)))} for arm in ARMS}
                for slice_name in ("primary", "temporal_adversarial")}
    report = {"manifest": manifest, "aggregate": aggregate, "by_slice": by_slice, "rows": rows}
    _save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=6)
    parser.add_argument("--start-per-category", type=int, default=21)
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-new-calls", type=int, default=180)
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
