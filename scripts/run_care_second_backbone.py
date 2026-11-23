"""Frozen 50-question CARE transfer to Qwen3-4B on the same LoCoMo cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from asem.care import CareBridge
from scripts.run_care_full_locomo import _append_jsonl, _load_events, selected_questions
from scripts.run_care_locomo import _save_json, score_locomo, turns_for


MODEL = {"repo": "Qwen/Qwen3-4B",
         "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
         "served_name": "asem-care-qwen3-4b"}


def run(dataset_path: Path, cohort_path: Path, output: Path, backend,
        *, question_limit: int = 0, max_new_calls: int = 120) -> dict:
    payload = dataset_path.read_bytes()
    dataset = json.loads(payload)
    cohort_bytes = cohort_path.read_bytes()
    cohort = json.loads(cohort_bytes)
    ids = cohort["question_ids"]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError("Second-backbone cohort must have 50 distinct question IDs")
    all_questions = {row["id"]: row for shard in range(5)
                     for row in selected_questions(dataset, shard)}
    if not set(ids) <= set(all_questions):
        raise ValueError("Cohort contains an unknown question")
    if question_limit:
        ids = ids[:question_limit]
    questions = [all_questions[qid] for qid in ids]
    manifest = {"protocol": "care-second-backbone-qwen3-4b-v1",
                "dataset_sha256": hashlib.sha256(payload).hexdigest(),
                "cohort_sha256": hashlib.sha256(cohort_bytes).hexdigest(),
                "model": MODEL, "question_ids": ids, "question_limit": question_limit,
                "window_tokens": 8192, "answer_cap": 224,
                "reader": "unchanged CARE Bridge first and conditional pass",
                "gold_in_model_context": False}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Refusing to resume a different backbone manifest")
    _save_json(manifest_path, manifest)
    events_path = output / "events.jsonl"
    events = _load_events(events_path)
    done = {event["id"] for event in events}
    if len(done) != len(events):
        raise ValueError("Duplicate checkpoint event")
    samples = {sample["sample_id"]: sample for sample in dataset}
    bridges = {}
    for item in questions:
        if item["id"] in done:
            continue
        if len(backend.calls) >= max_new_calls:
            raise RuntimeError("Second-backbone call cap reached")
        sid = item["sample_id"]
        if sid not in bridges:
            bridges[sid] = CareBridge(turns_for(samples[sid]["conversation"]))
        bridge = bridges[sid]
        before = len(backend.calls)
        started = time.monotonic()
        final = bridge.answer(item["question"], backend)
        receipts = backend.calls[before:]
        if len(receipts) != 1 + bool(final["escalated"]):
            raise ValueError(f"Unexpected model-call count at {item['id']}")
        first = final["first_pass"] or final
        gold = set(item["evidence"])

        def value(answer: dict, used_receipts: list[dict]) -> dict:
            return {"answer": answer["answer"], "source_ids": answer["source_ids"],
                    "bundle_ids": answer["bundle_ids"],
                    "score": score_locomo(answer["answer"], item["answer"], item["category"]),
                    "evidence_recall": (len(gold & set(answer["bundle_ids"])) / len(gold)
                                        if gold else None),
                    "calls": len(used_receipts),
                    "model_tokens": sum((receipt["prompt_tokens"] or 0)
                                        + (receipt["completion_tokens"] or 0)
                                        for receipt in used_receipts)}

        event = {"id": item["id"], "sample_id": sid, "category": item["category"],
                 "question": item["question"], "escalated": final["escalated"],
                 "seconds": time.monotonic() - started,
                 "systems": {"first": value(first, receipts[:1]),
                             "conditional": value(final, receipts)}}
        _append_jsonl(events_path, event)
        events.append(event)
        done.add(item["id"])
        if len(events) % 10 == 0:
            print(json.dumps({"done": len(events), "expected": len(ids)}), flush=True)
    by_id = {event["id"]: event for event in events}
    rows = [by_id[qid] for qid in ids]
    if len(rows) != len(ids):
        raise ValueError("Incomplete second-backbone report")
    report = {"manifest": manifest, "rows": rows,
              "aggregate": {arm: {
                  "score": sum(row["systems"][arm]["score"] for row in rows) / len(rows),
                  "calls": sum(row["systems"][arm]["calls"] for row in rows),
                  "model_tokens": sum(row["systems"][arm]["model_tokens"] for row in rows)}
                  for arm in ("first", "conditional")}}
    _save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, default=Path("cohort_ids.json"))
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--max-new-calls", type=int, default=120)
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--start-per-category", type=int, default=0)
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    from openai import OpenAI

    class Counted:
        def __init__(self):
            self.client = OpenAI(base_url="http://127.0.0.1:8000/v1",
                                 api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                                 timeout=180, max_retries=2)
            self.log_path = args.output / "calls.jsonl"
            self.calls = _load_events(self.log_path)

        def generate(self, prompt: str, **kwargs) -> str:
            if len(self.calls) >= args.max_new_calls:
                raise RuntimeError("Second-backbone call cap reached")
            response = self.client.chat.completions.create(
                model=MODEL["served_name"], messages=[{"role": "user", "content": prompt}],
                temperature=float(kwargs.get("temperature", 0.0)),
                max_tokens=int(kwargs.get("max_tokens", 224)),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            receipt = {"prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                       "completion_tokens": getattr(response.usage, "completion_tokens", None),
                       "finish_reason": response.choices[0].finish_reason}
            if receipt["finish_reason"] not in ("stop", "length"):
                raise RuntimeError(f"Unexpected finish reason: {receipt['finish_reason']}")
            _append_jsonl(self.log_path, receipt)
            self.calls.append(receipt)
            return response.choices[0].message.content or ""

    report = run(args.dataset, args.cohort, args.output, Counted(),
                 question_limit=args.question_limit, max_new_calls=args.max_new_calls)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
