"""Matched CARE-reader ablation: dense top-8 versus the saved sparse top-8.

Both dense outputs share one first model call. The conditional output adds a
second call only when CARE's unchanged gate fires. Gold is scored afterward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from asem.care import CareBridge
from scripts.run_care_full_locomo import (_append_jsonl, _load_events,
                                          selected_questions, SHARDS)
from scripts.run_care_locomo import _save_json, score_locomo, turns_for


def summarize(rows: list[dict], arm: str) -> dict:
    all_values = [row["systems"][arm] for row in rows]
    answerable = [row["systems"][arm] for row in rows if row["category"] != 5]
    unsupported = [row["systems"][arm] for row in rows if row["category"] == 5]
    return {"n": len(all_values),
            "mean_score": sum(v["score"] for v in all_values) / len(all_values),
            "answerable_mean_f1": sum(v["score"] for v in answerable) / len(answerable),
            "refusal_correct": sum(v["score"] == 1 for v in unsupported),
            "refusal_n": len(unsupported),
            "calls": sum(v["model_calls"] for v in all_values),
            "prompt_tokens": sum(v["prompt_tokens"] for v in all_values),
            "completion_tokens": sum(v["completion_tokens"] for v in all_values),
            "evidence_recall": sum(v["evidence_recall"] or 0 for v in answerable)
                               / len(answerable)}


def run(dataset_path: Path, rankings_path: Path, output: Path, backend: Any,
        *, shard_index: int, question_limit: int = 0, max_new_calls: int = 1000) -> dict:
    payload = dataset_path.read_bytes()
    dataset = json.loads(payload)
    questions = selected_questions(dataset, shard_index)
    if question_limit:
        questions = questions[:question_limit]
    rankings_bytes = rankings_path.read_bytes()
    rankings = json.loads(rankings_bytes)["rankings"]
    if any(item["id"] not in rankings for item in questions):
        raise ValueError("Dense rankings do not cover all selected questions")
    manifest = {"protocol": "care-matched-reader-dense-top8-v1",
                "dataset_sha256": hashlib.sha256(payload).hexdigest(),
                "dense_rankings_sha256": hashlib.sha256(rankings_bytes).hexdigest(),
                "shard_index": shard_index, "shard_samples": list(SHARDS[shard_index]),
                "question_ids": [item["id"] for item in questions],
                "question_limit": question_limit,
                "arms": ["dense_8_first", "dense_8_conditional"],
                "reader": "CareBridge._reader, identical prompt and source-binding checks",
                "first_bundle": "top 8 visible dense ranked source turns",
                "repair": "CareBridge.answer_with_initial, unchanged conditional gate and expansion",
                "gold_in_model_context": False}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Refusing to resume a different matched-reader manifest")
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
            raise RuntimeError("Matched reader call cap reached")
        sid = item["sample_id"]
        if sid not in bridges:
            bridges[sid] = CareBridge(turns_for(samples[sid]["conversation"]))
        bridge = bridges[sid]
        ids = rankings[item["id"]][:8]
        if len(ids) != 8 or len(set(ids)) != 8 or any(source not in bridge.by_id for source in ids):
            raise ValueError(f"Invalid top-8 dense ranking for {item['id']}")
        initial = [bridge.by_id[source] for source in ids]
        before = len(backend.calls)
        started = time.monotonic()
        final = bridge.answer_with_initial(item["question"], backend, initial)
        receipts = backend.calls[before:]
        if len(receipts) not in (1, 2) or len(receipts) != 1 + bool(final["escalated"]):
            raise ValueError(f"Unexpected model calls for {item['id']}")
        first = final["first_pass"] or final
        gold_sources = set(item["evidence"])

        def value(answer: dict, selected_receipts: list[dict]) -> dict:
            bundle = answer["bundle_ids"]
            return {"answer": answer["answer"], "source_ids": answer["source_ids"],
                    "bundle_ids": bundle,
                    "score": score_locomo(answer["answer"], item["answer"], item["category"]),
                    "evidence_recall": (len(gold_sources & set(bundle)) / len(gold_sources)
                                        if gold_sources else None),
                    "model_calls": len(selected_receipts),
                    "prompt_tokens": sum(v["prompt_tokens"] or 0 for v in selected_receipts),
                    "completion_tokens": sum(v["completion_tokens"] or 0 for v in selected_receipts)}

        event = {"id": item["id"], "sample_id": sid, "category": item["category"],
                 "question": item["question"], "escalated": final["escalated"],
                 "risk_reasons": final["risk_reasons"],
                 "first_invalid_reason": first["invalid_reason"],
                 "first_search_queries": first["search_queries"],
                 "seconds": time.monotonic() - started,
                 "systems": {"dense_8_first": value(first, receipts[:1]),
                             "dense_8_conditional": value(final, receipts)}}
        _append_jsonl(events_path, event)
        events.append(event)
        done.add(item["id"])
        if len(events) % 25 == 0:
            print(json.dumps({"shard": shard_index, "done": len(events),
                              "expected": len(questions)}), flush=True)
    by_id = {event["id"]: event for event in events}
    rows = [by_id[item["id"]] for item in questions]
    aggregate = {arm: summarize(rows, arm) for arm in manifest["arms"]}
    report = {"manifest": manifest, "aggregate": aggregate, "rows": rows}
    _save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, default=Path("dense_rankings_locked.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--max-new-calls", type=int, default=1000)
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--start-per-category", type=int, default=0)
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    from openai import OpenAI

    class Counted:
        def __init__(self) -> None:
            self.client = OpenAI(base_url="http://127.0.0.1:8000/v1",
                                 api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                                 timeout=180, max_retries=2)
            self.log_path = args.output / "calls.jsonl"
            self.calls = _load_events(self.log_path)

        def generate(self, prompt: str, **kwargs: Any) -> str:
            if len(self.calls) >= args.max_new_calls:
                raise RuntimeError("Matched reader call cap reached")
            response = self.client.chat.completions.create(
                model="asem-atlas-qwen3-8b",
                messages=[{"role": "user", "content": prompt}],
                temperature=float(kwargs.get("temperature", 0.0)),
                max_tokens=int(kwargs.get("max_tokens", 224)),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            receipt = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                       "completion_tokens": getattr(response.usage, "completion_tokens", None),
                       "finish_reason": response.choices[0].finish_reason}
            if receipt["finish_reason"] not in ("stop", "length"):
                raise RuntimeError(f"Unexpected finish reason: {receipt['finish_reason']}")
            _append_jsonl(self.log_path, receipt)
            self.calls.append(receipt)
            return response.choices[0].message.content or ""

    result = run(args.dataset, args.rankings, args.output, Counted(),
                 shard_index=args.shard_index, question_limit=args.question_limit,
                 max_new_calls=args.max_new_calls)
    print(json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
