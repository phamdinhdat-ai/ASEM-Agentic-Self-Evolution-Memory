"""External CARE transfer check on all 500 LongMemEval cleaned oracle questions.

The diagnostic scorer is normalized token F1/EM plus abstention accuracy; it
is not the official model-judged LongMemEval accuracy or full-haystack setup.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import string
import time

from asem.care import CareBridge
from asem.care.sourcefirst import SourceTurn
from scripts.run_care_full_locomo import _append_jsonl, _load_events
from scripts.run_care_locomo import _save_json


NO_INFO = "no information available"


def select_records(dataset: list[dict], shard_index: int) -> list[dict]:
    if not 0 <= shard_index < 5:
        raise ValueError("shard_index must be 0..4")
    return [row for row in dataset if int(hashlib.sha256(
        str(row["question_id"]).encode()).hexdigest(), 16) % 5 == shard_index]


def turns_for_record(record: dict) -> list[SourceTurn]:
    sessions = record["haystack_sessions"]
    dates = record["haystack_dates"]
    ids = record["haystack_session_ids"]
    if not (len(sessions) == len(dates) == len(ids)):
        raise ValueError(f"Malformed LongMemEval session arrays: {record['question_id']}")
    turns = []
    for session_index, (session, date) in enumerate(zip(sessions, dates)):
        session_id = f"s{session_index + 1}"
        for turn_index, message in enumerate(session):
            turns.append(SourceTurn(source_id=f"{session_id}:t{turn_index + 1}",
                                    session_id=session_id, date=str(date),
                                    speaker=str(message.get("role", "unknown")).capitalize(),
                                    text=str(message.get("content", "")),
                                    ordinal=len(turns)))
    if not turns:
        raise ValueError(f"No source turns: {record['question_id']}")
    return turns


def normalize(text: str) -> str:
    text = "".join(char for char in text.casefold() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def diagnostic_score(prediction: str, reference: str, unsupported: bool) -> dict:
    if unsupported:
        accepted = NO_INFO in prediction.casefold() or "not mentioned" in prediction.casefold()
        return {"score": float(accepted), "em": float(accepted),
                "f1": float(accepted)}
    p, a = normalize(prediction), normalize(reference)
    common = sum((Counter(p.split()) & Counter(a.split())).values())
    f1 = 2 * common / (len(p.split()) + len(a.split())) if common else 0.0
    return {"score": f1, "em": float(p == a), "f1": f1}


def aggregate(rows: list[dict], arm: str) -> dict:
    answerable = [row["systems"][arm] for row in rows if not row["unsupported"]]
    unsupported = [row["systems"][arm] for row in rows if row["unsupported"]]
    return {"n": len(rows), "mean_diagnostic_score": sum(
                row["systems"][arm]["score"] for row in rows) / len(rows),
            "answerable_f1": (sum(value["f1"] for value in answerable) / len(answerable)
                              if answerable else None),
            "answerable_em": (sum(value["em"] for value in answerable) / len(answerable)
                              if answerable else None),
            "unsupported_correct": sum(value["score"] == 1 for value in unsupported),
            "unsupported_n": len(unsupported),
            "source_session_recall": (sum(value["source_session_recall"] or 0
                                          for value in answerable) / len(answerable)
                                      if answerable else None),
            "calls": sum(row["systems"][arm]["calls"] for row in rows)}


def run(dataset_path: Path, output: Path, backend, *, shard_index: int,
        question_limit: int = 0, max_new_calls: int = 300) -> dict:
    payload = dataset_path.read_bytes()
    dataset = json.loads(payload)
    records = select_records(dataset, shard_index)
    if question_limit:
        records = records[:question_limit]
    manifest = {"protocol": "care-longmemeval-cleaned-oracle-v1",
                "dataset_sha256": hashlib.sha256(payload).hexdigest(),
                "shard_index": shard_index, "question_limit": question_limit,
                "question_ids": [str(row["question_id"]) for row in records],
                "model_window": 8192, "answer_cap": 224,
                "condition": "cleaned oracle sessions, original user/assistant turns",
                "scorer": "diagnostic normalized token F1/EM and phrase-based abstention; not official accuracy",
                "gold_in_model_context": False, "has_answer_flags_in_model_context": False}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Refusing to resume a different LongMemEval manifest")
    _save_json(manifest_path, manifest)
    events_path = output / "events.jsonl"
    events = _load_events(events_path)
    done = {event["id"] for event in events}
    if len(done) != len(events):
        raise ValueError("Duplicate LongMemEval checkpoint event")
    for record in records:
        qid = str(record["question_id"])
        if qid in done:
            continue
        if len(backend.calls) >= max_new_calls:
            raise RuntimeError("LongMemEval CARE call cap reached")
        turns = turns_for_record(record)
        bridge = CareBridge(turns)
        started = time.monotonic()
        before = len(backend.calls)
        final = bridge.answer(str(record["question"]), backend)
        calls = len(backend.calls) - before
        if calls != 1 + bool(final["escalated"]):
            raise ValueError(f"Unexpected call count at {qid}")
        first = final["first_pass"] or final
        gold = set(map(str, record.get("answer_session_ids") or []))
        session_ids = list(map(str, record["haystack_session_ids"]))
        unsupported = qid.endswith("_abs")

        def result(answer: dict, n_calls: int) -> dict:
            visible_local = {source.split(":")[0] for source in answer["bundle_ids"]}
            visible = {session_ids[int(sid[1:]) - 1] for sid in visible_local}
            return {"answer": answer["answer"], "source_ids": answer["source_ids"],
                    "bundle_turns": len(answer["bundle_ids"]),
                    "source_session_recall": len(gold & visible) / len(gold) if gold else None,
                    "calls": n_calls,
                    **diagnostic_score(answer["answer"], str(record["answer"]), unsupported)}

        event = {"id": qid, "question_type": record["question_type"],
                 "unsupported": unsupported, "escalated": final["escalated"],
                 "seconds": time.monotonic() - started,
                 "systems": {"first": result(first, 1),
                             "conditional": result(final, calls)}}
        _append_jsonl(events_path, event)
        events.append(event)
        done.add(qid)
        if len(events) % 20 == 0:
            print(json.dumps({"shard": shard_index, "done": len(events),
                              "expected": len(records)}), flush=True)
    by_id = {event["id"]: event for event in events}
    rows = [by_id[str(record["question_id"])] for record in records]
    report = {"manifest": manifest,
              "aggregate": {arm: aggregate(rows, arm) for arm in ("first", "conditional")},
              "by_question_type": {kind: {
                  arm: aggregate([row for row in rows if row["question_type"] == kind], arm)
                  for arm in ("first", "conditional")}
                  for kind in sorted({row["question_type"] for row in rows})},
              "rows": rows}
    _save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--max-new-calls", type=int, default=300)
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
                raise RuntimeError("LongMemEval CARE call cap reached")
            response = self.client.chat.completions.create(
                model="asem-atlas-qwen3-8b",
                messages=[{"role": "user", "content": prompt}],
                temperature=float(kwargs.get("temperature", 0.0)),
                max_tokens=int(kwargs.get("max_tokens", 224)),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            receipt = {"prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                       "completion_tokens": getattr(response.usage, "completion_tokens", None),
                       "finish_reason": response.choices[0].finish_reason}
            if receipt["finish_reason"] not in ("stop", "length"):
                raise RuntimeError(f"Unexpected finish reason at model call: {receipt['finish_reason']}")
            _append_jsonl(self.log_path, receipt)
            self.calls.append(receipt)
            return response.choices[0].message.content or ""

    report = run(args.dataset, args.output, Counted(), shard_index=args.shard_index,
                 question_limit=args.question_limit, max_new_calls=args.max_new_calls)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
