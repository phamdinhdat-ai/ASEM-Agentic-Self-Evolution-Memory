"""Does exposing LongMemEval's ``question_date`` to the reader help temporal questions?

The 500-question run never passed ``question_date`` to CARE's reader, although
many temporal questions are relative to "today" ("how many months ago ...").
Two arms on all 133 temporal-reasoning questions, same Qwen3-8B server:

* ``baseline`` - CareBridge unchanged (re-run: also measures run-to-run noise
                 against the saved 500-question results).
* ``qdate``    - identical, but the reader prompt gains the question date and
                 one sentence saying relative expressions are measured from it.

The date never enters retrieval (BM25 sees only the question text). No library
code is modified: a backend wrapper inserts the text before the QUESTION line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from asem.care import CareBridge
from scripts.run_care_longmemeval import NO_INFO, diagnostic_score, turns_for_record
from scripts.run_care_full_locomo import _append_jsonl, _load_events
from scripts.run_care_locomo import _save_json

ARMS = ("baseline", "qdate")
MARKER = "\nQUESTION: "
INSERT = ("\nQUESTION DATE: {date}\n"
          "(The question is being asked on QUESTION DATE. Resolve 'ago', 'last week/month/year', "
          "'currently' and 'so far' relative to it, using the session dates of the turns.)"
          "\nQUESTION: ")


class DateInserter:
    def __init__(self, backend, enabled: bool):
        self.backend, self.enabled, self.date = backend, enabled, ""

    @property
    def calls(self):
        return self.backend.calls

    def generate(self, prompt: str, **kwargs) -> str:
        if self.enabled:
            at, turns_at = prompt.find(MARKER), prompt.find("\nORIGINAL TURNS:")
            if not 0 <= at < turns_at:
                raise ValueError("Reader prompt changed; question-date insertion no longer applies")
            prompt = prompt[:at] + INSERT.format(date=self.date) + prompt[at + len(MARKER):]
        return self.backend.generate(prompt, **kwargs)


def run_arm(arm: str, record: dict, backend) -> dict:
    guarded = DateInserter(backend, enabled=(arm == "qdate"))
    guarded.date = str(record["question_date"])
    bridge = CareBridge(turns_for_record(record))
    before = len(backend.calls)
    final = bridge.answer(str(record["question"]), guarded)
    calls = len(backend.calls) - before
    unsupported = str(record["question_id"]).endswith("_abs")

    def result(answer: dict, n_calls: int) -> dict:
        return {"answer": answer["answer"], "source_ids": answer["source_ids"],
                "invalid_reason": answer.get("invalid_reason"), "calls": n_calls,
                **diagnostic_score(answer["answer"], str(record["answer"]), unsupported)}

    first = final["first_pass"] or final
    return {"first": result(first, 1), "conditional": result(final, calls),
            "escalated": final["escalated"]}


def summarize(rows: list[dict], arm: str, which: str) -> dict:
    answerable = [r["arms"][arm][which] for r in rows if not r["unsupported"]]
    unsupported = [r["arms"][arm][which] for r in rows if r["unsupported"]]
    return {"n_answerable": len(answerable),
            "answerable_f1": sum(x["f1"] for x in answerable) / max(len(answerable), 1),
            "answerable_em": sum(x["em"] for x in answerable) / max(len(answerable), 1),
            "refusal_rate": sum(NO_INFO in x["answer"].casefold() for x in answerable) / max(len(answerable), 1),
            "abstained_correctly": sum(x["score"] == 1 for x in unsupported),
            "unsupported_n": len(unsupported),
            "calls": sum(r["arms"][arm][which]["calls"] for r in rows)}


def run(dataset_path: Path, output: Path, backend, *, max_new_calls: int = 700) -> dict:
    payload = dataset_path.read_bytes()
    records = [r for r in json.loads(payload) if r["question_type"] == "temporal-reasoning"]
    manifest = {"protocol": "care-longmemeval-qdate-v1",
                "dataset_sha256": hashlib.sha256(payload).hexdigest(),
                "question_ids": [str(r["question_id"]) for r in records], "arms": list(ARMS),
                "note": "diagnostic; paired arms on all temporal-reasoning questions"}
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / "manifest.json", manifest)
    events_path = output / "events.jsonl"
    events = _load_events(events_path)
    done = {e["id"] for e in events}
    for record in records:
        qid = str(record["question_id"])
        if qid in done:
            continue
        if len(backend.calls) >= max_new_calls:
            raise RuntimeError("question-date run call cap reached")
        started = time.monotonic()
        event = {"id": qid, "unsupported": qid.endswith("_abs"),
                 "question_date": record["question_date"],
                 "arms": {arm: run_arm(arm, record, backend) for arm in ARMS},
                 "seconds": time.monotonic() - started}
        _append_jsonl(events_path, event)
        events.append(event)
    by_id = {e["id"]: e for e in events}
    rows = [by_id[str(r["question_id"])] for r in records]
    aggregate = {arm: {w: summarize(rows, arm, w) for w in ("first", "conditional")} for arm in ARMS}
    _save_json(output / "report.json", {"manifest": manifest, "aggregate": aggregate, "rows": rows})
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-calls", type=int, default=700)
    # Accepted for compatibility with scripts.run_care_kaggle.run; unused here.
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--start-per-category", type=int, default=0)
    parser.add_argument("--categories", nargs="+", type=int, default=[1])
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
                raise RuntimeError("question-date run call cap reached")
            response = self.client.chat.completions.create(
                model="asem-atlas-qwen3-8b", messages=[{"role": "user", "content": prompt}],
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

    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args.dataset, args.output, Counted(), max_new_calls=args.max_new_calls),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
