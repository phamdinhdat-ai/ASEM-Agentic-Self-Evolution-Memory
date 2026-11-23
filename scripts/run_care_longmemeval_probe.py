"""Small diagnostic probe: why does CARE refuse so often on LongMemEval?

Three arms are run on the SAME ~30 cleaned-oracle questions with the same
Qwen3-8B server, so the comparison is paired:

* ``baseline``  - CareBridge exactly as in the 500-question run (top-8 BM25
                  turns, 10k evidence characters, abstention-heavy prompt).
* ``wide``      - same prompt, but a larger first bundle (top-20 turns, 20k
                  characters; the 8,192-token window was only ~27% used).
* ``wide_prompt`` - ``wide`` plus a reader prompt that does not instruct the
                  model to refuse when evidence is partial, and asks it to
                  count / sum / subtract across turns. Unsupported-premise
                  abstention is kept, so the `_abs` questions measure the cost.

No library code is modified: the prompt change is applied by a backend wrapper
that rewrites two sentences of the reader prompt (asserted to exist).
This is a diagnostic probe, not a result to put in the paper.
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

# (question_type, count) for answerable questions; ABS_COUNT `_abs` questions
# are added on top, drawn from every type.
QUOTA = (("multi-session", 12), ("temporal-reasoning", 9),
         ("knowledge-update", 2), ("single-session-user", 2))
ABS_COUNT = 5

FIRST_K = {"baseline": 8, "wide": 20, "wide_prompt": 20}
EVIDENCE_CHARS = {"baseline": 10000, "wide": 20000, "wide_prompt": 20000}

OLD_PRESUPPOSE = "If the question presupposes an unsupported fact, answer 'No information available'. "
OLD_MISSING = ("If evidence is missing, set answer to "
               "'No information available' and propose searches about the named person/event. ")
NEW_COMPUTE = ("If the question asks how many, a total, or a difference, find EVERY relevant turn, "
               "do the arithmetic yourself, and give the final number with its unit; partial "
               "evidence is still enough to give your best computed answer. "
               "Answer 'No information available' ONLY if the question's premise is "
               "contradicted or the topic is never mentioned at all. ")
NEW_MISSING = ("If you truly find nothing relevant, set answer to 'No information available' "
               "and propose searches. ")


def select_questions(dataset: list[dict]) -> list[dict]:
    def key(record: dict) -> str:
        return hashlib.sha256(("probe:" + str(record["question_id"])).encode()).hexdigest()

    chosen = []
    for kind, count in QUOTA:
        pool = [r for r in dataset if r["question_type"] == kind
                and not str(r["question_id"]).endswith("_abs")]
        chosen += sorted(pool, key=key)[:count]
    chosen += sorted((r for r in dataset if str(r["question_id"]).endswith("_abs")),
                     key=key)[:ABS_COUNT]
    return chosen


class PromptRewriter:
    """Wrap a backend and rewrite the reader's abstention sentences."""

    def __init__(self, backend, rewrite: bool):
        self.backend, self.rewrite = backend, rewrite

    @property
    def calls(self):
        return self.backend.calls

    def generate(self, prompt: str, **kwargs) -> str:
        if self.rewrite:
            for old in (OLD_PRESUPPOSE, OLD_MISSING):
                if old not in prompt:
                    raise ValueError("Reader prompt changed; probe rewrite no longer applies")
            prompt = prompt.replace(OLD_PRESUPPOSE, NEW_COMPUTE).replace(OLD_MISSING, NEW_MISSING)
        return self.backend.generate(prompt, **kwargs)


def run_arm(arm: str, record: dict, backend) -> dict:
    turns = turns_for_record(record)
    bridge = CareBridge(turns, evidence_chars=EVIDENCE_CHARS[arm])
    guarded = PromptRewriter(backend, rewrite=(arm == "wide_prompt"))
    before = len(backend.calls)
    question = str(record["question"])
    final = bridge.answer_with_initial(question, guarded, bridge.rank(question)[:FIRST_K[arm]])
    calls = len(backend.calls) - before
    unsupported = str(record["question_id"]).endswith("_abs")
    gold = set(map(str, record.get("answer_session_ids") or []))
    session_ids = list(map(str, record["haystack_session_ids"]))

    def result(answer: dict, n_calls: int) -> dict:
        visible = {session_ids[int(source.split(":")[0][1:]) - 1] for source in answer["bundle_ids"]}
        return {"answer": answer["answer"], "source_ids": answer["source_ids"],
                "invalid_reason": answer.get("invalid_reason"),
                "bundle_turns": len(answer["bundle_ids"]),
                "source_session_recall": len(gold & visible) / len(gold) if gold else None,
                "calls": n_calls,
                **diagnostic_score(answer["answer"], str(record["answer"]), unsupported)}

    first = final["first_pass"] or final
    return {"first": result(first, 1), "conditional": result(final, calls),
            "escalated": final["escalated"]}


def summarize(rows: list[dict], arm: str, which: str) -> dict:
    answerable = [r["arms"][arm][which] for r in rows if not r["unsupported"]]
    unsupported = [r["arms"][arm][which] for r in rows if r["unsupported"]]
    return {"n_answerable": len(answerable), "answerable_f1": sum(x["f1"] for x in answerable) / max(len(answerable), 1),
            "answerable_em": sum(x["em"] for x in answerable) / max(len(answerable), 1),
            "refusal_rate": sum(NO_INFO in x["answer"].casefold() for x in answerable) / max(len(answerable), 1),
            "abstained_correctly": sum(x["score"] == 1 for x in unsupported), "unsupported_n": len(unsupported),
            "calls": sum(r["arms"][arm][which]["calls"] for r in rows)}


def run(dataset_path: Path, output: Path, backend, *, max_new_calls: int = 220) -> dict:
    payload = dataset_path.read_bytes()
    records = select_questions(json.loads(payload))
    manifest = {"protocol": "care-longmemeval-probe-v1", "dataset_sha256": hashlib.sha256(payload).hexdigest(),
                "question_ids": [str(r["question_id"]) for r in records], "arms": list(FIRST_K),
                "first_k": FIRST_K, "evidence_chars": EVIDENCE_CHARS,
                "note": "diagnostic probe; paired arms on the same questions; not official accuracy"}
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / "manifest.json", manifest)
    events_path = output / "events.jsonl"
    events = _load_events(events_path)
    done = {event["id"] for event in events}
    for record in records:
        qid = str(record["question_id"])
        if qid in done:
            continue
        if len(backend.calls) >= max_new_calls:
            raise RuntimeError("LongMemEval probe call cap reached")
        started = time.monotonic()
        event = {"id": qid, "question_type": record["question_type"],
                 "unsupported": qid.endswith("_abs"),
                 "arms": {arm: run_arm(arm, record, backend) for arm in FIRST_K},
                 "seconds": time.monotonic() - started}
        _append_jsonl(events_path, event)
        events.append(event)
    by_id = {event["id"]: event for event in events}
    rows = [by_id[str(r["question_id"])] for r in records]
    aggregate = {arm: {which: summarize(rows, arm, which) for which in ("first", "conditional")}
                 for arm in FIRST_K}
    _save_json(output / "report.json", {"manifest": manifest, "aggregate": aggregate, "rows": rows})
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-calls", type=int, default=220)
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
                raise RuntimeError("LongMemEval probe call cap reached")
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
