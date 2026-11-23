"""Paired CARE-SourceFirst versus BM25 pilot on locked LoCoMo conversations.

The sample is selected by hash of conversation/question index. Gold answers and
evidence IDs are used only after model calls for scoring/diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import string
import time
from typing import Any

from asem.care import CareSourceFirst, SourceTurn


def turns_for(conversation: dict[str, Any]) -> list[SourceTurn]:
    turns = []
    sessions = sorted((int(m.group(1)), key) for key in conversation
                      if (m := re.fullmatch(r"session_(\d+)", key)))
    for _, session_key in sessions:
        date = str(conversation.get(session_key + "_date_time", ""))
        for ordinal, turn in enumerate(conversation[session_key]):
            text = str(turn.get("text") or "")
            caption = turn.get("blip_caption")
            if caption:
                text += " [image caption: " + str(caption) + "]"
            turns.append(SourceTurn(str(turn["dia_id"]), session_key, date,
                str(turn.get("speaker") or "unknown"), text, ordinal))
    return turns


def select_questions(dataset: list[dict[str, Any]], *, per_category: int = 4,
                     categories: tuple[int, ...] = (1, 2, 3, 4),
                     start_per_category: int = 0) -> list[dict[str, Any]]:
    buckets = defaultdict(list)
    for conversation in dataset:
        for ordinal, qa in enumerate(conversation["qa"]):
            category = int(qa["category"])
            if category not in categories:
                continue
            question_id = f"{conversation['sample_id']}:{ordinal}"
            item = {"id": question_id, "sample_id": conversation["sample_id"],
                    "category": category, "question": str(qa["question"]),
                    "answer": str(qa.get("answer", "")), "evidence": list(qa.get("evidence") or [])}
            key = hashlib.sha256(("care-sourcefirst-locomo-pilot-v1:" + question_id).encode()).hexdigest()
            buckets[category].append((key, item))
    selected = []
    for category in categories:
        candidates = sorted(buckets[category])
        if len(candidates) < start_per_category + per_category:
            raise ValueError(f"Insufficient LoCoMo category {category}")
        selected.extend(item for _, item in candidates[start_per_category:start_per_category + per_category])
    return selected


def _normalize(text: str) -> str:
    text = str(text).replace(",", "").lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def score_locomo(prediction: str, gold: str, category: int) -> float:
    """LoCoMo task_eval/evaluation.py token F1 and category-5 rule."""
    if category == 5:
        p = prediction.lower()
        return float("no information available" in p or "not mentioned" in p)
    if category == 3:
        gold = gold.split(";")[0].strip()
    from nltk.stem import PorterStemmer
    stem = PorterStemmer().stem

    def single(p: str, g: str) -> float:
        a = [stem(word) for word in _normalize(p).split()]
        b = [stem(word) for word in _normalize(g).split()]
        common = sum((Counter(a) & Counter(b)).values())
        if not common:
            return 0.0
        precision, recall = common / len(a), common / len(b)
        return 2 * precision * recall / (precision + recall)

    if category == 1:
        predictions = [part.strip() for part in prediction.split(",")]
        references = [part.strip() for part in gold.split(",")]
        return sum(max(single(p, g) for p in predictions) for g in references) / len(references)
    return single(prediction, gold)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run(dataset_path: Path, output: Path, backend: Any, *, per_category: int = 4,
        categories: tuple[int, ...] = (1, 2, 3, 4), max_new_calls: int = 100,
        start_per_category: int = 0) -> dict[str, Any]:
    dataset_path, output = Path(dataset_path), Path(output)
    payload = dataset_path.read_bytes()
    dataset = json.loads(payload)
    questions = select_questions(dataset, per_category=per_category, categories=categories,
                                 start_per_category=start_per_category)
    identities = {str(conv["sample_id"]): conv for conv in dataset}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": "care-sourcefirst-locomo-pilot-v2", "data_sha256": hashlib.sha256(payload).hexdigest(),
                "selected_ids": [item["id"] for item in questions], "categories": list(categories),
                "per_category": per_category, "start_per_category": start_per_category,
                "arms": ["bm25", "care"],
                "scorer": "LoCoMo task_eval/evaluation.py token-F1 semantics", "official_full_run": False,
                "context": "raw text turns with speaker/date and available image captions"}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Refusing to resume a different CARE pilot manifest")
    _save_json(manifest_path, manifest)
    rows_path = output / "rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    done = {(row["id"], arm) for row in rows for arm in row["systems"]}
    memories = {}
    for item in questions:
        if (item["id"], "bm25") in done and (item["id"], "care") in done:
            continue
        sample = item["sample_id"]
        if sample not in memories:
            memories[sample] = CareSourceFirst(turns_for(identities[sample]["conversation"]),
                                                candidate_limit=18, evidence_chars=10000)
        memory = memories[sample]
        row = next((record for record in rows if record["id"] == item["id"]), None)
        if row is None:
            row = {key: item[key] for key in ("id", "sample_id", "category", "question")}
            row["systems"] = {}
            rows.append(row)
        for arm, mode in (("bm25", "never"), ("care", "risk")):
            if (item["id"], arm) in done:
                continue
            before = len(getattr(backend, "calls", []))
            if before >= max_new_calls:
                _save_json(rows_path, rows)
                raise RuntimeError("CARE pilot model-call cap reached")
            started = time.monotonic()
            try:
                result = memory.answer(item["question"], backend, compile_mode=mode)
                error = None
            except Exception as exc:
                if "model-call cap" in str(exc):
                    _save_json(rows_path, rows)
                    raise
                result, error = {"answer": "", "source_ids": [], "bundle_ids": [], "candidate_ids": [],
                                 "risk_reasons": [], "compiled": mode != "never"}, type(exc).__name__
            result.pop("raw_reader_response", None)
            gold_evidence = set(item["evidence"])
            row["systems"][arm] = {**result, "error": error,
                "f1": score_locomo(result["answer"], item["answer"], item["category"]),
                "evidence_recall": (len(gold_evidence & set(result["bundle_ids"])) / len(gold_evidence)
                                    if gold_evidence else None),
                "new_calls": len(getattr(backend, "calls", [])) - before,
                "seconds": time.monotonic() - started}
            _save_json(rows_path, rows)
            print(json.dumps({"id": item["id"], "arm": arm, "f1": row["systems"][arm]["f1"],
                              "error": error, "calls": row["systems"][arm]["new_calls"]}), flush=True)
    aggregate = {}
    for arm in ("bm25", "care"):
        completed = [row for row in rows if arm in row["systems"]]
        recalls = [row["systems"][arm]["evidence_recall"] for row in completed
                   if row["systems"][arm]["evidence_recall"] is not None]
        aggregate[arm] = {"n": len(completed),
            "mean_f1": sum(row["systems"][arm]["f1"] for row in completed) / len(completed) if completed else 0,
            "errors": sum(row["systems"][arm]["error"] is not None for row in completed),
            "mean_evidence_recall": sum(recalls) / len(recalls) if recalls else None}
    report = {"manifest": manifest, "aggregate": aggregate, "rows": rows}
    _save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--start-per-category", type=int, default=0)
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--max-new-calls", type=int, default=100)
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
                raise RuntimeError("CARE pilot model-call cap reached")
            response = self.client.chat.completions.create(model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(kwargs.get("temperature", 0.0)),
                max_tokens=int(kwargs.get("max_tokens", 192)),
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
                 categories=tuple(args.categories), start_per_category=args.start_per_category,
                 max_new_calls=args.max_new_calls)
    _save_json(args.output / "calls.json", backend.calls)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
