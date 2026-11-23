"""CARE component ablation worker (docs/CARE_ablation_protocol_20261121.md).

Arms (asem.care.ablation.ARMS) switch off one mechanism of CARE at a time. All use the BM25 top-8 first
bundle, the lexical repair and the same Qwen3-8B reader, so no GPU encoder is needed. A prompt-keyed
memo (asem.care.memo_backend) lets arms that send identical prompts share one physical call while every
arm still records the calls and tokens it would cost alone. Gold labels are read only for scoring.

    first_pass          first read of the care arm, no repair (reference, 1 call)
    care                the full method (reference; must reproduce the saved CARE answers)
    no_cite_req ... repair_no_search   see asem.care.ablation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from asem.care.ablation import ARMS, CareAblation, flags_as_dict
from asem.care.memo_backend import MemoBackend
from scripts.run_care_full_locomo import _append_jsonl, _load_events, selected_questions
from scripts.run_care_locomo import _save_json, score_locomo, turns_for

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "care-ablation-v1"
FIRST = "first_pass"
ARM_NAMES = [FIRST, *ARMS]
REF = ("no information available", "not mentioned")
CODE_FILES = ("asem/care/ablation.py", "asem/care/memo_backend.py", "asem/care/hybrid.py", "asem/care/bridge.py",
              "asem/care/sourcefirst.py", "scripts/run_care_ablation.py")


def refused(text: str) -> bool:
    return any(k in (text or "").lower() for k in REF) or not (text or "").strip()


def load_items(benchmark: str, dataset: list, shard: int, limit: int) -> list[dict]:
    items = []
    if benchmark == "locomo":
        convs = {str(c["sample_id"]): c for c in dataset}
        for q in selected_questions(dataset, shard):
            items.append({"id": q["id"], "history": q["sample_id"], "question": q["question"],
                          "answer": q["answer"], "category": int(q["category"]),
                          "evidence": list(q["evidence"]), "unsupported": int(q["category"]) == 5,
                          "turns_factory": (lambda c=convs[q["sample_id"]]: turns_for(c["conversation"]))})
    else:
        from scripts.run_care_longmemeval import select_records, turns_for_record
        for r in select_records(dataset, shard):
            gold = {f"s{si + 1}:t{ti + 1}" for si, sess in enumerate(r["haystack_sessions"])
                    for ti, m in enumerate(sess) if m.get("has_answer")}
            items.append({"id": str(r["question_id"]), "history": str(r["question_id"]),
                          "question": str(r["question"]), "answer": str(r["answer"]),
                          "category": str(r["question_type"]), "evidence": sorted(gold),
                          "unsupported": str(r["question_id"]).endswith("_abs"),
                          "turns_factory": (lambda rec=r: turns_for_record(rec))})
    return items[:limit] if limit else items


def score(benchmark: str, item: dict, answer: str) -> float:
    if benchmark == "locomo":
        return score_locomo(answer, item["answer"], item["category"])
    from scripts.run_care_longmemeval import diagnostic_score
    return diagnostic_score(answer, item["answer"], item["unsupported"])["score"]


def run(benchmark: str, dataset_path: Path, output: Path, backend: MemoBackend, *, shard_index: int,
        question_limit: int = 0, code_sha: str = "") -> dict:
    payload = dataset_path.read_bytes()
    items = load_items(benchmark, json.loads(payload), shard_index, question_limit)
    manifest = {"protocol": PROTOCOL, "benchmark": benchmark, "dataset_sha256": hashlib.sha256(payload).hexdigest(),
                "shard_index": shard_index, "question_limit": question_limit,
                "question_ids": [i["id"] for i in items], "arms": ARM_NAMES, "reference_arm": "care",
                "flags": flags_as_dict(), "first_bundle": 8, "repair": "lexical",
                "code_sha256": code_sha, "gold_in_model_context": False}
    output.mkdir(parents=True, exist_ok=True)
    mpath = output / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text(encoding="utf-8")) != manifest:
        raise ValueError("refusing to resume a different ablation manifest")
    _save_json(mpath, manifest)
    events_path = output / "events.jsonl"
    events = _load_events(events_path)
    done = {e["id"] for e in events}
    cache: dict[str, dict[str, CareAblation]] = {}
    for item in items:
        if item["id"] in done:
            continue
        if item["history"] not in cache:
            cache.clear()                        # keep one history in memory
            turns = item["turns_factory"]()
            cache[item["history"]] = {name: CareAblation(turns, flags=flags) for name, flags in ARMS.items()}
        arms = cache[item["history"]]
        q, evidence, started = item["question"], set(item["evidence"]), time.monotonic()
        initial = arms["care"].rank(q)[:8]
        systems: dict[str, dict] = {}

        def record(arm: str, answer: str, bundle: list[str], receipts: list[dict], extra: dict) -> None:
            systems[arm] = {"answer": answer, "bundle_ids": bundle, "score": score(benchmark, item, answer),
                            "evidence_recall": (len(evidence & set(bundle)) / len(evidence) if evidence else None),
                            "calls": len(receipts),
                            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in receipts),
                            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in receipts),
                            **extra}

        physical_before = backend.physical
        for name, bridge in arms.items():
            before = len(backend.calls)
            fp = bridge.first_pass(q, backend, [bridge.by_id[t.source_id] for t in initial])
            first_receipts = backend.calls[before:]
            if len(first_receipts) != 1:
                raise ValueError(f"first pass made {len(first_receipts)} calls at {item['id']} ({name})")
            if name == "care":
                f = fp["first"]
                record(FIRST, f["answer"], f["bundle_ids"], first_receipts,
                       {"source_ids": f["source_ids"], "invalid_reason": f["invalid_reason"],
                        "search_queries": f["search_queries"]})
            res = bridge.repair(q, backend, fp, "lexical")
            receipts = backend.calls[before:]
            if len(receipts) != 1 + int(res["escalated"]):
                raise ValueError(f"call count mismatch at {item['id']} ({name})")
            record(name, res["answer"], res["bundle_ids"], receipts,
                   {"escalated": res["escalated"], "source_ids": res["source_ids"],
                    "invalid_reason": res["invalid_reason"], "first_invalid_reason": fp["first"]["invalid_reason"],
                    "n_candidates": len(res["candidate_ids"])})
        event = {"id": item["id"], "history": item["history"], "category": item["category"],
                 "unsupported": item["unsupported"], "seconds": time.monotonic() - started,
                 "physical_calls": backend.physical - physical_before, "systems": systems}
        _append_jsonl(events_path, event)
        events.append(event)
        if len(events) % 20 == 0:
            print(json.dumps({"benchmark": benchmark, "shard": shard_index, "done": len(events),
                              "expected": len(items), "physical_calls": backend.physical}), flush=True)
    by_id = {e["id"]: e for e in events}
    rows = [by_id[i["id"]] for i in items]
    report = {"manifest": manifest, "aggregate": {a: aggregate(rows, a) for a in ARM_NAMES},
              "physical_calls_total": sum(r["physical_calls"] for r in rows), "rows": rows}
    _save_json(output / "report.json", report)
    return report


def aggregate(rows: list[dict], arm: str) -> dict:
    values = [r["systems"][arm] for r in rows]
    answerable = [r["systems"][arm] for r in rows if not r["unsupported"]]
    unsupported = [r["systems"][arm] for r in rows if r["unsupported"]]
    recalls = [v["evidence_recall"] for v in answerable if v["evidence_recall"] is not None]
    return {"n": len(values), "mean_score": sum(v["score"] for v in values) / max(len(values), 1),
            "answerable_mean": sum(v["score"] for v in answerable) / max(len(answerable), 1),
            "refusal_correct": sum(v["score"] == 1 for v in unsupported), "refusal_n": len(unsupported),
            "refused_answerable": sum(refused(v["answer"]) for v in answerable),
            "calls": sum(v["calls"] for v in values),
            "prompt_tokens": sum(v["prompt_tokens"] for v in values),
            "completion_tokens": sum(v["completion_tokens"] for v in values),
            "evidence_recall": sum(recalls) / len(recalls) if recalls else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("locomo", "longmemeval"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--max-new-calls", type=int, default=3000)
    # Accepted for compatibility with scripts.run_care_kaggle.run; unused here.
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--start-per-category", type=int, default=0)
    parser.add_argument("--categories", nargs="+", type=int, default=[1])
    args = parser.parse_args()
    from openai import OpenAI
    code_sha = hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in CODE_FILES)).hexdigest()
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                    timeout=300, max_retries=2)

    def complete(prompt: str, temperature: float, max_tokens: int) -> tuple[str, dict[str, Any]]:
        response = client.chat.completions.create(
            model="asem-atlas-qwen3-8b", messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        receipt = {"prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                   "completion_tokens": getattr(response.usage, "completion_tokens", None),
                   "finish_reason": response.choices[0].finish_reason}
        if receipt["finish_reason"] not in ("stop", "length"):
            raise RuntimeError(f"unexpected finish reason {receipt['finish_reason']}")
        return response.choices[0].message.content or "", receipt

    args.output.mkdir(parents=True, exist_ok=True)
    backend = MemoBackend(complete, args.output / "responses.jsonl", args.max_new_calls)
    report = run(args.benchmark, args.dataset, args.output, backend, shard_index=args.shard_index,
                 question_limit=args.question_limit, code_sha=code_sha)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
