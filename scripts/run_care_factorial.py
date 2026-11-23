"""Pre-registered factorial run: first bundle {BM25, dense} x repair {none, lexical, hybrid}.

Every arm uses the same Qwen3-8B reader, CARE prompt, source-binding checks,
10,000-character evidence cap and one-repair ceiling. Arms that share a first
bundle share one physical first call. Baselines run in the same harness:

    bm25_first, bm25_lex (= CARE), bm25_hyb
    dense_first, dense_lex, dense_hyb (= proposed system, fixed before any result)
    rag_dense_top18        plain RAG prompt, dense top-18, one call
    no_memory, full_context_8k   (LongMemEval only; LoCoMo reuses the earlier runs)

Dense scores come from Qwen3-Embedding-0.6B computed on the Kaggle GPU inside
this worker. Gold labels are read only after the model calls, for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from asem.care.graph_memory import GraphConfig, LayeredMemoryGraph
from asem.care.hybrid import CareTwoStage
from scripts.run_care_full_locomo import _append_jsonl, _load_events, selected_questions
from scripts.run_care_locomo import _save_json, score_locomo, turns_for

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "care-factorial-v1"
CARE_ARMS = ("bm25_first", "bm25_lex", "bm25_hyb", "dense_first", "dense_lex", "dense_hyb", "dense_graph")
GRAPH_METHOD, GRAPH_LAMBDA = "mix", 0.7   # chosen on LoCoMo dev recall@18 (analysis/CARE_graph_retrieval_locomo_dev_k18_20261117.json)
RAG_ARM = "rag_dense_top18"
CONTEXT_ARMS = ("no_memory", "full_context_8k")
RAG_TOP_K = 18
EVIDENCE_CHARS = 10000
REF = ("no information available", "not mentioned")


def refused(text: str) -> bool:
    return any(k in (text or "").lower() for k in REF) or not (text or "").strip()


class DenseIndex:
    """Dense ranking of one history's turns with a shared embedder."""

    def __init__(self, turns, embed_docs: Callable, embed_query: Callable):
        self.turns = list(turns)
        self.matrix = embed_docs([f"{t.speaker}: {t.text}" for t in self.turns]).astype(np.float32)
        self.embed_query = embed_query
        self.cache: dict[str, np.ndarray] = {}

    def vector(self, text: str) -> np.ndarray:
        if text not in self.cache:
            self.cache[text] = self.embed_query(text).astype(np.float32)
        return self.cache[text]

    def rank(self, text: str, k: int):
        scores = self.matrix @ self.vector(text)
        order = sorted(range(len(self.turns)), key=lambda i: (-float(scores[i]), i))[:k]
        return [self.turns[i] for i in order]


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
            gold_turns = {f"s{si + 1}:t{ti + 1}" for si, sess in enumerate(r["haystack_sessions"])
                          for ti, m in enumerate(sess) if m.get("has_answer")}
            items.append({"id": str(r["question_id"]), "history": str(r["question_id"]),
                          "question": str(r["question"]), "answer": str(r["answer"]),
                          "category": str(r["question_type"]), "evidence": sorted(gold_turns),
                          "unsupported": str(r["question_id"]).endswith("_abs"),
                          "turns_factory": (lambda rec=r: turns_for_record(rec))})
    return items[:limit] if limit else items


def score(benchmark: str, item: dict, answer: str) -> float:
    if benchmark == "locomo":
        return score_locomo(answer, item["answer"], item["category"])
    from scripts.run_care_longmemeval import diagnostic_score
    return diagnostic_score(answer, item["answer"], item["unsupported"])["score"]


def rag_prompt(tokenizer, index: DenseIndex, question: str) -> tuple[str, list[str]]:
    from scripts.run_care_context_baselines import MAX_ANSWER_TOKENS, SAFETY_TOKENS, WINDOW, token_count
    from scripts.run_care_sim_retrieval import PROMPT
    chosen, chars = [], 0
    for turn in index.rank(question, RAG_TOP_K):
        line = f"[{turn.date}] {turn.source_id} {turn.speaker}: {turn.text}"
        if chars + len(line) > EVIDENCE_CHARS:
            continue
        chosen.append((turn.source_id, line))
        chars += len(line)
    while chosen:
        prompt = PROMPT.format(context="\n".join(line for _, line in chosen), query=question)
        if token_count(tokenizer, prompt) <= WINDOW - MAX_ANSWER_TOKENS - SAFETY_TOKENS:
            return prompt, [sid for sid, _ in chosen]
        chosen.pop()
    raise ValueError("no retrieved turn fits the answer window")


def run(benchmark: str, dataset_path: Path, output: Path, backend: Any, *, embed_docs: Callable,
        embed_query: Callable, tokenizer: Any, shard_index: int, question_limit: int = 0,
        max_new_calls: int = 3000, code_sha: str = "") -> dict:
    payload = dataset_path.read_bytes()
    items = load_items(benchmark, json.loads(payload), shard_index, question_limit)
    arms = list(CARE_ARMS) + [RAG_ARM] + (list(CONTEXT_ARMS) if benchmark == "longmemeval" else [])
    manifest = {"protocol": PROTOCOL, "benchmark": benchmark,
                "dataset_sha256": hashlib.sha256(payload).hexdigest(),
                "shard_index": shard_index, "question_limit": question_limit,
                "question_ids": [i["id"] for i in items], "arms": arms,
                "proposed_arm": "dense_hyb", "reference_arm": "bm25_lex", "graph_arm": "dense_graph",
                "graph_config": {**GraphConfig().__dict__, "method": GRAPH_METHOD, "lambda": GRAPH_LAMBDA},
                "first_bundle": 8, "dense_k_repair": 6, "rag_top_k": RAG_TOP_K,
                "evidence_chars": EVIDENCE_CHARS, "code_sha256": code_sha,
                "gold_in_model_context": False}
    output.mkdir(parents=True, exist_ok=True)
    mpath = output / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text(encoding="utf-8")) != manifest:
        raise ValueError("refusing to resume a different factorial manifest")
    _save_json(mpath, manifest)
    events_path = output / "events.jsonl"
    events = _load_events(events_path)
    done = {e["id"] for e in events}
    histories: dict[str, tuple] = {}
    for item in items:
        if item["id"] in done:
            continue
        if len(backend.calls) >= max_new_calls:
            raise RuntimeError("factorial call cap reached")
        if item["history"] not in histories:
            histories.clear()  # keep one history in memory
            turns = item["turns_factory"]()
            index = DenseIndex(turns, embed_docs, embed_query)
            graph = LayeredMemoryGraph(turns, index.matrix, GraphConfig())
            histories[item["history"]] = (turns, index, graph)
        turns, index, graph = histories[item["history"]]

        def graph_rank(text: str, k: int, _g=graph, _i=index, _t=turns):
            return [_t[j] for j in _g.rank(_i.vector(text), GRAPH_METHOD, k, GRAPH_LAMBDA)]

        bridge = CareTwoStage(turns, dense_rank=index.rank, graph_rank=graph_rank)
        q = item["question"]
        evidence = set(item["evidence"])
        started = time.monotonic()
        systems: dict[str, dict] = {}

        def record(arm: str, answer: str, bundle: list[str], receipts: list[dict], extra: dict) -> None:
            systems[arm] = {"answer": answer, "bundle_ids": bundle,
                            "score": score(benchmark, item, answer),
                            "evidence_recall": (len(evidence & set(bundle)) / len(evidence)
                                                if evidence and bundle is not None else None),
                            "calls": len(receipts),
                            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in receipts),
                            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in receipts),
                            **extra}

        for prefix, initial in (("bm25", bridge.rank(q)[:8]), ("dense", index.rank(q, 8))):
            before = len(backend.calls)
            fp = bridge.first_pass(q, backend, initial)
            first_receipts = backend.calls[before:]
            if len(first_receipts) != 1:
                raise ValueError(f"first pass made {len(first_receipts)} calls at {item['id']}")
            f = fp["first"]
            record(f"{prefix}_first", f["answer"], f["bundle_ids"], first_receipts,
                   {"initial_ids": fp["initial_ids"], "source_ids": f["source_ids"],
                    "invalid_reason": f["invalid_reason"], "search_queries": f["search_queries"]})
            modes = [("lexical", f"{prefix}_lex"), ("hybrid", f"{prefix}_hyb")]
            if prefix == "dense":
                modes.append(("graph", "dense_graph"))
            for mode, arm in modes:
                before = len(backend.calls)
                res = bridge.repair(q, backend, fp, mode)
                extra_receipts = backend.calls[before:]
                if len(extra_receipts) != int(res["escalated"]):
                    raise ValueError(f"repair call count mismatch at {item['id']}")
                record(arm, res["answer"], res["bundle_ids"], first_receipts + extra_receipts,
                       {"escalated": res["escalated"], "source_ids": res["source_ids"],
                        "n_candidates": len(res["candidate_ids"])})
        before = len(backend.calls)
        prompt, rag_ids = rag_prompt(tokenizer, index, q)
        answer = backend.generate(prompt, temperature=0.0, max_tokens=224)
        record(RAG_ARM, answer, rag_ids, backend.calls[before:], {})
        if benchmark == "longmemeval":
            from scripts.run_care_context_baselines import (FULL_CONTEXT_PROMPT, NO_MEMORY_PROMPT,
                                                            fit_context)
            before = len(backend.calls)
            answer = backend.generate(NO_MEMORY_PROMPT.format(query=q), temperature=0.0, max_tokens=224)
            record("no_memory", answer, None, backend.calls[before:], {})
            context, info = fit_context(tokenizer, turns, [q])
            before = len(backend.calls)
            answer = backend.generate(FULL_CONTEXT_PROMPT.format(context=context, query=q),
                                      temperature=0.0, max_tokens=224)
            record("full_context_8k", answer, None, backend.calls[before:],
                   {"included_turns": info["included_turns"], "truncated": info["truncated"]})
        event = {"id": item["id"], "history": item["history"], "category": item["category"],
                 "unsupported": item["unsupported"], "dense_top8": [t.source_id for t in index.rank(q, 8)],
                 "graph_stats": graph.stats(),
                 "seconds": time.monotonic() - started, "systems": systems}
        _append_jsonl(events_path, event)
        events.append(event)
        if len(events) % 20 == 0:
            print(json.dumps({"benchmark": benchmark, "shard": shard_index, "done": len(events),
                              "expected": len(items), "calls": len(backend.calls)}), flush=True)
    by_id = {e["id"]: e for e in events}
    rows = [by_id[i["id"]] for i in items]
    report = {"manifest": manifest, "aggregate": {a: aggregate(rows, a) for a in arms}, "rows": rows}
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
    from transformers import AutoTokenizer
    from scripts.embed_care_turns import DOC_MAX_TOKENS, Embedder, query_text
    from scripts.run_care_context_baselines import MODEL

    # Amendment 2: 2,048 padded tokens per encoder batch (was 8,192). The 8,192 budget ran
    # out of GPU memory beside the vLLM server on lme-s3; see the protocol file.
    embedder = Embedder(token_budget=2048)
    tokenizer = AutoTokenizer.from_pretrained(MODEL["repo"], revision=MODEL["revision"])
    code_sha = hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in (
        "asem/care/hybrid.py", "asem/care/graph_memory.py", "asem/care/bridge.py", "asem/care/sourcefirst.py",
        "scripts/run_care_factorial.py", "scripts/embed_care_turns.py"))).hexdigest()

    class Counted:
        def __init__(self):
            self.client = OpenAI(base_url="http://127.0.0.1:8000/v1",
                                 api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                                 timeout=300, max_retries=2)
            self.log_path = args.output / "calls.jsonl"
            self.calls = _load_events(self.log_path)

        def generate(self, prompt: str, **kwargs) -> str:
            if len(self.calls) >= args.max_new_calls:
                raise RuntimeError("factorial call cap reached")
            response = self.client.chat.completions.create(
                model="asem-atlas-qwen3-8b", messages=[{"role": "user", "content": prompt}],
                temperature=float(kwargs.get("temperature", 0.0)),
                max_tokens=int(kwargs.get("max_tokens", 224)),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            receipt = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                       "completion_tokens": getattr(response.usage, "completion_tokens", None),
                       "finish_reason": response.choices[0].finish_reason}
            if receipt["finish_reason"] not in ("stop", "length"):
                raise RuntimeError(f"unexpected finish reason {receipt['finish_reason']}")
            _append_jsonl(self.log_path, receipt)
            self.calls.append(receipt)
            return response.choices[0].message.content or ""

    args.output.mkdir(parents=True, exist_ok=True)
    report = run(args.benchmark, args.dataset, args.output, Counted(),
                 embed_docs=lambda texts: embedder.encode(texts, DOC_MAX_TOKENS),
                 embed_query=lambda text: embedder.encode([query_text(text)], 512)[0],
                 tokenizer=tokenizer, shard_index=args.shard_index,
                 question_limit=args.question_limit, max_new_calls=args.max_new_calls,
                 code_sha=code_sha)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
