"""Dense similarity retrieval over raw LoCoMo turns, paired with locked CARE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoModel, AutoTokenizer

from scripts.run_care_context_baselines import (MAX_ANSWER_TOKENS, MODEL,
                                                 SAFETY_TOKENS, WINDOW, token_count)
from scripts.run_care_full_locomo import (ModelBackendError, SHARDS, _append_jsonl,
                                          _load_events, selected_questions)
from scripts.run_care_locomo import _save_json, score_locomo, turns_for


ARM = "sim_retrieval"
EMBEDDER = {"repo": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"}
TOP_K = 18
EVIDENCE_CHARS = 10000
PROMPT = (
    "Use the retrieved ORIGINAL conversation turns to answer the question. "
    "Each turn includes its absolute session date and speaker. Resolve relative "
    "dates against that session date. Give only a short answer. If the turns do "
    "not support an answer, reply 'No information available'.\n\n"
    "Retrieved turns:\n{context}\n\nQuestion: {query}\n\nAnswer:"
)


def embed(model, tokenizer, texts: list[str]) -> torch.Tensor:
    vectors = []
    for start in range(0, len(texts), 64):
        batch = tokenizer(texts[start:start + 64], padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu())
    return torch.cat(vectors) if vectors else torch.empty((0, 384))


def retrieval_prompt(turns, matrix, query, embed_model, embed_tokenizer,
                     answer_tokenizer):
    q = embed(embed_model, embed_tokenizer, [query])[0]
    scores = matrix @ q
    order = sorted(range(len(turns)), key=lambda i: (-float(scores[i]), i))[:TOP_K]
    chosen, chars = [], 0
    for i in order:
        turn = turns[i]
        line = f"[{turn.date}] {turn.source_id} {turn.speaker}: {turn.text}"
        if chars + len(line) > EVIDENCE_CHARS:
            continue
        chosen.append((turn.source_id, line))
        chars += len(line)
    while chosen:
        prompt = PROMPT.format(context="\n".join(line for _, line in chosen), query=query)
        if token_count(answer_tokenizer, prompt) <= WINDOW - MAX_ANSWER_TOKENS - SAFETY_TOKENS:
            return prompt, [sid for sid, _ in chosen]
        chosen.pop()
    raise ValueError("No retrieved turn fits answer context")


def summarize(rows):
    values = [r["systems"][ARM] for r in rows]
    answerable = [r["systems"][ARM] for r in rows if r["category"] != 5]
    adversarial = [r["systems"][ARM] for r in rows if r["category"] == 5]
    return {"n": len(values),
            "mean_score": sum(v["score"] for v in values) / len(values) if values else None,
            "answerable_mean_f1": (sum(v["score"] for v in answerable) / len(answerable)
                                   if answerable else None),
            "adversarial_correct": sum(v["score"] == 1 for v in adversarial),
            "adversarial_n": len(adversarial),
            "evidence_recall": (sum(v["evidence_recall"] for v in answerable) / len(answerable)
                                if answerable else None),
            "errors": sum(v["error"] is not None for v in values),
            "calls": sum(v["new_calls"] for v in values),
            "seconds": sum(v["seconds"] for v in values)}


def run(dataset_path: Path, output: Path, *, shard_index: int,
        max_new_calls: int = 1000, question_limit: int = 0):
    from openai import APIError, OpenAI

    dataset_bytes = dataset_path.read_bytes()
    dataset = json.loads(dataset_bytes)
    questions = selected_questions(dataset, shard_index)
    if question_limit:
        questions = questions[:question_limit]
    samples = {sample["sample_id"]: sample for sample in dataset}
    torch.set_num_threads(4)
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDER["repo"], revision=EMBEDDER["revision"])
    embed_model = AutoModel.from_pretrained(EMBEDDER["repo"], revision=EMBEDDER["revision"]).eval().cpu()
    answer_tokenizer = AutoTokenizer.from_pretrained(MODEL["repo"], revision=MODEL["revision"], use_fast=True)
    turns_by_sample, matrices = {}, {}
    for sample_id in SHARDS[shard_index]:
        if not any(q["sample_id"] == sample_id for q in questions):
            continue
        turns = turns_for(samples[sample_id]["conversation"])
        turns_by_sample[sample_id] = turns
        matrices[sample_id] = embed(embed_model, embed_tokenizer,
                                    [f"[{t.date}] {t.speaker}: {t.text}" for t in turns])
    manifest = {
        "protocol": "care-sim-retrieval-locomo-v1",
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "model": MODEL, "embedder": EMBEDDER,
        "shard_index": shard_index, "shard_samples": list(SHARDS[shard_index]),
        "question_limit": question_limit, "question_ids": [q["id"] for q in questions],
        "arm": ARM, "top_k": TOP_K, "evidence_chars": EVIDENCE_CHARS,
        "window_tokens": WINDOW, "max_answer_tokens": MAX_ANSWER_TOKENS,
        "safety_tokens": SAFETY_TOKENS,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "embedding_policy": "mean pool last hidden state, attention mask, L2 normalization, 256 tokens",
        "retrieval_policy": "cosine top-18 raw turns, descending similarity, 10000-character cap",
        "scorer": "LoCoMo token-F1 and category-5 refusal semantics",
        "data_adapter": "text turns plus available image captions, with session dates",
        "gold_in_model_context": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Refusing to resume a different retrieval manifest")
    _save_json(manifest_path, manifest)
    events_path = output / "events.jsonl"
    events = _load_events(events_path)
    done = {e["id"] for e in events}
    if len(done) != len(events):
        raise ValueError("Duplicate checkpoint event")
    client = OpenAI(base_url="http://127.0.0.1:8000/v1",
                    api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                    timeout=180, max_retries=2)
    calls = 0
    for q in questions:
        if q["id"] in done:
            continue
        if calls >= max_new_calls:
            raise RuntimeError("Retrieval model-call cap reached")
        prompt, source_ids = retrieval_prompt(
            turns_by_sample[q["sample_id"]], matrices[q["sample_id"]],
            q["question"], embed_model, embed_tokenizer, answer_tokenizer)
        prompt_tokens = token_count(answer_tokenizer, prompt)
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model="asem-atlas-qwen3-8b", messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=MAX_ANSWER_TOKENS,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        except APIError as exc:
            raise ModelBackendError(f"Model API failed: {type(exc).__name__}: {exc}") from exc
        choice = response.choices[0]
        answer = choice.message.content or ""
        if choice.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"Unexpected finish reason: {choice.finish_reason}")
        gold_ids = set(q["evidence"])
        value = {
            "answer": answer, "error": None,
            "score": score_locomo(answer, q["answer"], q["category"]),
            "source_ids": source_ids,
            "evidence_recall": len(gold_ids.intersection(source_ids)) / len(gold_ids) if gold_ids else 0.0,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_tokens_local": prompt_tokens,
            "prompt_tokens_server": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "finish_reason": choice.finish_reason, "new_calls": 1,
            "seconds": time.monotonic() - started,
        }
        _append_jsonl(events_path, {"id": q["id"], "value": value})
        events.append({"id": q["id"], "value": value})
        done.add(q["id"])
        calls += 1
        if len(events) % 25 == 0:
            print(json.dumps({"shard": shard_index, "completed_predictions": len(events),
                              "expected_predictions": len(questions)}), flush=True)
    by_id = {q["id"]: {"id": q["id"], "sample_id": q["sample_id"],
                       "category": q["category"], "question": q["question"],
                       "systems": {}} for q in questions}
    for event in events:
        by_id[event["id"]]["systems"][ARM] = event["value"]
    rows = [by_id[q["id"]] for q in questions]
    if any(ARM not in row["systems"] for row in rows):
        raise RuntimeError("Incomplete retrieval shard")
    report = {"manifest": manifest, "aggregate": {ARM: summarize(rows)}, "rows": rows}
    _save_json(output / "rows.json", rows)
    _save_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--max-new-calls", type=int, default=1000)
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--start-per-category", type=int, default=0)
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    result = run(args.dataset, args.output, shard_index=args.shard_index,
                 max_new_calls=args.max_new_calls, question_limit=args.question_limit)
    print(json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
