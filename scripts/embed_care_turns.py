"""Phase 0: embed every LoCoMo / LongMemEval-oracle turn and question on a GPU.

Model: Qwen3-Embedding-0.6B (pinned revision), fp16, last-token pooling, left
padding, instruction-prefixed queries (as in the model card). Output is a set
of .npz files that retrieval is then evaluated on offline - no LLM is called.

Documents are ``"<speaker>: <text>"`` so they index the same fields as the
BM25 ranker (speaker + text; dates and IDs are not embedded).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np

MODEL = {"repo": "Qwen/Qwen3-Embedding-0.6B",
         "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"}
TASK = ("Given a question about a past conversation, retrieve the conversation "
        "turns that contain the information needed to answer it")
DOC_MAX_TOKENS = 1024
TOKEN_BUDGET = 16384  # tokens per batch (padded), conservative for a 15 GB T4


def query_text(question: str) -> str:
    return f"Instruct: {TASK}\nQuery:{question}"


def doc_text(speaker: str, text: str) -> str:
    return f"{speaker}: {text}"


def download_locked(lock: dict, cache: Path, name: str) -> Path:
    spec = lock["sources"][name]
    target = cache / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != spec["sha256"]:
        part = target.with_suffix(".part")
        urllib.request.urlretrieve(spec["url"], part)
        if hashlib.sha256(part.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"{name} checksum mismatch")
        part.replace(target)
    return target


class Embedder:
    def __init__(self, device: str = "cuda", token_budget: int = TOKEN_BUDGET):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device, self.token_budget = torch, device, token_budget
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL["repo"], revision=MODEL["revision"], padding_side="left")
        self.model = AutoModel.from_pretrained(
            MODEL["repo"], revision=MODEL["revision"], torch_dtype=torch.float16).to(device).eval()

    def encode(self, texts: list[str], max_length: int) -> np.ndarray:
        torch = self.torch
        lengths = [len(self.tokenizer(t, truncation=True, max_length=max_length)["input_ids"])
                   for t in texts]
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float16)
        start = 0
        while start < len(order):
            width, end = 0, start
            while (end < len(order) and end - start < 256
                   and max(width, lengths[order[end]]) * (end - start + 1) <= self.token_budget):
                width = max(width, lengths[order[end]])
                end += 1
            end = max(end, start + 1)
            idx = order[start:end]
            batch = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                   max_length=max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
                # left padding: the last position is the last real token
                pooled = torch.nn.functional.normalize(hidden[:, -1].float(), p=2, dim=1)
            out[idx] = pooled.cpu().numpy().astype(np.float16)
            start = end
        return out


def sanity_check(embedder: Embedder) -> dict:
    q = embedder.encode([query_text("What is the capital of China?")], 256)[0].astype(np.float32)
    docs = embedder.encode(["The capital of China is Beijing.", "Gravity is a force of attraction."],
                           256).astype(np.float32)
    near, far = float(q @ docs[0]), float(q @ docs[1])
    if not near > far:
        raise RuntimeError(f"Embedding sanity check failed: {near:.3f} <= {far:.3f}")
    return {"relevant": near, "irrelevant": far}


def build_locomo(path: Path) -> tuple[list[dict], list[dict]]:
    from scripts.run_care_locomo import turns_for
    docs, queries = [], []
    for conversation in json.loads(path.read_bytes()):
        sample = str(conversation["sample_id"])
        for turn in turns_for(conversation["conversation"]):
            docs.append({"key": f"{sample}|{turn.source_id}", "text": doc_text(turn.speaker, turn.text)})
        for ordinal, qa in enumerate(conversation["qa"]):
            queries.append({"key": f"{sample}:{ordinal}", "text": str(qa["question"])})
    return docs, queries


def build_longmemeval(path: Path) -> tuple[list[dict], list[dict]]:
    from scripts.run_care_longmemeval import turns_for_record
    docs, queries = [], []
    for record in json.loads(path.read_bytes()):
        qid = str(record["question_id"])
        for turn in turns_for_record(record):
            docs.append({"key": f"{qid}|{turn.source_id}", "text": doc_text(turn.speaker, turn.text)})
        queries.append({"key": qid, "text": str(record["question"])})
    return docs, queries


def embed_dataset(embedder: Embedder, name: str, docs: list[dict], queries: list[dict],
                  output: Path) -> dict:
    started = time.monotonic()
    doc_matrix = embedder.encode([d["text"] for d in docs], DOC_MAX_TOKENS)
    query_matrix = embedder.encode([query_text(q["text"]) for q in queries], 512)
    np.savez_compressed(output / f"{name}.npz",
                        doc_keys=np.array([d["key"] for d in docs]), docs=doc_matrix,
                        query_keys=np.array([q["key"] for q in queries]), queries=query_matrix)
    over = sum(len(embedder.tokenizer(d["text"])["input_ids"]) > DOC_MAX_TOKENS for d in docs)
    return {"docs": len(docs), "queries": len(queries), "dim": int(doc_matrix.shape[1]),
            "docs_truncated_at_max_tokens": over, "seconds": time.monotonic() - started,
            "doc_text_sha256": hashlib.sha256("\n".join(d["key"] + "\t" + d["text"]
                                                        for d in docs).encode()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    embedder = Embedder()
    manifest = {"model": MODEL, "task": TASK, "doc_max_tokens": DOC_MAX_TOKENS,
                "sanity": sanity_check(embedder), "datasets": {}}
    for name, filename, builder in (("locomo", "locomo10.json", build_locomo),
                                    ("longmemeval_oracle", "longmemeval_oracle.json", build_longmemeval)):
        path = download_locked(lock, args.cache, filename)
        docs, queries = builder(path)
        manifest["datasets"][name] = {"source_sha256": lock["sources"][filename]["sha256"],
                                      **embed_dataset(embedder, name, docs, queries, args.output)}
        print(json.dumps({name: manifest["datasets"][name]}), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
