"""Before/after preview of the retrieval-context format (old JSON payload vs new blocks).

Replays the real ASEM retrieval on a frozen `ds_fixed` bank with a probe backend
(no LLM), then renders the SAME notes twice:

  OLD — `json.dumps([AnswerAgent._note_payload(n, in_context=...)])`, the format the
        Qwen3-4B run actually used (UUIDs, escaped unicode, one JSON line).
  NEW — `asem.answer_agent._render_notes_block`, the numbered block format.

Usage
-----
    # before/after for specific questions
    python scratch_diag/render_context_preview.py --conv locomo_0009 --idx 1941 1942
    python scratch_diag/render_context_preview.py --idx 104 157 505      # conv inferred
    # size measurement across the whole benchmark (LLM-free)
    python scratch_diag/render_context_preview.py --measure 60
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from asem.answer_agent import (  # noqa: E402
    AnswerAgent,
    _FACT_CHAR_LIMIT,
    _CONTENT_CHAR_LIMIT,
    _render_notes_block,
)
from asem.backends.langchain_backend import _build_embedder  # noqa: E402
from asem.token_budget import estimate_tokens  # noqa: E402
from eval.systems import build_system  # noqa: E402
from scripts.run_locomo10_experiments import convert_locomo10_to_eval  # noqa: E402

TAG = "ds_fixed"
MODEL = "qwen_qwen3_4b_instruct_2507"
DATASET = os.path.join(ROOT, "datasets", "locomo", "locomo10.json")
BANK_ROOT = os.path.join(ROOT, "static", "memory_banks", "locomo10", TAG)
RES = os.path.join(ROOT, "data", "benchmarks", "results", "static", "locomo10")
EMBED_CFG = {"embedder_provider": "huggingface",
             "embedder_name": "sentence-transformers/all-MiniLM-L6-v2"}


class ProbeEmbed:
    """Embedder only — the retrieval path never calls `generate`."""

    default_max_tokens = 512

    def __init__(self, embedder):
        self.embedder = embedder
        self._cache = {}

    def embed(self, text):
        if text not in self._cache:
            self._cache[text] = np.asarray(self.embedder.embed_documents([text])[0], dtype="float32")
        return self._cache[text]

    def generate(self, *a, **k):
        return "PROBE"


def old_block(notes) -> str:
    """The context exactly as the evaluated run rendered it."""
    in_context = {n.id for n in notes}
    return json.dumps([AnswerAgent._note_payload(n, content_chars=_CONTENT_CHAR_LIMIT,
                                                 in_context=in_context) for n in notes])


def new_block(notes) -> str:
    return _render_notes_block(notes, content_chars=_CONTENT_CHAR_LIMIT, fact_chars=_FACT_CHAR_LIMIT)


def retrieve_for(items, embedder, config, wanted=None):
    """Replay real retrieval for `wanted` idxs; yield (item, notes) per conversation.

    Only the requested questions are answered — the retriever is the costly step
    and the other rows would be discarded anyway.
    """
    by_conv = {}
    for item in items:
        if wanted is not None and item["_idx"] not in wanted:
            continue
        by_conv.setdefault(item["session_id"], []).append(item)

    for conv, conv_items in by_conv.items():
        bank_dir = os.path.join(BANK_ROOT, "ASEM", conv)
        if not os.path.isdir(bank_dir):
            continue
        tmp = tempfile.mkdtemp(prefix="render_")
        for name in os.listdir(bank_dir):
            shutil.copy2(os.path.join(bank_dir, name), os.path.join(tmp, name))
        probe = ProbeEmbed(embedder)
        system = build_system("ASEM", config, tmp, backend=probe)
        bank = system.pipeline.memory_bank
        notes_all = bank.list_notes()
        bank.list_notes = lambda _n=notes_all: _n

        captured = {}
        orig = system.pipeline.retriever.retrieve

        def wrapped(query, M, _orig=orig, _cap=captured):
            res = _orig(query, M)
            _cap["notes"] = res
            return res

        system.pipeline.retriever.retrieve = wrapped
        for item in conv_items:
            captured["notes"] = []
            try:
                system.answer(str(item["query"]), [])
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] {conv} idx={item['_idx']}: {exc}")
            if captured["notes"]:
                yield item, list(captured["notes"])
        bank.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models/qwen3_4b_api.yaml")
    ap.add_argument("--conv", default=None, help="conversation id (e.g. locomo_0009)")
    ap.add_argument("--idx", nargs="*", type=int, default=None)
    ap.add_argument("--measure", type=int, default=0,
                    help="measure N questions spread over the benchmark instead of previewing")
    ap.add_argument("--out", default=os.path.join(ROOT, "scratch_diag", "context_format_preview.md"))
    args = ap.parse_args()

    items = convert_locomo10_to_eval(DATASET, limit=None)
    embedder = _build_embedder(EMBED_CFG)

    if args.measure:
        step = max(1, len(items) // args.measure)
        wanted = set(range(0, len(items), step))
        rows = []
        for item, notes in retrieve_for(items, embedder, args.config, wanted):
            if item["_idx"] not in wanted:
                continue
            old, new = old_block(notes), new_block(notes)
            rows.append((item, len(notes), len(old), len(new),
                         estimate_tokens(old), estimate_tokens(new)))
        if not rows:
            print("nothing measured")
            return
        print(f"measured {len(rows)} questions (ASEM, {TAG} banks)")
        print(f"{'':<22}{'p50':>10}{'mean':>10}{'max':>10}")
        for label, i in (("notes/query", 1), ("OLD chars", 2), ("NEW chars", 3),
                         ("OLD tokens", 4), ("NEW tokens", 5)):
            vals = [r[i] for r in rows]
            print(f"{label:<22}{statistics.median(vals):>10.0f}{statistics.mean(vals):>10.0f}{max(vals):>10.0f}")
        old_t = sum(r[4] for r in rows)
        new_t = sum(r[5] for r in rows)
        print(f"\ncontext tokens: OLD {old_t:,} -> NEW {new_t:,}  "
              f"({100 * (old_t - new_t) / old_t:.1f}% smaller, {old_t / new_t:.2f}x)")
        return

    idxs = args.idx or [1941, 104, 157, 505, 49]
    chosen = [it for it in items if it["_idx"] in idxs
              and (args.conv is None or it["session_id"] == args.conv)]
    if not chosen:
        print("no matching question")
        return

    wanted = {it["_idx"] for it in chosen}
    lines = ["# Retrieval-context format — before / after", "",
             "Cùng một tập note (retrieval thật trên bank `ds_fixed`), hai cách render.", ""]
    for item, notes in retrieve_for(items, embedder, args.config, wanted):
        if item["_idx"] not in wanted:
            continue
        old, new = old_block(notes), new_block(notes)
        preds = {}
        for name in ("ASEM",):
            path = os.path.join(RES, "preds", f"{TAG}__{MODEL}__{name}.jsonl")
            for line in open(path, encoding="utf-8"):
                rec = json.loads(line)
                if rec["idx"] == item["_idx"]:
                    preds[name] = rec
        rec = preds.get("ASEM", {})
        question = item.get("raw_question") or item.get("query", "")
        lines += [
            f"## `#{item['_idx']}` — {question}",
            "",
            f"- gold: `{item.get('answer')}`",
            f"- note lấy về: **{len(notes)}**",
            f"- context cũ: **{len(old)}** ký tự (~{estimate_tokens(old)} token) · "
            f"context mới: **{len(new)}** ký tự (~{estimate_tokens(new)} token) "
            f"→ giảm **{100 * (len(old) - len(new)) / max(1, len(old)):.0f}%**",
            "",
            "<details><summary>OLD — JSON payload (UUID, 1 dòng)</summary>",
            "",
            "```json",
            old[:2000] + (" …" if len(old) > 2000 else ""),
            "```",
            "",
            "</details>",
            "",
            "<details open><summary>NEW — blocked context</summary>",
            "",
            "```text",
            new,
            "```",
            "",
            "</details>",
            "",
        ]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
