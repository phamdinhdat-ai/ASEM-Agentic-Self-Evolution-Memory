"""Dump the ACTUAL context (prompt) each method sent to the model — wrong and correct cases.

The static eval does not persist prompts, so this script REPLAYS each system on the
frozen `ds_fixed` banks with a **probe backend** that records the prompt and returns a
canned answer. No LLM call, no API key, no GPU beyond the embedder.

Why the replay is faithful
--------------------------
* Bank systems call `read_path(strip_query_prefix(query))` — same query, same retriever,
  same `max_context_notes` / token budget, because `answer_budget_from_config` supplies
  the same numbers the real run used.
* `MemoryBank.list_notes` is cached to one list per bank. The notes are immutable during a
  read-only eval, so `bm25_search` / `search_by_entities` / `ann_search` return exactly
  what they returned in the real run — only ~100x faster.
* The canned answer is a valid `{"selected_ids": [], "answer": ...}` and is NOT an
  abstention, so `ASEMPipeline._recover` never fires: the recorded prompt is the FIRST-pass
  prompt, i.e. the one that produced the answers in `preds/`.

Each dumped case carries an automatic diagnosis:
  INGESTION  — no note in the bank carries the gold answer
  RETRIEVAL  — the gold-bearing note exists but is not in the retrieved context
  GENERATION — the gold-bearing note IS in the context, the answer was still wrong
  NO_CONTEXT — NoMemory (no retrieval at all)
and, for FullContext, whether every evidence turn of the dataset survived the trim.

Usage
-----
    python scratch_diag/dump_context_samples.py                       # default 4 systems, all categories
    python scratch_diag/dump_context_samples.py --wrong-per-cell 3 --right-per-cell 1
    python scratch_diag/dump_context_samples.py --systems ASEM FastASEM
    python scratch_diag/dump_context_samples.py --categories adversarial single_hop
    python scratch_diag/dump_context_samples.py --must-have 49 104 112 157
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from asem.backends.langchain_backend import _build_embedder  # noqa: E402
from asem.token_budget import estimate_tokens  # noqa: E402
from eval.systems import build_system, strip_query_prefix  # noqa: E402
from scripts.run_locomo10_experiments import (  # noqa: E402
    _build_turn_index,
    _turn_to_text,
    convert_locomo10_to_eval,
)

TAG = "ds_fixed"
MODEL = "qwen_qwen3_4b_instruct_2507"
DATASET = os.path.join(ROOT, "datasets", "locomo", "locomo10.json")
BANK_ROOT = os.path.join(ROOT, "static", "memory_banks", "locomo10", TAG)
RES = os.path.join(ROOT, "data", "benchmarks", "results", "static", "locomo10")
ALL_SYSTEMS = ["ASEM", "FastASEM", "FullContext", "NoMemory"]
CATEGORIES = ["adversarial", "temporal", "open_domain", "multi_hop", "single_hop"]
# harness labels stored in preds (category_name) -> corrected label
HARNESS_TO_REAL = {
    "adversarial": "adversarial",
    "temporal": "temporal",
    "commonsense": "open_domain",
    "single_hop": "multi_hop",
    "conversational": "single_hop",
}
BANK_FILES = {"ASEM": "asem.sqlite", "FastASEM": "fast_asem.sqlite"}
NO_BANK = ("NoMemory", "FullContext")
BANK_SYSTEMS = ("ASEM", "FastASEM")
WINDOW_DEFAULT = 8192
MAX_TOKENS_DEFAULT = 512

EMBED_CFG = {"embedder_provider": "huggingface",
             "embedder_name": "sentence-transformers/all-MiniLM-L6-v2"}

STOP = set("""a an the of to in on at for with and or is are was were be been being it its this that
his her their they he she them i you we my your our as by from into about over under had has have did
do does not no but if then than so such very more most much many""".split())

PLACEHOLDER = (
    "PROBE ANSWER: this text is only a placeholder so that no recovery pass is triggered; "
    "it is never parsed, scored, or compared against anything in this dump."
)


def toks(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']+", (text or "").lower())
            if w not in STOP and len(w) > 2}


# --------------------------------------------------------------------------- probe
class ProbeBackend:
    """Records every prompt; returns a canned, non-abstention answer."""

    default_max_tokens = MAX_TOKENS_DEFAULT

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self._vec_cache: dict = {}
        self.prompts: list = []
        self.per_call_kwargs: list = []

    def reset(self) -> None:
        self.prompts = []
        self.per_call_kwargs = []

    def embed(self, text):
        if text not in self._vec_cache:
            vec = self.embedder.embed_documents([text])[0]
            self._vec_cache[text] = np.asarray(vec, dtype="float32")
        return self._vec_cache[text]

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        self.per_call_kwargs.append(dict(kwargs))
        if "selected_ids" in prompt:  # P_distil asks for a JSON object
            return json.dumps({"selected_ids": [], "answer": PLACEHOLDER})
        return PLACEHOLDER


# --------------------------------------------------------------------------- io
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_bank(system: str, conv: str) -> str | None:
    d = os.path.join(BANK_ROOT, system, conv)
    if not os.path.isdir(d):
        return None
    exact = os.path.join(d, BANK_FILES.get(system, ""))
    if os.path.exists(exact):
        return exact
    cands = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".sqlite")]
    return cands[0] if cands else None


def stage_bank(system: str, conv: str) -> str | None:
    """Copy the frozen bank into a temp dir so the dump can never write to it."""
    src = find_bank(system, conv)
    if not src:
        return None
    tmp = tempfile.mkdtemp(prefix="ctxprobe_")
    dst = os.path.join(tmp, os.path.basename(src))
    shutil.copy2(src, dst)
    for side in ("-wal", "-shm"):
        if os.path.exists(src + side):
            shutil.copy2(src + side, dst + side)
    return tmp


# --------------------------------------------------------------------------- selection
def select_cases(preds, scores, systems, categories, n_wrong, n_right, must_have, seed):
    """Deterministic, stratified-by-conversation selection of wrong and correct rows."""
    chosen = defaultdict(set)
    rng = random.Random(seed)
    for s in systems:
        idx_by_cat = defaultdict(list)
        for i, r in preds[s].items():
            real = HARNESS_TO_REAL.get(r.get("category_name", ""), r.get("category_name", ""))
            idx_by_cat[real].append(i)

        for cat in categories:
            for want_correct, want in ((False, n_wrong), (True, n_right)):
                if want <= 0:
                    continue
                pool = [i for i in sorted(idx_by_cat.get(cat, []))
                        if bool(scores[s][i].get("judge_correct")) == want_correct]
                by_conv = defaultdict(list)
                for i in pool:
                    by_conv[preds[s][i]["conversation_id"]].append(i)
                order = sorted(by_conv)
                rng.shuffle(order)
                picked, k = [], 0
                while len(picked) < want and any(by_conv[c] for c in order):
                    c = order[k % len(order)]
                    if by_conv[c]:
                        picked.append(by_conv[c].pop(0))
                    k += 1
                chosen[s].update(picked)

        for i in must_have:
            if i in preds[s]:
                chosen[s].add(i)
    return chosen


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/models/qwen3_4b_api.yaml")
    ap.add_argument("--systems", nargs="+", default=ALL_SYSTEMS)
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--wrong-per-cell", type=int, default=2, help="wrong cases per system x category")
    ap.add_argument("--right-per-cell", type=int, default=1, help="correct cases per system x category")
    ap.add_argument("--must-have", nargs="*", type=int,
                    default=[49, 104, 112, 118, 137, 157, 159, 324, 505])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "scratch_diag", "context_samples"))
    ap.add_argument("--excerpt-head", type=int, default=1600)
    ap.add_argument("--excerpt-tail", type=int, default=600)
    args = ap.parse_args()

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    ctx_dir = os.path.join(out_dir, "contexts")
    os.makedirs(ctx_dir, exist_ok=True)

    preds = {s: {r["idx"]: r for r in load_jsonl(os.path.join(RES, "preds", f"{TAG}__{MODEL}__{s}.jsonl"))}
             for s in args.systems}
    scores = {s: {r["idx"]: r for r in load_jsonl(os.path.join(RES, "scores", f"{TAG}__{MODEL}__{s}.jsonl"))}
              for s in args.systems}

    chosen = select_cases(preds, scores, args.systems, args.categories,
                          args.wrong_per_cell, args.right_per_cell, args.must_have, args.seed)

    eval_items = convert_locomo10_to_eval(DATASET, limit=None)
    items_by_conv = defaultdict(dict)
    for item in eval_items:
        items_by_conv[item["session_id"]][item["_idx"]] = item

    raw = json.load(open(DATASET, encoding="utf-8"))
    turn_index = {f"locomo_{i:04d}": _build_turn_index(rec.get("conversation", {}))
                  for i, rec in enumerate(raw)}

    embedder = None
    records = []
    for conv in sorted(items_by_conv):
        for system in args.systems:
            idxs = sorted(i for i in chosen[system] if i in items_by_conv[conv])
            if not idxs:
                continue

            probe = ProbeBackend(embedder)
            if system in BANK_SYSTEMS:
                if embedder is None:
                    embedder = _build_embedder(EMBED_CFG)
                    probe.embedder = embedder
                db_dir = stage_bank(system, conv)
                if db_dir is None:
                    print(f"  [skip] no bank for {system}/{conv}")
                    continue
            else:
                db_dir = tempfile.mkdtemp(prefix="ctxprobe_empty_")

            built = build_system(system, args.config, db_dir, backend=probe)
            retrieved_log: list = []

            if system in BANK_SYSTEMS:
                bank = built.pipeline.memory_bank
                notes = bank.list_notes()
                bank.list_notes = lambda _n=notes: _n     # exact cache: read-only eval
                blobs = [(n.id, toks(" ".join(str(x) for x in
                                             [n.c, n.K, n.G, n.X, n.entities, n.session_date] if x)))
                         for n in notes]
                orig_retrieve = built.pipeline.retriever.retrieve

                def wrapped(query, M, _orig=orig_retrieve, _log=retrieved_log):
                    res = _orig(query, M)
                    _log.append([n.id for n in res])
                    return res

                built.pipeline.retriever.retrieve = wrapped
            else:
                blobs = []
                notes = []

            for idx in idxs:
                item = items_by_conv[conv][idx]
                query = str(item["query"])
                hist = [str(h) for h in item.get("history", [])] if system in NO_BANK else []
                probe.reset()
                retrieved_log.clear()

                error = None
                try:
                    built.answer(query, hist)
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                prompt = probe.prompts[0] if probe.prompts else "(no prompt captured)"

                row = preds[system][idx]
                sc = scores[system][idx]
                real_cat = HARNESS_TO_REAL.get(row.get("category_name", ""), row.get("category_name", ""))
                correct = bool(sc.get("judge_correct"))

                # ---- diagnosis -------------------------------------------------
                context_ids = []
                bucket = "NO_CONTEXT"
                gold_ids = []
                evidence_in_prompt = None
                if system in BANK_SYSTEMS:
                    rt = toks(row["ref"])
                    gold_ids = sorted(nid for nid, bt in blobs if rt and len(rt & bt) / len(rt) >= 0.6)
                    context_ids = retrieved_log[0] if retrieved_log else []
                    if not gold_ids:
                        bucket = "INGESTION"
                    elif set(gold_ids) & set(context_ids):
                        bucket = "GENERATION"
                    else:
                        bucket = "RETRIEVAL"
                else:
                    ev = item.get("evidence", []) or []
                    ti = turn_index.get(conv, {})
                    ev_texts = [_turn_to_text(ti[e]) for e in ev if e in ti]
                    if ev_texts:
                        evidence_in_prompt = all(t in prompt for t in ev_texts)
                    bucket = "FULL_CONTEXT" if system == "FullContext" else "NO_RETRIEVAL"

                slug = f"{system}__{conv}__{idx:04d}"
                ctx_path = os.path.join(ctx_dir, slug + ".txt")
                header = (
                    f"# system={system} idx={idx} conv={conv} category={real_cat}"
                    f" (harness label={row.get('category_name')})\n"
                    f"# question: {row['question']}\n"
                    f"# gold: {row['ref']}\n"
                    f"# pred (real run): {row['pred']}\n"
                    f"# judge_correct={correct} judge_error={sc.get('judge_error')} "
                    f"f1={row.get('f1'):.4f} em={row.get('em')}\n"
                    f"# diagnosis={bucket} gold_bearing_notes={gold_ids} "
                    f"evidence_in_prompt={evidence_in_prompt}\n"
                    f"# prompt_chars={len(prompt)} prompt_tokens~{estimate_tokens(prompt)}\n"
                    + ("=" * 100) + "\n"
                )
                with open(ctx_path, "w", encoding="utf-8") as fh:
                    fh.write(header + prompt + "\n")

                records.append({
                    "system": system, "idx": idx, "conversation_id": conv,
                    "category": real_cat, "harness_category": row.get("category_name"),
                    "question": row["question"], "query": query,
                    "gold": row["ref"], "pred": row["pred"],
                    "judge_correct": correct, "judge_error": sc.get("judge_error"),
                    "judge_reasoning": sc.get("judge_reasoning", ""),
                    "f1": row.get("f1"), "em": row.get("em"), "rougeL": row.get("rougeL"),
                    "diagnosis": bucket,
                    "gold_bearing_note_ids": gold_ids,
                    "retrieved_note_ids": context_ids,
                    "evidence_in_prompt": evidence_in_prompt,
                    "retriever_stats": dict(getattr(built.pipeline, "retriever", None).stats)
                    if system in BANK_SYSTEMS else {},
                    "prompt_chars": len(prompt),
                    "prompt_tokens_est": estimate_tokens(prompt),
                    "prompt_path": os.path.relpath(ctx_path, out_dir).replace("\\", "/"),
                    "probe_error": error,
                })

            if system in BANK_SYSTEMS:
                built.pipeline.memory_bank.close()

    # ---------------------------------------------------------------- index
    with open(os.path.join(out_dir, "index.jsonl"), "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "tag": TAG, "model": MODEL, "config": args.config,
        "systems": args.systems, "categories": args.categories,
        "wrong_per_cell": args.wrong_per_cell, "right_per_cell": args.right_per_cell,
        "must_have": args.must_have, "seed": args.seed,
        "n_cases": len(records),
        "cases_per_system": dict(Counter(r["system"] for r in records)),
        "cases_per_bucket": dict(Counter(r["diagnosis"] for r in records)),
        "note": ("Contexts were replayed with a probe backend (no LLM). The recorded prompt "
                 "is the first-pass prompt, i.e. the one that produced preds/*.jsonl."),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    write_report(out_dir, records, args, manifest)
    print(f"wrote {len(records)} cases -> {out_dir}")
    print(json.dumps({k: manifest[k] for k in ("n_cases", "cases_per_system", "cases_per_bucket")},
                     ensure_ascii=False, indent=2))


def write_report(out_dir, records, args, manifest):
    def excerpt(text, head, tail):
        if len(text) <= head + tail:
            return text
        return (text[:head] + f"\n\n…[{len(text) - head - tail} chars omitted — full prompt in the "
                f".txt file]…\n\n" + text[-tail:])

    lines = [
        "# Context samples — context thật mà từng phương pháp gửi cho model",
        "",
        f"- Tag bank: `{manifest['tag']}` · model: `{manifest['model']}` · config: `{manifest['config']}`",
        f"- Lấy mẫu: **{args.wrong_per_cell} ca sai + {args.right_per_cell} ca đúng** cho mỗi (hệ thống × loại câu hỏi),"
        f" cộng các idx bắt buộc {args.must_have}. seed={args.seed}.",
        f"- Tổng: **{manifest['n_cases']} ca** — {manifest['cases_per_system']}",
        f"- Chẩn đoán: {manifest['cases_per_bucket']}",
        "",
        "> Context được **replay bằng probe backend** (không gọi LLM): prompt ghi lại chính là prompt",
        "> first-pass đã tạo ra `preds/*.jsonl`. Prompt đầy đủ nằm trong `contexts/<system>__<conv>__<idx>.txt`,",
        "> metadata đầy đủ trong `index.jsonl`.",
        "",
        "Chẩn đoán tự động: **INGESTION** = bank không có note mang gold · **RETRIEVAL** = có note nhưng",
        "không vào context · **GENERATION** = note mang gold đã ở trong context mà vẫn trả lời sai ·",
        "**FULL_CONTEXT** = baseline ngữ cảnh đầy đủ (kèm cờ `evidence_in_prompt`) · **NO_RETRIEVAL** = NoMemory.",
        "",
        "## Tổng hợp",
        "",
        "| System | Loại | Sai | Đúng | Bucket (sai) |",
        "|---|---|---|---|---|",
    ]
    for system in args.systems:
        for cat in args.categories:
            rows = [r for r in records if r["system"] == system and r["category"] == cat]
            if not rows:
                continue
            nw = sum(1 for r in rows if not r["judge_correct"])
            nc = len(rows) - nw
            buckets = Counter(r["diagnosis"] for r in rows if not r["judge_correct"])
            lines.append(f"| {system} | {cat} | {nw} | {nc} | "
                         + (", ".join(f"{k}:{v}" for k, v in buckets.most_common()) or "—") + " |")

    lines += ["", "---", ""]

    # ---- cases where the CONTEXT itself is the cause -------------------------
    retrieval_cases = [r for r in records if r["diagnosis"] == "RETRIEVAL"]
    ingestion_cases = [r for r in records if r["diagnosis"] == "INGESTION"]
    fc_missing = [r for r in records if r["system"] == "FullContext" and r["evidence_in_prompt"] is False]
    fc_all = [r for r in records if r["system"] == "FullContext" and r["evidence_in_prompt"] is not None]

    lines += ["## Context là nguyên nhân — đọc trước", ""]
    lines += [
        "### A. Note mang gold CÓ trong bank nhưng bị bỏ khỏi context (`RETRIEVAL`)",
        "",
        "| System | idx | judge | Loại | Câu hỏi | Gold | pred | note mang gold | note được lấy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(retrieval_cases, key=lambda x: (x["system"], x["idx"])):
        lines.append(
            f"| {r['system']} | {r['idx']} | {'✓' if r['judge_correct'] else '✗'} | {r['category']} "
            f"| {r['question'][:70]} | `{r['gold'][:45]}` "
            f"| `{r['pred'][:60]}` | `{','.join(r['gold_bearing_note_ids'][:4])}` "
            f"| `{','.join(r['retrieved_note_ids'][:4])}` |"
        )
    if not retrieval_cases:
        lines.append("| — | — | — | — | — | — | — | — | — |")

    lines += [
        "",
        "### B. Bank KHÔNG có note mang gold (`INGESTION`) — lỗi tầng ingest",
        "",
        "| System | idx | judge | Loại | Câu hỏi | Gold | pred |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(ingestion_cases, key=lambda x: (x["system"], x["idx"])):
        lines.append(f"| {r['system']} | {r['idx']} | {'✓' if r['judge_correct'] else '✗'} "
                     f"| {r['category']} | {r['question'][:70]} "
                     f"| `{r['gold'][:45]}` | `{r['pred'][:60]}` |")
    if not ingestion_cases:
        lines.append("| — | — | — | — | — | — | — |")

    lines += [
        "",
        "### C. FullContext: evidence của dataset KHÔNG còn trong prompt",
        "",
        f"- Trong {len(fc_all)} ca FullContext được dump: **{len(fc_all) - len(fc_missing)} ca còn evidence**,"
        f" **{len(fc_missing)} ca mất evidence**.",
        "",
        "| idx | judge | Loại | Câu hỏi | Gold | pred |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(fc_missing, key=lambda x: x["idx"]):
        lines.append(f"| {r['idx']} | {'✓' if r['judge_correct'] else '✗'} | {r['category']} "
                     f"| {r['question'][:70]} | `{r['gold'][:45]}` | `{r['pred'][:60]}` |")
    if not fc_missing:
        lines.append("| — | — | — | — | — | — |")

    lines += ["", "---", ""]
    for system in args.systems:
        sys_rows = [r for r in records if r["system"] == system]
        if not sys_rows:
            continue
        lines += [f"# {system}", ""]
        for cat in args.categories:
            cat_rows = [r for r in sys_rows if r["category"] == cat]
            if not cat_rows:
                continue
            lines += [f"## {cat}", ""]
            for want_correct in (False, True):
                for r in sorted(cat_rows, key=lambda x: (x["judge_correct"] != want_correct, x["idx"])):
                    if r["judge_correct"] != want_correct:
                        continue
                    mark = "✓ đúng" if r["judge_correct"] else "✗ SAI"
                    gold_in_ctx = {
                        "GENERATION": "có", "RETRIEVAL": "không", "INGESTION": "không (bank thiếu)",
                    }.get(r["diagnosis"])
                    diag = (f" · diagnosis=**{r['diagnosis']}**" if not r["judge_correct"]
                            else (f" · note mang gold trong context: {gold_in_ctx}" if gold_in_ctx
                                  else f" · {r['diagnosis']}"))
                    lines += [
                        f"### [{mark}] `#{r['idx']}` {r['question'][:110]}",
                        "",
                        f"- gold: `{r['gold']}`",
                        f"- pred (real run): `{r['pred'][:300]}`",
                        f"- judge={r['judge_correct']} error={r['judge_error']} f1={r['f1']:.4f} em={r['em']}"
                        + diag,
                        f"- context: {r['prompt_tokens_est']} token ({r['prompt_chars']} ký tự)"
                        + (f" · note ids: `{r['retrieved_note_ids']}`" if r["retrieved_note_ids"] else "")
                        + (f" · gold-bearing ids: `{r['gold_bearing_note_ids']}`"
                           if r["gold_bearing_note_ids"] else ""),
                        f"- prompt đầy đủ: `{r['prompt_path']}`",
                    ]
                    if r["judge_reasoning"]:
                        lines.append(f"- judge nói: {r['judge_reasoning'][:400]}")
                    if r["evidence_in_prompt"] is not None:
                        lines.append(f"- evidence của dataset có trong prompt: **{r['evidence_in_prompt']}**")
                    if r["probe_error"]:
                        lines.append(f"- ⚠ probe error: {r['probe_error']}")

                    body = ""
                    if os.path.exists(os.path.join(out_dir, r["prompt_path"])):
                        body = open(os.path.join(out_dir, r["prompt_path"]), encoding="utf-8").read()
                        body = body.split("=" * 100 + "\n", 1)[-1]
                    if body:
                        lines += ["", "```text",
                                  excerpt(body, args.excerpt_head, args.excerpt_tail), "```", ""]
        lines.append("")

    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
