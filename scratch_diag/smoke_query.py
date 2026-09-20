"""Retrieve + answer against the smoke-ingested bank, showing the REAL prompt.

Builds the production ASEM system on an already-ingested bank, then for a few
questions prints:
  * the retrieved graph nodes (speaker / entities / typed relations)
  * the exact context block the answer agent renders
  * the model's final answer vs the gold reference

Usage:
  python scratch_diag/smoke_query.py <db_dir> [conv_index]
"""
from __future__ import annotations

import json
import os
import re
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_env(path: str = ".env") -> None:
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return
    with open(full, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_env()

from eval.phase_runner import build_eval_system, close_system  # noqa: E402
from eval.systems import strip_query_prefix  # noqa: E402

PRED_DIR = os.path.join(ROOT, "data", "benchmarks", "results", "static", "locomo10", "preds")
CONFIG = os.path.join(ROOT, "configs", "models", "deepseek_openai.yaml")

QUESTIONS = [
    "When did Melanie run a charity race?",
    "What is Caroline's relationship status?",
    "What did Melanie paint recently?",
    "What is Caroline's identity?",
    "How long have Mel and her husband been married?",
    "What did the charity race raise awareness for?",
]


def load_refs(conv: str) -> dict[str, dict]:
    path = os.path.join(PRED_DIR, "ds_nothink__deepseek_v4_flash__ASEM.jsonl")
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["conversation_id"] == conv:
                out[r["question"]] = r
    return out


def main() -> None:
    db_dir = sys.argv[1]
    conv = f"locomo_{int(sys.argv[2]):04d}" if len(sys.argv) > 2 else "locomo_0000"
    refs = load_refs(conv)

    system = build_eval_system("ASEM", CONFIG, db_dir)
    backend = system.pipeline.answer_agent.backend

    captured: dict[str, str] = {}
    original = backend.generate

    def spy(prompt, **kwargs):
        captured["prompt"] = prompt
        response = original(prompt, **kwargs)
        captured["response"] = response
        return response

    backend.generate = spy  # type: ignore[method-assign]

    print(f"bank={db_dir}  notes={system.bank_size}")
    print(f"prefix strip sanity: {strip_query_prefix('Conversation between Caroline and Melanie. Question: When X?')!r}")

    prompts_path = os.path.join(ROOT, "scratch_diag", "_smoke_prompts.txt")
    report_path = os.path.join(ROOT, "scratch_diag", "_smoke_report.md")
    prompts_fh = open(prompts_path, "w", encoding="utf-8")
    report = [f"# Smoke query report — {db_dir}\n"]

    try:
        for q in QUESTIONS:
            r = refs.get(q, {})
            enriched = r.get("query") or f"Conversation between Caroline and Melanie. Question: {q}"
            captured.clear()
            ans = system.answer(enriched, [])

            prompt = captured.get("prompt", "")
            response = captured.get("response", "")
            prompts_fh.write("\n" + "=" * 90 + f"\nQUERY: {q}\n" + "=" * 90 + "\n")
            prompts_fh.write(prompt + "\n--- RAW RESPONSE ---\n" + response + "\n")

            candidates = []
            for line in prompt.splitlines():
                if line.strip().startswith('[{"id"'):
                    try:
                        candidates = json.loads(line.strip())
                    except Exception:
                        pass
                    break
            selected = set()
            m = re.search(r'"selected_ids"\s*:\s*\[([^\]]*)\]', response)
            if m:
                selected = {s.strip().strip('"') for s in m.group(1).split(",") if s.strip()}
            by_id = {c.get("id"): i + 1 for i, c in enumerate(candidates)}

            print("\n" + "=" * 78)
            print(f"Q   : {q}")
            print(f"ref : {r.get('ref')!r}")
            print(f"old : {(r.get('pred') or '')[:70]!r}")
            print(f"NEW : {ans!r}")
            print(f"  candidates={len(candidates)}  selected={len(selected)}")
            for i, c in enumerate(candidates):
                mark = "*" if c.get("id") in selected else " "
                rels = [f"{x.get('relation')}->[{by_id.get(x.get('target_id'), '?')}]"
                        for x in (c.get("relations") or []) if x.get("target_id") in by_id]
                print(f"   {mark}[{i+1}] speaker={c.get('speaker')!r} entities={c.get('entities')}")
                if rels:
                    print(f"        relations: {'; '.join(rels)}")

            report.append(f"\n## {q}\n- ref: `{r.get('ref')}`\n- old: `{(r.get('pred') or '')[:90]}`\n- **new: `{ans}`**\n")
            for i, c in enumerate(candidates):
                mark = "x" if c.get("id") in selected else " "
                report.append(f"  - [{mark}] speaker={c.get('speaker')} entities={c.get('entities')}")
    finally:
        prompts_fh.close()
        close_system(system)

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))
    print(f"\nfull prompts -> {prompts_path}")
    print(f"report       -> {report_path}")


if __name__ == "__main__":
    main()
