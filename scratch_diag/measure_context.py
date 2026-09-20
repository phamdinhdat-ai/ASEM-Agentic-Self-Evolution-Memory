"""Measure the REAL prompt sizes the retrieval/answer paths produce.

Every system answers with ONE model call, and the answer prompt is trimmed to
`answer.context_window - max_tokens - safety` (see `asem/token_budget.py`). This
probe reports what those prompts actually cost, so a window can be sized from
measurements instead of guesses.

IMPORTANT: the sizes are only the UNTRIMMED ones when the config's
`answer.context_window` is unset or comfortably large (e.g.
`configs/models/deepseek_openai.yaml`, 32768). Pass the small-window config
(`qwen3_4b_openai.yaml`, 8192) and the ASEM/FastASEM prompts shown here are the
post-trim ones instead.

This drives the production code with a SPY backend that records the exact prompt
string and returns a canned response — so it makes ZERO API calls — and reports
the prompt-size distribution per path.

Usage:
  python scratch_diag/measure_context.py [TAG] [CONVS] [CONFIG]
    TAG    : bank tag to read (default ds_fixed)
    CONVS  : how many conversations to sample (default 1)
    CONFIG : backbone config whose answer budget is reported (default
             configs/models/qwen3_4b_openai.yaml)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict

import yaml

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
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

from eval.phase_runner import (  # noqa: E402
    bank_file,
    build_eval_system,
    close_system,
    conversation_index,
    extract_sessions,
    load_raw_dataset,
)
from scripts.run_locomo10_experiments import (  # noqa: E402
    convert_locomo10_to_eval,
    group_by_conversation,
)

# The config only affects prompt RENDERING + the answer budget here (the spy
# backend replaces generation, and the embedder is local).
CONFIG = os.path.join(
    ROOT, "configs", "models",
    sys.argv[3] if len(sys.argv) > 3 else "qwen3_4b_openai.yaml",
)
BANK_ROOT = os.path.join(ROOT, "static", "memory_banks", "locomo10")
DATASET = os.path.join(ROOT, "datasets", "locomo", "locomo10.json")
# Small-window reference the trimming guard was originally sized against.
REF_WINDOW, REF_OUT = 8192, 512


def configured_budget(path: str) -> tuple[int | None, int | None, int | None]:
    """(context_window, effective cap, prompt budget) declared by a config."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    ans = raw.get("answer", {}) or {}
    inf = raw.get("inference", {}) or {}
    block = inf.get(inf.get("backend")) or {}
    window = int(ans.get("context_window") or 0) or None
    cap = int(ans.get("max_tokens") or block.get("max_tokens") or 0) or None
    budget = None if not window else window - (cap or 0) - 64
    return window, cap, budget


class SpyBackend:
    """Wraps a real backend; records prompts, returns canned answers.

    First call finishes normally; a refusal on the FIRST distil call is followed
    by a real answer, which forces the recovery pass so its (wider) prompt is
    measured too.
    """

    def __init__(self, inner):
        self._inner = inner
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if "selected_ids" in prompt:
            # First distil call per question abstains -> forces the recovery pass
            # so its (wider) prompt is measured too.
            if len(self.prompts) % 2 == 1:
                return '{"selected_ids": [], "answer": "I do not know"}'
            return '{"selected_ids": [], "answer": "a measured answer"}'
        return "a measured answer"

    def embed(self, text):
        return self._inner.embed(text)


def est_tokens(text: str) -> tuple[int, int]:
    """(optimistic, conservative) token estimates for a character count."""
    n = len(text)
    return n // 4, n // 3


def pct(values, p):
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))
    return values[idx]


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "ds_fixed"
    n_convs = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # Silence the pipeline's per-query INFO logs; we only want the final table.
    from asem.logging_utils import setup_logging

    try:
        setup_logging(level="ERROR")
    except Exception:  # noqa: BLE001
        import logging
        logging.getLogger().setLevel(logging.ERROR)

    raw = load_raw_dataset(DATASET)
    groups = group_by_conversation(convert_locomo10_to_eval(DATASET, limit=None))

    sizes: dict[str, list[int]] = defaultdict(list)

    # ---------------------------------------------------------------- #
    # FullContext baseline: template + WHOLE history (no truncation).  #
    # Pure string work — no backend, no bank.                          #
    # ---------------------------------------------------------------- #
    fc_item = groups[0][0] if groups else None
    if fc_item:
        from eval.systems import _FULL_CONTEXT_PROMPT

        for item in groups[0]:
            history = item.get("history") or []
            context = "\n".join(str(h) for h in history)
            prompt = _FULL_CONTEXT_PROMPT.format(query=item.get("query", ""), context=context)
            sizes["FullContext (whole history)"].append(len(prompt))

    # ---------------------------------------------------------------- #
    # ASEM / FastASEM: drive the real pipeline with the spy backend.    #
    # ---------------------------------------------------------------- #
    for system in ("ASEM", "FastASEM"):
        for gi, group in enumerate(groups[:n_convs]):
            conv_id = str(group[0].get("session_id", ""))
            src = os.path.join(BANK_ROOT, tag, system, conv_id,
                               "asem.sqlite" if system == "ASEM" else "fast_asem.sqlite")
            if not os.path.exists(src):
                print(f"  (no {system} bank for {conv_id} under tag {tag})")
                continue
            work = tempfile.mkdtemp(prefix=f"measure_{system}_")
            dst = os.path.join(work, os.path.basename(src))
            shutil.copy2(src, dst)
            # The bank must sit in a dir; build_eval_system expects a dir.
            bankdir = os.path.join(work, os.path.basename(src).replace(".sqlite", ""))
            os.makedirs(bankdir, exist_ok=True)
            shutil.move(dst, os.path.join(bankdir, os.path.basename(src)))

            system_obj = build_eval_system(system, CONFIG, bankdir)
            real = system_obj.pipeline.answer_agent.backend
            spy = SpyBackend(real)
            system_obj.pipeline.answer_agent.backend = spy
            try:
                for item in group:
                    # Reset the spy per query so "first call" logic is per question.
                    spy.prompts = []
                    try:
                        system_obj.answer(item.get("query", ""), [])
                    except Exception as exc:  # noqa: BLE001
                        print(f"    [{system} {conv_id}] answer failed: {type(exc).__name__}")
                    for pr in spy.prompts:
                        sizes[f"{system} answer"].append(len(pr))
            finally:
                close_system(system_obj)

    # ---------------------------------------------------------------- #
    # Report                                                           #
    # ---------------------------------------------------------------- #
    window, cap, budget = configured_budget(CONFIG)
    budgets: list[tuple[str, int]] = []
    if budget:
        budgets.append((f"config {window}/{cap}", budget))
    budgets.append((f"reference {REF_WINDOW}/{REF_OUT}", REF_WINDOW - REF_OUT))

    out = []
    out.append("=" * 96)
    out.append(f"PROMPT SIZE BY PATH   tag={tag}  convs={n_convs}  config={os.path.basename(CONFIG)}")
    out.append(f"config window={window or 'unset'}  cap={cap}  -> prompt budget={budget or 'none (no trimming)'}")
    out.append("=" * 96)
    out.append(f"{'path':<30}{'n':>6}{'p50':>8}{'p90':>8}{'p99':>8}{'max':>9}")
    for path, chars in sorted(sizes.items()):
        opt = [c // 4 for c in chars]
        out.append(f"{path:<30}{len(chars):>6}{pct(opt,50):>8}{pct(opt,90):>8}"
                   f"{pct(opt,99):>8}{max(opt):>9}")

    out.append("")
    out.append("tokens at 4 chars/token; over(opt)/over(cons) = prompts above the budget")
    out.append("at 4 and 3 chars/token (a gap means the estimate is too optimistic).")
    for label, b in budgets:
        out.append(f"  budget {b:>6}  ({label})")
        for path, chars in sorted(sizes.items()):
            opt = [c // 4 for c in chars]
            cons = [c // 3 for c in chars]
            over_opt = sum(1 for v in opt if v > b)
            over_cons = sum(1 for v in cons if v > b)
            out.append(f"    {path:<30} over(opt)={over_opt:>5}  over(cons)={over_cons:>5}")

    report = "\n".join(out)
    print(report)
    dest = os.path.join(ROOT, "scratch_diag", "_context_report.txt")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nreport -> {dest}")


if __name__ == "__main__":
    main()
