"""Measure the FullContext prompt against the configured answer token budget.

Answers the practical question the budget introduces: *does declaring
`answer.context_window` actually change any real LoCoMo row, or is it inert?*
For every conversation (and the longest history in it) it renders the prompt
twice through the PRODUCTION code path:

  * unbounded  — the whole history (what `context_window: null` still does);
  * budgeted   — `FullContext.answer`, which trims to
    `context_window - max_tokens - SAFETY_MARGIN_TOKENS`.

No model is loaded and no API call is made: a spy backend records the prompt and
returns a canned answer.

Usage:
  python scratch_diag/fullcontext_budget.py [CONFIG]
    CONFIG : backbone config declaring `answer.context_window` (default
             configs/models/deepseek_openai.yaml)
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from asem.token_budget import estimate_tokens  # noqa: E402
from eval.baselines import FullContext, NoMemory  # noqa: E402
from eval.systems import (  # noqa: E402
    _FULL_CONTEXT_PROMPT,
    _NO_MEMORY_PROMPT,
    answer_budget_from_config,
)
from scripts.run_locomo10_experiments import (  # noqa: E402
    convert_locomo10_to_eval,
    group_by_conversation,
)

DATASET = os.path.join(ROOT, "datasets", "locomo", "locomo10.json")


class SpyBackend:
    """Records prompts; returns a canned answer. Never touches a model."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(dict(kwargs))
        return "a measured answer"

    def embed(self, text):  # pragma: no cover - not used by these baselines
        raise NotImplementedError


def _pct(values, p: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))
    return values[idx]


def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/models/deepseek_openai.yaml"
    config_path = os.path.join(ROOT, config) if not os.path.isabs(config) else config
    max_tokens, window = answer_budget_from_config(config_path)
    budget = None if not window else window - (max_tokens or 0) - 64

    print(f"config         : {config}")
    print(f"answer cap     : {max_tokens}")
    print(f"context window : {window}")
    print(f"prompt budget  : {budget}   (tokens)")
    print()

    groups = group_by_conversation(convert_locomo10_to_eval(DATASET, limit=None))

    unbounded_sizes: list[int] = []
    fitted_sizes: list[int] = []
    trimmed_rows = 0
    rows = 0

    header = (f"{'conv':<12}{'QA':>5}{'turns':>7}{'unbounded':>11}"
              f"{'budgeted':>10}{'kept':>7}{'trimmed':>9}")
    print(header)
    print("-" * len(header))

    for group in groups:
        conv_id = str(group[0].get("session_id", ""))
        # The last item carries the longest history in the conversation, so it
        # is the worst case for every earlier question too.
        item = group[-1]
        history = [str(h) for h in item.get("history", [])]
        query = str(item.get("query", ""))

        backend = SpyBackend()
        raw_prompt = _FULL_CONTEXT_PROMPT.format(
            query=query, context="\n".join(history) if history else "(no prior conversation)"
        )
        system = FullContext(
            backend=backend,
            prompt_template=_FULL_CONTEXT_PROMPT,
            max_tokens=max_tokens,
            context_window=window,
        )
        system.answer(query, history)
        fitted = backend.prompts[0]
        cap_sent = backend.kwargs[0].get("max_tokens")

        raw_tokens = estimate_tokens(raw_prompt)
        fit_tokens = estimate_tokens(fitted)
        kept = sum(1 for turn in history if turn in fitted)
        trimmed = "yes" if kept < len(history) else "no"
        trimmed_rows += 1 if kept < len(history) else 0
        rows += 1

        unbounded_sizes.append(raw_tokens)
        fitted_sizes.append(fit_tokens)
        print(f"{conv_id:<12}{len(group):>5}{len(history):>7}{raw_tokens:>11}"
              f"{fit_tokens:>10}{kept:>7}{trimmed:>9}{'' if cap_sent == max_tokens else '  ??cap'}")

    print()
    print(f"rows measured        : {rows} (10 conversations x last question)")
    print(f"unbounded  : min {min(unbounded_sizes):>7}  p50 {_pct(unbounded_sizes, 50):>7}"
          f"  max {max(unbounded_sizes):>7}")
    print(f"budgeted   : min {min(fitted_sizes):>7}  p50 {_pct(fitted_sizes, 50):>7}"
          f"  max {max(fitted_sizes):>7}")
    print(f"rows trimmed by the token budget : {trimmed_rows}/{rows}")
    if budget is not None:
        over = [s for s in fitted_sizes if s > budget]
        print(f"rows still over the {budget}-token budget : {len(over)}")

    # NoMemory has no context at all; its prompt must be tiny and use the cap.
    nm = SpyBackend()
    NoMemory(backend=nm, prompt_template=_NO_MEMORY_PROMPT,
             max_tokens=max_tokens).answer("a question", ["x" * 500])
    print(f"NoMemory prompt tokens : {estimate_tokens(nm.prompts[0])}"
          f"  (cap sent: {nm.kwargs[0].get('max_tokens')})")


if __name__ == "__main__":
    main()
