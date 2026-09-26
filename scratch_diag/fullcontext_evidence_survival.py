"""Does the FullContext prompt actually contain the gold evidence turn?

`convert_locomo10_to_eval` builds each QA row's history as ALL turns from session
1 up to the latest session named in `evidence`, and the evidence turns are
INCLUDED. FullContext then trims that history to the answer budget, keeping the
opening `head_turns` and the most recent turns.

This script replays that exact production path with a spy backend (no model, no
API call) and reports, per question category:

  * how many history turns survive the trim;
  * whether every evidence turn's text is still inside the prompt that reaches
    the model;
  * where the last evidence turn sits in the kept sequence.

If the evidence survives almost always, then FullContext's score is not evidence
of long-context robustness — it is a context window that is guaranteed to hold
the answer.

Usage:
  python scratch_diag/fullcontext_evidence_survival.py [CONFIG]
"""
from __future__ import annotations

import os
import re
import statistics
import sys
from collections import defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from asem.token_budget import estimate_tokens  # noqa: E402
from eval.baselines import FullContext  # noqa: E402
from eval.systems import _FULL_CONTEXT_PROMPT, answer_budget_from_config  # noqa: E402
from scripts.run_locomo10_experiments import (  # noqa: E402
    _build_turn_index,
    _turn_to_text,
    _parse_dia_id,
    convert_locomo10_to_eval,
)

DATASET = os.path.join(ROOT, "datasets", "locomo", "locomo10.json")


class SpyBackend:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(dict(kwargs))
        return "a measured answer"

    def embed(self, text):  # pragma: no cover
        raise NotImplementedError


def main() -> None:
    import json

    config = sys.argv[1] if len(sys.argv) > 1 else "configs/models/qwen3_4b_api.yaml"
    config_path = os.path.join(ROOT, config) if not os.path.isabs(config) else config
    max_tokens, window = answer_budget_from_config(config_path)
    print(f"config: {config}   max_tokens={max_tokens}  context_window={window}")
    print()

    raw = json.load(open(DATASET, "r", encoding="utf-8"))
    turn_index_by_conv = {}
    for i, rec in enumerate(raw):
        turn_index_by_conv[f"locomo_{i:04d}"] = _build_turn_index(rec.get("conversation", {}))

    items = convert_locomo10_to_eval(DATASET, limit=None)

    stats = defaultdict(lambda: {
        "n": 0, "all_ev_survive": 0, "trimmed": 0,
        "kept": [], "total": [], "prompt_tokens": [], "last_rank": [], "ev_n": [],
    })

    for item in items:
        conv = item.get("session_id", "")
        cat = item.get("category_name") or f"cat{item.get('category')}"
        history = [str(h) for h in item.get("history", [])]
        query = str(item.get("query", ""))
        evidence = item.get("evidence", []) or []

        backend = SpyBackend()
        system = FullContext(
            backend=backend,
            prompt_template=_FULL_CONTEXT_PROMPT,
            max_tokens=max_tokens,
            context_window=window,
        )
        system.answer(query, history)
        prompt = backend.prompts[0]
        kept = [t for t in history if t in prompt]

        turn_index = turn_index_by_conv.get(conv, {})
        ev_texts = []
        for eid in evidence:
            turn = turn_index.get(str(eid))
            if turn is not None:
                ev_texts.append(_turn_to_text(turn))
        ev_survive = [t for t in ev_texts if t in prompt]

        s = stats[cat]
        s["n"] += 1
        s["ev_n"].append(len(ev_texts))
        if ev_texts and len(ev_survive) == len(ev_texts):
            s["all_ev_survive"] += 1
        if len(kept) < len(history):
            s["trimmed"] += 1
        s["kept"].append(len(kept))
        s["total"].append(len(history))
        s["prompt_tokens"].append(estimate_tokens(prompt))
        if ev_survive:
            last = ev_survive[-1]
            rank = next((i for i, t in enumerate(kept) if t == last), -1)
            s["last_rank"].append(rank / max(1, len(kept) - 1))

    print(f"{'category':<15}{'n':>5}{'hist turns p50':>15}{'kept p50':>10}{'trimmed%':>10}"
          f"{'prompt tok p50':>15}{'ALL evidence kept%':>20}{'last ev pos':>12}")
    for cat, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        pct = 100 * s["all_ev_survive"] / s["n"]
        print(f"{cat:<15}{s['n']:>5}{statistics.median(s['total']):>15.0f}"
              f"{statistics.median(s['kept']):>10.0f}{100*s['trimmed']/s['n']:>9.1f}%"
              f"{statistics.median(s['prompt_tokens']):>15.0f}{pct:>19.1f}%"
              f"{statistics.median(s['last_rank']):>12.2f}")


if __name__ == "__main__":
    main()
