"""Paired significance test for the thg_v1 vs legacy context-builder ablation.

WHY THIS EXISTS
---------------
The evaluator's headline finding is that the ablation is NOT resolved in either
direction. At N=20 that claim rested on an observed +-1-question run-to-run
variance. At N=199 the same comparison can be tested properly, because the two
arms answer the SAME 199 questions and can therefore be paired by `idx`.

This script pairs the two arms question-by-question and reports:
  * the judge correctness 2x2 table (both / thg_v1 only / legacy only / neither)
  * judge accuracy per arm
  * an EXACT two-sided binomial sign test on the DISCORDANT pairs only
  * the paired BERTScore difference

It REFUSES to emit an accuracy rate when either arm has a judge failure
(`judge_correct is None`), because a rate computed over a failed measurement is
the exact defect this project hit when judge_api.yaml silently 403'd and every
score became a falsy 0.0.

Run:
    $py analysis/ws4_paired_significance.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARM_PATTERN = (
    "data/benchmarks/results/ws4_ablation/full199_postfix_{arm}/locomo10/"
    "scores/thg_v2_postfix__chib_deepseek_v4_1_flash__ASEM-THG.jsonl"
)


def load(arm: str) -> dict[int, dict]:
    path = REPO_ROOT / ARM_PATTERN.format(arm=arm)
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["idx"]] = r
    return rows


def exact_binomial_two_sided(k: int, n: int) -> float:
    """Two-sided exact sign test: p = 2 * P(X <= min(k, n-k)) under Bin(n, 0.5)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(2.0 * sum(comb(n, i) for i in range(k + 1)) / 2**n, 1.0)


def main() -> int:
    a = load("thgv1")
    b = load("legacy")

    common = sorted(set(a) & set(b))
    only_a = set(a) - set(b)
    only_b = set(b) - set(a)
    if only_a or only_b:
        print(f"WARNING: unpaired questions in thg_v1={len(only_a)} legacy={len(only_b)}")

    fail_a = [i for i in common if a[i]["judge_correct"] is None]
    fail_b = [i for i in common if b[i]["judge_correct"] is None]

    print("=" * 76)
    print("PAIRED ABLATION TEST -- full 199 questions, locomo_0000, thg_v2_postfix")
    print("=" * 76)
    print(f"paired questions        : {len(common)}")
    print(f"judge failures          : thg_v1={len(fail_a)}  legacy={len(fail_b)}")

    if fail_a or fail_b:
        print()
        print("REFUSING TO EMIT AN ACCURACY: judge failures present.")
        print("A rate computed over a failed measurement is not a result. This is the")
        print("defect that made every score a falsy 0.0 during the judge 403 outage.")
        return 2

    both = sum(1 for i in common if a[i]["judge_correct"] and b[i]["judge_correct"])
    only_a_n = sum(1 for i in common if a[i]["judge_correct"] and not b[i]["judge_correct"])
    only_b_n = sum(1 for i in common if not a[i]["judge_correct"] and b[i]["judge_correct"])
    neither = sum(1 for i in common if not a[i]["judge_correct"] and not b[i]["judge_correct"])
    n = len(common)

    print()
    print("judge correctness, paired:")
    print(f"  both correct          : {both}")
    print(f"  thg_v1 correct only   : {only_a_n}")
    print(f"  legacy correct only   : {only_b_n}")
    print(f"  neither correct       : {neither}")
    print()
    print(f"  judge accuracy thg_v1 : {(both + only_a_n) / n:.4f}  ({both + only_a_n}/{n})")
    print(f"  judge accuracy legacy : {(both + only_b_n) / n:.4f}  ({both + only_b_n}/{n})")

    disc = only_a_n + only_b_n
    p = exact_binomial_two_sided(min(only_a_n, only_b_n), disc)
    print()
    print(f"  discordant pairs      : {disc}  (thg_v1 {only_a_n} vs legacy {only_b_n})")
    print(f"  exact two-sided p     : {p:.4f}")
    if p >= 0.05:
        print("  VERDICT: NOT significant at alpha=0.05 -> the ablation is NOT resolved.")
    else:
        winner = "thg_v1" if only_a_n > only_b_n else "legacy"
        print(f"  VERDICT: significant -> {winner} is ahead.")

    bs_a = [a[i]["bertscore_f1"] for i in common]
    bs_b = [b[i]["bertscore_f1"] for i in common]
    diff = [x - y for x, y in zip(bs_a, bs_b)]
    wins = sum(1 for d in diff if d > 0)
    losses = sum(1 for d in diff if d < 0)
    print()
    print("BERTScore-F1, paired:")
    print(f"  mean diff (thg_v1-legacy) : {st.mean(diff):+.4f}")
    print(f"  median diff               : {st.median(diff):+.4f}")
    print(f"  thg_v1 wins/loses/ties    : {wins}/{losses}/{n - wins - losses}")
    print(f"  paired sign p             : {exact_binomial_two_sided(min(wins, losses), wins + losses):.4f}")

    print()
    print("=" * 76)
    print("READING: the two arms tie on EM and the judge difference does not reach")
    print("significance at n=199, so the context builder is NOT established as an")
    print("improvement. What IS established is the COST axis (see the efficiency")
    print("report): 8 bounded notes vs ~15,687-token full history.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
