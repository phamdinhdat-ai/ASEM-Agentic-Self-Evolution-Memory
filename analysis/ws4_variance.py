"""WS4 — quantify §5's run-to-run variance (the repeat-seed measurement).

WHY THIS EXISTS. During this work the IDENTICAL command produced two different
results:

    run 1:  em 0.35  f1 0.5961  bertscore 0.9047  judge 0.70
    run 2:  em 0.30  f1 0.5762  bertscore 0.9090  judge 0.70

with `answer.temperature: 0.0` configured (the W7/W8 wiring is in place). The
first-run artifact was destroyed by the `display_tag` collision (report §4.3), so
the two answer sets could not be diffed. That left the ablation's +0.05 judge
delta — exactly one question — unmoored from any measured noise floor.

This script reads N repeated runs of the SAME arm and prints:
  * per-metric spread (min/max/range) across repeats
  * the per-question agreement between repeats, using the `idx` field in
    `scores/*.jsonl`, so we can see WHICH questions flip
  * the implied noise floor in QUESTIONS, which is the unit the ablation delta
    is quoted in

R2 discipline: each repeat reports its own row count and an explicit count of
rows with no judge verdict (`judge_correct is None`). A repeat with any such row
is excluded from the spread and named as excluded, rather than silently averaged.

Run:
  $env:PYTHONPATH="."
  & $py analysis/ws4_variance.py --dirs <d1> <d2> ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys


def load_repeat(directory):
    """(aggregate metrics, {idx: judge_correct}) for one run directory."""
    agg = None
    for path in glob.glob(os.path.join(directory, "**", "*.json"), recursive=True):
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if "systems" in d and "ASEM-THG" in d["systems"]:
            agg = d["systems"]["ASEM-THG"]["overall"]
            break
    verdicts = {}
    n_rows = 0
    n_none = 0
    for path in glob.glob(os.path.join(directory, "**", "scores", "*.jsonl"),
                          recursive=True):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                n_rows += 1
                if r.get("judge_correct") is None:
                    n_none += 1
                    continue
                verdicts[r.get("idx")] = bool(r.get("judge_correct"))
    return agg, verdicts, n_rows, n_none


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--label", default="repeated runs of the same arm")
    args = ap.parse_args()

    print("=" * 84)
    print(f"WS4 VARIANCE — {args.label}")
    print("=" * 84)

    reps = []
    for d in args.dirs:
        agg, verdicts, n_rows, n_none = load_repeat(d)
        if agg is None:
            print(f"\n[{d}]  -> no ASEM-THG block found; EXCLUDED")
            continue
        print(f"\n[{d}]")
        print(f"  n={agg.get('n')}  em={agg.get('em')}  f1={agg.get('f1')}  "
              f"rougeL={agg.get('rougeL')}  bs={agg.get('bertscore_f1')}  "
              f"judge={agg.get('judge')}  cov={agg.get('judge_coverage')}")
        print(f"  judge rows={n_rows}  rows with NO verdict={n_none}")
        if n_none:
            print("  !! EXCLUDED from the spread (R2: a rate over errored rows "
                  "is not a rate)")
            continue
        reps.append({"dir": d, "agg": agg, "verdicts": verdicts})

    if len(reps) < 2:
        print("\n<2 usable repeats — no spread computable.")
        return

    print("\n" + "=" * 84)
    print("SPREAD ACROSS REPEATS (same command, same config)")
    print("=" * 84)
    for m in ("em", "f1", "rougeL", "bertscore_f1", "judge"):
        v = [r["agg"][m] for r in reps if r["agg"].get(m) is not None]
        if not v:
            continue
        lo, hi = min(v), max(v)
        print(f"  {m:<14} min={lo:.4f}  max={hi:.4f}  range={hi - lo:.4f}  "
              f"values={[round(x, 4) for x in v]}")

    print("\n" + "=" * 84)
    print("PER-QUESTION AGREEMENT (the unit the ablation delta is quoted in)")
    print("=" * 84)
    base = reps[0]
    for other in reps[1:]:
        common = sorted(set(base["verdicts"]) & set(other["verdicts"]))
        if not common:
            print(f"  {base['dir']} vs {other['dir']}: no common idx — "
                  "cannot compare")
            continue
        flips = [i for i in common
                 if base["verdicts"][i] != other["verdicts"][i]]
        n_yes_b = sum(1 for i in common if base["verdicts"][i])
        n_yes_o = sum(1 for i in common if other["verdicts"][i])
        print(f"\n  {os.path.basename(base['dir'])} vs "
              f"{os.path.basename(other['dir'])}")
        print(f"    common questions : {len(common)}")
        print(f"    judge correct    : {n_yes_b} vs {n_yes_o} "
              f"(delta {n_yes_o - n_yes_b:+d})")
        print(f"    QUESTIONS THAT FLIPPED : {len(flips)}   <- the noise floor")
        if flips:
            print(f"    flipped idx      : {flips[:20]}"
                  + (" ..." if len(flips) > 20 else ""))
        print(f"    agreement rate   : "
              f"{(len(common) - len(flips)) / len(common):.4f}")

    print("\n" + "=" * 84)
    print("INTERPRETATION")
    print("=" * 84)
    print("  The ablation's judge delta on this slice was 1 question (0.05 at")
    print("  n=20). Compare that against the flip count above. If the flip count")
    print("  between two IDENTICAL runs is >= the ablation delta, the ablation is")
    print("  inside the noise floor and must not be reported as a result.")


if __name__ == "__main__":
    main()
