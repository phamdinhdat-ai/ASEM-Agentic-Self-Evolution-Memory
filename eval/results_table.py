"""Generate a Markdown results table from evaluation JSON."""

from __future__ import annotations

import argparse
import json
from typing import Dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate results table")
    parser.add_argument("--results", required=True, help="Evaluation results JSON")
    parser.add_argument("--output", required=True, help="Markdown output path")

    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as handle:
        results: Dict[str, Dict[str, float]] = json.load(handle)

    rows = []
    has_judge = False
    for key, metrics in results.items():
        row = (
            key,
            metrics.get("em", 0.0),
            metrics.get("rougeL", 0.0),
            metrics.get("bertscore_f1", 0.0),
            metrics.get("judge_mean", None),
            metrics.get("judge_pct_perfect", None),
            metrics.get("judge_pct_acceptable", None),
        )
        if row[4] is not None:
            has_judge = True
        rows.append(row)

    rows.sort(key=lambda item: item[0])

    if has_judge:
        lines = [
            "| Run | EM | ROUGE-L | BERTScore-F1 | Judge Mean | Judge % Perfect | Judge % Acceptable |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for run, em, rouge, bert, jmean, jperf, jacc in rows:
            jmean_str = f"{jmean:.3f}" if jmean is not None else "—"
            jperf_str = f"{jperf:.1f}%" if jperf is not None else "—"
            jacc_str = f"{jacc:.1f}%" if jacc is not None else "—"
            lines.append(f"| {run} | {em:.4f} | {rouge:.4f} | {bert:.4f} | {jmean_str} | {jperf_str} | {jacc_str} |")
    else:
        lines = [
            "| Run | EM | ROUGE-L | BERTScore-F1 |",
            "| --- | --- | --- | --- |",
        ]
        for run, em, rouge, bert, _, _, _ in rows:
            lines.append(f"| {run} | {em:.4f} | {rouge:.4f} | {bert:.4f} |")

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"Wrote table to {args.output}")


if __name__ == "__main__":
    main()
