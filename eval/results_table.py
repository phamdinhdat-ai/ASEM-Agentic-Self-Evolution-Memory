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
    for key, metrics in results.items():
        rows.append((key, metrics.get("em", 0.0), metrics.get("rougeL", 0.0), metrics.get("bertscore_f1", 0.0)))

    rows.sort(key=lambda item: item[0])

    lines = [
        "| Run | EM | ROUGE-L | BERTScore-F1 |",
        "| --- | --- | --- | --- |",
    ]
    for run, em, rouge, bert in rows:
        lines.append(f"| {run} | {em:.4f} | {rouge:.4f} | {bert:.4f} |")

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"Wrote table to {args.output}")


if __name__ == "__main__":
    main()
