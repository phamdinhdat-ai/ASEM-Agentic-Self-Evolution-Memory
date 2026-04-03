"""Prepare human evaluation annotation sets."""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_predictions(path: str) -> Dict[str, str]:
    preds = {}
    for item in _load_jsonl(path):
        preds[str(item["id"])] = str(item["prediction"])
    return preds


def build_annotation_set(
    dataset_path: str,
    prediction_paths: List[str],
    seed: int,
) -> List[Dict[str, Any]]:
    data = _load_jsonl(dataset_path)
    systems = [(_load_predictions(path), path) for path in prediction_paths]

    rng = random.Random(seed)
    annotation_items = []

    for item in data:
        item_id = str(item.get("id"))
        systems_payload = []
        for preds, label in systems:
            answer = preds.get(item_id, "")
            systems_payload.append({"label": label, "answer": answer})
        rng.shuffle(systems_payload)

        annotation_items.append(
            {
                "id": item_id,
                "query": item.get("query", ""),
                "history": item.get("history", []),
                "reference": item.get("answer", ""),
                "systems": systems_payload,
            }
        )

    return annotation_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build human eval annotation set")
    parser.add_argument("--dataset", required=True, help="Dataset JSONL path")
    parser.add_argument(
        "--predictions",
        required=True,
        nargs=2,
        help="Prediction JSONL files for two systems",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    annotations = build_annotation_set(args.dataset, args.predictions, args.seed)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(annotations, handle, indent=2)

    print(f"Wrote annotations to {args.output}")


if __name__ == "__main__":
    main()
