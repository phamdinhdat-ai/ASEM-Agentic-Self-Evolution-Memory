"""Evaluation harness for ASEM and baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import evaluate as hf_evaluate

from asem.logging_utils import setup_logging


@dataclass
class DatasetPaths:
    """Filesystem locations for evaluation datasets."""
    longmemeval: str
    locomo: str
    personalmembench: str


@dataclass
class EvalConfig:
    """Evaluation configuration for metrics and outputs."""
    datasets: DatasetPaths
    results_path: str
    metrics: List[str] = field(default_factory=lambda: ["em", "rougeL", "bertscore_f1"])
    bertscore_lang: str = "en"


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Run data/benchmarks/download_datasets.py"
        )
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_longmemeval(path: str) -> List[Dict[str, Any]]:
    return _load_jsonl(path)


def load_locomo(path: str) -> List[Dict[str, Any]]:
    return _load_jsonl(path)


def load_personalmembench(path: str) -> List[Dict[str, Any]]:
    return _load_jsonl(path)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match(preds: Iterable[str], refs: Iterable[str]) -> float:
    matches = [
        1.0 if _normalize(p) == _normalize(r) else 0.0
        for p, r in zip(preds, refs)
    ]
    if not matches:
        return 0.0
    return sum(matches) / len(matches)


def compute_metrics(
    preds: List[str],
    refs: List[str],
    config: EvalConfig,
) -> Dict[str, float]:
    results: Dict[str, float] = {}

    if "em" in config.metrics:
        results["em"] = exact_match(preds, refs)

    if "rougeL" in config.metrics:
        rouge = hf_evaluate.load("rouge")
        scores = rouge.compute(predictions=preds, references=refs)
        results["rougeL"] = float(scores.get("rougeL", 0.0))

    if "bertscore_f1" in config.metrics:
        bert = hf_evaluate.load("bertscore")
        scores = bert.compute(predictions=preds, references=refs, lang=config.bertscore_lang)
        results["bertscore_f1"] = float(sum(scores["f1"]) / len(scores["f1"]))

    return results


def _extract_history(item: Dict[str, Any]) -> List[str]:
    history = item.get("history")
    if isinstance(history, list):
        return [str(h) for h in history]
    return []


def run_baseline(
    baseline: Any,
    dataset: List[Dict[str, Any]],
    config: EvalConfig,
) -> Dict[str, float]:
    preds: List[str] = []
    refs: List[str] = []

    for item in dataset:
        query = str(item.get("query", ""))
        answer = str(item.get("answer", ""))
        history = _extract_history(item)
        preds.append(baseline.answer(query, history))
        refs.append(answer)

    return compute_metrics(preds, refs, config)


def run_all(
    baselines: Mapping[str, Any],
    datasets: Mapping[str, List[Dict[str, Any]]],
    config: EvalConfig,
) -> Dict[str, Dict[str, float]]:
    setup_logging()
    results: Dict[str, Dict[str, float]] = {}

    for dataset_name, dataset in datasets.items():
        for baseline_name, baseline in baselines.items():
            key = f"{dataset_name}/{baseline_name}"
            results[key] = run_baseline(baseline, dataset, config)

    os.makedirs(os.path.dirname(config.results_path), exist_ok=True)
    with open(config.results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return results


def load_datasets(config: EvalConfig) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "longmemeval": load_longmemeval(config.datasets.longmemeval),
        "locomo": load_locomo(config.datasets.locomo),
        "personalmembench": load_personalmembench(config.datasets.personalmembench),
    }
