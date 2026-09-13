"""Shared QA evaluation metrics (EM, token-F1, ROUGE-L, BERTScore-F1, judge).

This module is the single source of truth for answer-quality scoring used by the
static-bank evaluation pipeline (``scripts/run_static_eval.py``). It is
deliberately independent of ``eval/phase_runner.py`` so that existing phase
benchmark behaviour and tests cannot regress.

All functions operate on a single ``(prediction, reference)`` pair except
``bertscore_f1_batch`` (batched, because BERTScore is a model forward pass) and
``compute_metrics`` (corpus aggregate). Heavy dependencies (``bert_score``,
``evaluate``) are imported lazily so ``import eval.metrics`` stays cheap.

Canonical metric keys: ``em``, ``em_loose``, ``f1``, ``rougeL``,
``bertscore_f1``, ``judge``. Use :func:`canonical_metrics` to normalise user
input such as ``rouge-l`` / ``token_f1`` / ``bert``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

from asem.logging_utils import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric naming
# ---------------------------------------------------------------------------

#: Canonical metric keys this module can compute.
CANONICAL_METRICS = ("em", "em_loose", "f1", "rougeL", "bertscore_f1", "judge")

#: Accepted spelling variants -> canonical key.
METRIC_ALIASES: Dict[str, str] = {
    "em": "em",
    "exact_match": "em",
    "exactmatch": "em",
    "em_loose": "em_loose",
    "em-loose": "em_loose",
    "loose_em": "em_loose",
    "f1": "f1",
    "token_f1": "f1",
    "token-f1": "f1",
    "squad_f1": "f1",
    "rougel": "rougeL",
    "rouge_l": "rougeL",
    "rouge-l": "rougeL",
    "bertscore": "bertscore_f1",
    "bertscore_f1": "bertscore_f1",
    "bertscore-f1": "bertscore_f1",
    "bert": "bertscore_f1",
    "bert_f1": "bertscore_f1",
    "judge": "judge",
    "llm_judge": "judge",
    "llm-as-a-judge": "judge",
}

#: Metrics that require a (possibly batched) model call rather than pure Python.
MODEL_METRICS = ("bertscore_f1",)


def canonical_metric(name: str) -> str:
    """Map a metric spelling variant to its canonical key.

    Raises:
        ValueError: if the name is not a known metric.
    """
    key = str(name).strip().lower()
    if key in METRIC_ALIASES:
        return METRIC_ALIASES[key]
    raise ValueError(
        f"Unknown metric {name!r}. Known: {sorted(set(METRIC_ALIASES.values()))}"
    )


def canonical_metrics(names: Iterable[str]) -> List[str]:
    """Normalise an iterable of metric names, preserving order and dropping dupes."""
    out: List[str] = []
    for name in names:
        key = canonical_metric(name)
        if key not in out:
            out.append(key)
    return out


def metric_needs_bertscore(names: Sequence[str]) -> bool:
    return "bertscore_f1" in names


def metric_needs_judge(names: Sequence[str]) -> bool:
    return "judge" in names


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_ARTICLE_RE = None  # compiled lazily to keep module import cheap
_PUNCT_RE = None


def _regexes():
    global _ARTICLE_RE, _PUNCT_RE
    if _ARTICLE_RE is None:
        import re

        _ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
        _PUNCT_RE = re.compile(r"[^\w\s]")
    return _ARTICLE_RE, _PUNCT_RE


def normalize_text(text: Any) -> str:
    """Lowercase, drop articles/punctuation and collapse whitespace.

    Matches the normalisation used by ``eval/benchmark_runner.py`` so the new
    numbers stay comparable with previously reported results.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        # LoCoMo gold answers are occasionally ints (years, counts).
        text = str(text)
    lose = text.lower().strip()
    article_re, punct_re = _regexes()
    lose = article_re.sub(" ", lose)
    lose = punct_re.sub(" ", lose)
    return " ".join(lose.split())


# ---------------------------------------------------------------------------
# Per-item metrics
# ---------------------------------------------------------------------------

def exact_match(pred: Any, ref: Any) -> float:
    """1.0 when the normalised prediction equals the normalised reference."""
    norm_p = normalize_text(pred)
    norm_r = normalize_text(ref)
    if not norm_p or not norm_r:
        return 0.0
    return 1.0 if norm_p == norm_r else 0.0


def em_loose(pred: Any, ref: Any) -> float:
    """Legacy substring EM from ``eval/benchmark_runner.py``.

    Counts a hit when the normalised reference contains, or is contained in,
    the normalised prediction. Kept so old and new numbers can be compared.
    """
    norm_p = normalize_text(pred)
    norm_r = normalize_text(ref)
    if not norm_p or not norm_r:
        return 0.0
    if norm_p == norm_r or norm_r in norm_p or norm_p in norm_r:
        return 1.0
    return 0.0


def token_f1(pred: Any, ref: Any) -> float:
    """SQuAD-style token-level F1 over the multiset token overlap.

    Returns 0.0 when either side is empty or the overlap is empty.
    """
    p_tokens = normalize_text(pred).split()
    r_tokens = normalize_text(ref).split()
    if not p_tokens or not r_tokens:
        return 0.0
    n_same = sum((Counter(p_tokens) & Counter(r_tokens)).values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_tokens)
    recall = n_same / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def rouge_l(pred: Any, ref: Any) -> float:
    """LCS-based ROUGE-L F-score on normalised tokens.

    Computed per item (rather than via HF ``evaluate``'s corpus aggregate) so
    per-category and incremental/crash-safe scoring are possible; the mean over
    items is the same corpus quantity.
    """
    p_tokens = normalize_text(pred).split()
    r_tokens = normalize_text(ref).split()
    if not p_tokens or not r_tokens:
        return 0.0

    # Two-row LCS DP keeps memory O(min(m, n)).
    m, n = len(p_tokens), len(r_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        p_i = p_tokens[i - 1]
        for j in range(1, n + 1):
            if p_i == r_tokens[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return (2 * precision * recall) / (precision + recall)


def bertscore_f1_batch(
    preds: Sequence[Any],
    refs: Sequence[Any],
    model_type: str = "roberta-base",
    device: str = "cpu",
    batch_size: int = 32,
) -> List[float]:
    """BERTScore-F1 for a batch of pairs via ``bert_score`` (roberta-base/CPU).

    Never raises: on any failure every score in the batch becomes 0.0 and a
    warning is logged, so a missing model cannot abort an evaluation run.
    """
    n = len(preds)
    if n == 0:
        return []
    str_preds = ["" if p is None else str(p) for p in preds]
    str_refs = ["" if r is None else str(r) for r in refs]
    if n != len(refs):
        _logger.warning(
            "bertscore_f1_batch: length mismatch preds={} refs={} — returning zeros", n, len(refs)
        )
        return [0.0] * n
    try:
        from bert_score import score as _bs_score

        _p, _r, f1 = _bs_score(
            str_preds,
            str_refs,
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            verbose=False,
        )
        return [float(x) for x in f1]
    except Exception as exc:  # noqa: BLE001
        _logger.opt(exception=exc).warning(
            "BERTScore failed (model={}) — using 0.0 for {} pairs", model_type, n
        )
        return [0.0] * n


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def compute_metrics(
    preds: Sequence[Any],
    refs: Sequence[Any],
    metric_names: Sequence[str],
    em_loose_flags: Optional[Sequence[float]] = None,
    bertscore_scores: Optional[Sequence[float]] = None,
    judge_flags: Optional[Sequence[Optional[bool]]] = None,
) -> Dict[str, float]:
    """Aggregate corpus metrics for one system.

    Args:
        preds: System answers.
        refs: Gold answers (aligned with ``preds``).
        metric_names: Any spelling variants accepted by :func:`canonical_metric`.
        em_loose_flags: Pre-computed loose-EM scores (optional; computed here if
            ``em_loose`` is requested and not supplied).
        bertscore_scores: Pre-computed BERTScore-F1 per item (optional). When
            omitted and ``bertscore_f1`` is requested, BERTScore is computed
            here in one batch.
        judge_flags: Per-item judge verdicts; ``None`` marks a judge failure and
            is excluded from the accuracy denominator.

    Returns:
        ``{canonical_metric: score}`` plus ``n`` and, when the judge ran,
        ``judge_coverage`` (fraction of items with a usable verdict).
    """
    wanted = canonical_metrics(metric_names)
    preds = list(preds)
    refs = list(refs)
    if len(preds) != len(refs):
        raise ValueError(f"preds/refs length mismatch: {len(preds)} != {len(refs)}")

    results: Dict[str, float] = {"n": float(len(preds))}
    if not preds:
        for name in wanted:
            results[name] = 0.0
        return results

    if "em" in wanted:
        results["em"] = _mean([exact_match(p, r) for p, r in zip(preds, refs)])

    if "em_loose" in wanted:
        scores = (
            list(em_loose_flags)
            if em_loose_flags is not None
            else [em_loose(p, r) for p, r in zip(preds, refs)]
        )
        results["em_loose"] = _mean(scores)

    if "f1" in wanted:
        results["f1"] = _mean([token_f1(p, r) for p, r in zip(preds, refs)])

    if "rougeL" in wanted:
        results["rougeL"] = _mean([rouge_l(p, r) for p, r in zip(preds, refs)])

    if "bertscore_f1" in wanted:
        if bertscore_scores is not None and len(bertscore_scores) == len(preds):
            scores = [float(x) for x in bertscore_scores]
        else:
            scores = bertscore_f1_batch(preds, refs)
        results["bertscore_f1"] = _mean(scores)

    if "judge" in wanted:
        if judge_flags is None:
            results["judge"] = 0.0
            results["judge_coverage"] = 0.0
        else:
            flags = list(judge_flags)
            usable = [1.0 if f else 0.0 for f in flags if f is not None]
            results["judge"] = _mean(usable)
            results["judge_coverage"] = (
                float(len(usable) / len(flags)) if flags else 0.0
            )

    return results


def per_category_metrics(
    preds: Sequence[Any],
    refs: Sequence[Any],
    categories: Sequence[str],
    metric_names: Sequence[str],
    em_loose_flags: Optional[Sequence[float]] = None,
    bertscore_scores: Optional[Sequence[float]] = None,
    judge_flags: Optional[Sequence[Optional[bool]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Group by category label and compute :func:`compute_metrics` per group."""
    buckets: Dict[str, Dict[str, List[Any]]] = {}
    for i, cat in enumerate(categories):
        bucket = buckets.setdefault(str(cat) or "unknown", {"preds": [], "refs": [], "loose": [], "bert": [], "judge": []})
        bucket["preds"].append(preds[i])
        bucket["refs"].append(refs[i])
        if em_loose_flags is not None:
            bucket["loose"].append(em_loose_flags[i])
        if bertscore_scores is not None:
            bucket["bert"].append(bertscore_scores[i])
        if judge_flags is not None:
            bucket["judge"].append(judge_flags[i])

    out: Dict[str, Dict[str, float]] = {}
    for cat, bucket in sorted(buckets.items()):
        out[cat] = compute_metrics(
            bucket["preds"],
            bucket["refs"],
            metric_names,
            em_loose_flags=bucket["loose"] or None,
            bertscore_scores=bucket["bert"] or None,
            judge_flags=bucket["judge"] or None,
        )
    return out


__all__ = [
    "CANONICAL_METRICS",
    "METRIC_ALIASES",
    "MODEL_METRICS",
    "bertscore_f1_batch",
    "canonical_metric",
    "canonical_metrics",
    "compute_metrics",
    "em_loose",
    "exact_match",
    "metric_needs_bertscore",
    "metric_needs_judge",
    "normalize_text",
    "per_category_metrics",
    "rouge_l",
    "token_f1",
]
