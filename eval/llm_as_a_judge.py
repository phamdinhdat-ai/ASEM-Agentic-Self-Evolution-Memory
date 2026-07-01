"""
LLM-as-a-Judge evaluation for ASEM answer quality.

Uses the same inference backend to score predicted answers against gold answers
on a 1–5 Likert scale, given the conversation context.  Complements traditional
metrics (Exact Match, ROUGE-L, BERTScore-F1) with semantic quality judgments.

Usage (standalone)::

    python eval/llm_as_a_judge.py \
        --preds data/benchmarks/results/preds/locomo_ASEM.jsonl \
        --config configs/locomo_openai.yaml \
        --output data/benchmarks/results/judge_asem.json

Usage (from benchmark runner)::

    python scripts/run_locomo_benchmark.py ... --judge
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from asem.backends.base import InferenceBackend
from asem.logging_utils import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are an expert evaluator for conversational question-answering systems. Your job is to compare a predicted answer to a gold (reference) answer, given the conversation context, and assign a score from 1 to 5.

## Scoring Rubric
- **5 (Perfect)**: The predicted answer is semantically identical to the gold answer. All key facts match. Minor wording differences are acceptable as long as the meaning is exactly the same.
- **4 (Mostly Correct)**: The predicted answer captures most key information but has minor omissions, slightly different wording, or includes a small amount of extra (but not incorrect) detail.
- **3 (Partially Correct)**: The predicted answer captures some key information but misses important details, or is partially correct but partially wrong/incomplete.
- **2 (Mostly Incorrect)**: The predicted answer is only tangentially related to the gold answer. It may mention the right topic but gets the core facts wrong.
- **1 (Completely Wrong)**: The predicted answer is unrelated, contradictory, or nonsensical relative to the gold answer.

## Evaluation Guidelines
- Focus on factual accuracy and semantic equivalence, not exact string matching.
- If the gold answer contains multiple facts, the predicted answer should capture the majority of them for a high score.
- If the predicted answer is more specific than the gold (e.g., gold="Tuesday" pred="last Tuesday"), consider it correct (score 5) as long as it doesn't add incorrect information.
- If the predicted answer is empty or says "I don't know", score it 1 unless the gold is also empty.
- If the predicted answer is a substring of the gold or vice versa and the meaning is preserved, score accordingly (usually 5 or 4).

## Output Format
Output ONLY a single valid JSON object. Do NOT include any commentary, markdown fences, or extra text.
{{"score": <integer 1-5>, "reasoning": "<brief explanation of why this score>"}}

## Input
Conversation Context:
{context}

Question:
{query}

Gold (Reference) Answer:
{gold}

Predicted Answer:
{pred}

## Your Judgment
Output a JSON object with "score" (1-5) and "reasoning" (1-2 sentences)."""


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    """A single judgment with score and reasoning."""
    score: int
    reasoning: str
    query: str = ""
    pred: str = ""
    gold: str = ""


@dataclass
class JudgeMetrics:
    """Aggregated judge metrics across a batch of judgments."""
    mean_score: float
    median_score: float
    pct_perfect: float      # % scoring 5
    pct_acceptable: float    # % scoring >= 4
    pct_poor: float          # % scoring <= 2
    score_distribution: Dict[int, int]  # {score: count}
    num_judgments: int


@dataclass
class LLMJudge:
    """Score predicted answers against gold answers using an LLM.

    Parameters
    ----------
    backend : InferenceBackend
        The inference backend used to generate judgments.
    prompt_template : str, optional
        Custom judge prompt. Must contain {context}, {query}, {gold}, {pred}.
    max_context_chars : int, optional
        Truncate conversation context to this many characters (default 3000).
    """

    backend: InferenceBackend
    prompt_template: str = _JUDGE_PROMPT
    max_context_chars: int = 3000

    def judge(
        self,
        query: str,
        pred: str,
        gold: str,
        context: str = "",
    ) -> JudgeResult:
        """Score a single predicted answer against the gold answer.

        Parameters
        ----------
        query : str
            The question that was asked.
        pred : str
            The system's predicted answer.
        gold : str
            The gold (reference) answer.
        context : str
            The conversation context (may be empty for NoMemory baseline).

        Returns
        -------
        JudgeResult
            Contains score (1–5) and reasoning string.
        """
        # Truncate context if too long
        if len(context) > self.max_context_chars:
            context = context[:self.max_context_chars] + "\n... [truncated]"

        prompt = self.prompt_template.format(
            context=context if context else "(no conversation context available)",
            query=query,
            gold=gold,
            pred=pred if pred else "(empty prediction)",
        )

        raw = self.backend.generate(prompt)
        score, reasoning = self._parse_judgment(raw)

        _logger.debug("judge | query={!r} | score={} | reasoning={!r}",
                      query[:80], score, reasoning[:100])

        return JudgeResult(
            score=score,
            reasoning=reasoning,
            query=query,
            pred=pred,
            gold=gold,
        )

    def judge_batch(
        self,
        examples: List[Dict[str, str]],
        context_key: str = "context",
        query_key: str = "query",
        pred_key: str = "pred",
        gold_key: str = "ref",
    ) -> List[JudgeResult]:
        """Judge a batch of examples.

        Parameters
        ----------
        examples : list of dict
            Each dict must contain keys for context, query, prediction, and gold.
        context_key, query_key, pred_key, gold_key : str
            Keys to use for each field in the example dicts.

        Returns
        -------
        list of JudgeResult
        """
        results: List[JudgeResult] = []
        total = len(examples)
        for i, ex in enumerate(examples):
            result = self.judge(
                query=str(ex.get(query_key, "")),
                pred=str(ex.get(pred_key, "")),
                gold=str(ex.get(gold_key, "")),
                context=str(ex.get(context_key, "")),
            )
            results.append(result)
            if (i + 1) % 10 == 0 or (i + 1) == total:
                _logger.info("judge_batch | {}/{} judged | latest score={}",
                            i + 1, total, result.score)
        return results

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_judgment(raw: str) -> tuple:
        """Parse LLM judgment output. Returns (score, reasoning)."""
        fallback_score = 3
        fallback_reasoning = "Failed to parse judge output"

        # Try to extract JSON from the response
        try:
            # Find the first JSON object in the response
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = raw[start:end + 1]
                data = json.loads(json_str)
            else:
                data = json.loads(raw)
        except json.JSONDecodeError:
            _logger.warning("_parse_judgment | JSON parse failed | raw={!r}", raw[:200])
            return fallback_score, f"{fallback_reasoning}. Raw: {raw[:200]}"

        # Extract score
        score = data.get("score", fallback_score)
        try:
            score = int(score)
        except (ValueError, TypeError):
            score = fallback_score
        score = max(1, min(5, score))  # clamp to 1–5

        # Extract reasoning
        reasoning = str(data.get("reasoning", fallback_reasoning))

        return score, reasoning


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def compute_judge_metrics(results: List[JudgeResult]) -> JudgeMetrics:
    """Compute aggregate metrics from a list of judge results.

    Parameters
    ----------
    results : list of JudgeResult
        Individual judgments from LLMJudge.judge() or judge_batch().

    Returns
    -------
    JudgeMetrics
        Aggregated metrics including mean, distribution, etc.
    """
    if not results:
        return JudgeMetrics(
            mean_score=0.0,
            median_score=0.0,
            pct_perfect=0.0,
            pct_acceptable=0.0,
            pct_poor=0.0,
            score_distribution={},
            num_judgments=0,
        )

    scores = [r.score for r in results]
    n = len(scores)
    mean = sum(scores) / n

    sorted_scores = sorted(scores)
    if n % 2 == 1:
        median = float(sorted_scores[n // 2])
    else:
        median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2.0

    distribution: Dict[int, int] = {}
    for s in range(1, 6):
        distribution[s] = scores.count(s)

    return JudgeMetrics(
        mean_score=round(mean, 3),
        median_score=round(median, 3),
        pct_perfect=round(100.0 * distribution.get(5, 0) / n, 1),
        pct_acceptable=round(100.0 * sum(distribution.get(s, 0) for s in (4, 5)) / n, 1),
        pct_poor=round(100.0 * sum(distribution.get(s, 0) for s in (1, 2)) / n, 1),
        score_distribution=distribution,
        num_judgments=n,
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main() -> None:
    """Standalone entry point for LLM-as-Judge evaluation."""
    import argparse
    import os
    import sys

    # Ensure project root on path
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    from asem.backends import build_backend
    import yaml

    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation")
    parser.add_argument("--preds", required=True,
                       help="Path to predictions JSONL file (from benchmark)")
    parser.add_argument("--config", required=True,
                       help="Path to YAML config for inference backend")
    parser.add_argument("--output", default=None,
                       help="Output path for judge results JSON")
    parser.add_argument("--limit", type=int, default=None,
                       help="Only judge first N examples")
    parser.add_argument("--max-context-chars", type=int, default=3000,
                       help="Truncate context to this many chars")
    args = parser.parse_args()

    # Load config and build backend
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    backend = build_backend(cfg["inference"])

    # Load predictions
    examples = []
    with open(args.preds, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if args.limit:
        examples = examples[:args.limit]

    print(f"Loaded {len(examples)} predictions from {args.preds}")

    # Build context from history (if available in preds)
    # Predictions may not have context — in that case, context is empty
    for ex in examples:
        if "context" not in ex:
            ex["context"] = ""

    judge = LLMJudge(backend=backend, max_context_chars=args.max_context_chars)
    results = judge.judge_batch(examples, context_key="context",
                                query_key="query", pred_key="pred", gold_key="ref")

    metrics = compute_judge_metrics(results)
    print(f"\nLLM-as-Judge Results ({len(results)} judgments):")
    print(f"  Mean Score:      {metrics.mean_score:.3f}")
    print(f"  Median Score:    {metrics.median_score:.3f}")
    print(f"  % Perfect (5):   {metrics.pct_perfect:.1f}%")
    print(f"  % Acceptable (≥4): {metrics.pct_acceptable:.1f}%")
    print(f"  % Poor (≤2):     {metrics.pct_poor:.1f}%")
    print(f"  Distribution:    {metrics.score_distribution}")

    # Save results
    output_data = {
        "metrics": {
            "mean_score": metrics.mean_score,
            "median_score": metrics.median_score,
            "pct_perfect": metrics.pct_perfect,
            "pct_acceptable": metrics.pct_acceptable,
            "pct_poor": metrics.pct_poor,
            "score_distribution": metrics.score_distribution,
            "num_judgments": metrics.num_judgments,
        },
        "judgments": [
            {"score": r.score, "reasoning": r.reasoning,
             "query": r.query, "pred": r.pred, "gold": r.gold}
            for r in results
        ],
    }

    out_path = args.output or args.preds.replace(".jsonl", "_judge.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
