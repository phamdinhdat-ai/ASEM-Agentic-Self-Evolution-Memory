"""LLM-as-a-Judge evaluation for ASEM answer quality.

Compares AI-generated responses against expected (gold) answers using an LLM
judge.  Provides per-question verdicts (is_correct, reasoning, error) and
aggregate metrics across all questions and per category.

Usage (standalone):
    from eval.llm_as_a_judge import LLMJudge, compute_judge_metrics

    judge = LLMJudge(backend)
    verdict = judge.judge(question, expected_answer, ai_response)
    print(verdict.is_correct, verdict.reasoning)

Usage (batch):
    verdicts = judge.judge_batch(questions, expected_answers, ai_responses)
    metrics = compute_judge_metrics(verdicts)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from asem.backends.base import InferenceBackend
from asem.logging_utils import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Judge prompt template with few-shot examples for calibration
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator. Your task is to compare an AI-generated "
    "answer against a gold (expected) answer and determine whether the AI "
    "answer is CORRECT.\n\n"
    "Rules:\n"
    "- Mark is_correct=true if the AI answer contains the KEY INFORMATION "
    "  from the gold answer, even if phrased differently or containing extra "
    "  (non-contradictory) detail.\n"
    "- Mark is_correct=false if the AI answer is missing key facts, "
    "  contradicts the gold answer, or hallucinates incorrect information.\n"
    "- For numeric/temporal answers, accept equivalent expressions "
    "  (e.g., '2022' ≈ 'the year 2022').\n"
    "- Provide a 1-2 sentence reasoning explaining your decision.\n"
    "- If the AI answer is incorrect, set the 'error' field to one of:\n"
    "  'missing_info', 'contradiction', 'hallucination', or 'vague'.\n"
    "- If correct, set 'error' to null.\n\n"
    "Few-shot examples:\n\n"
    "Example 1:\n"
    'Gold: "His old Prius and his new Prius."\n'
    'AI: "Evan had his old Prius break down, and his new Prius also broke '
    'down shortly after purchase."\n'
    'Output: {"is_correct": true, "reasoning": "Both vehicles are correctly '
    'identified as broken, matching the gold answer.", "error": null}\n\n'
    "Example 2:\n"
    'Gold: "7 May 2023"\n'
    'AI: "Caroline went to the support group in early May 2023."\n'
    'Output: {"is_correct": true, "reasoning": "Early May 2023 is '
    'semantically equivalent to 7 May 2023 in this context.", '
    '"error": null}\n\n'
    "Example 3:\n"
    'Gold: "Psychology, counseling certification"\n'
    'AI: "Caroline would likely pursue a career in social work."\n'
    'Output: {"is_correct": false, "reasoning": "The AI answer mentions '
    'social work instead of psychology/counseling, missing the specific '
    'fields from the gold answer.", "error": "missing_info"}\n\n'
    "Example 4:\n"
    'Gold: "4 years"\n'
    'AI: "She has had her friends for about 3 years."\n'
    'Output: {"is_correct": false, "reasoning": "The duration is off by '
    'one year compared to the gold answer of 4 years.", '
    '"error": "contradiction"}\n'
)

_JUDGE_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Gold (expected) answer: {expected_answer}\n\n"
    "AI-generated answer: {ai_response}\n\n"
    "Output your evaluation as JSON with keys: is_correct (bool), "
    "reasoning (string), error (string or null)."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class JudgeVerdict:
    """Single evaluation verdict from the LLM judge."""

    is_correct: bool
    reasoning: str
    error: Optional[str] = None  # 'missing_info', 'contradiction', 'hallucination', 'vague', or None

    # Metadata (set by caller)
    question: str = ""
    expected_answer: str = ""
    ai_response: str = ""
    conversation_id: str = ""
    question_type: str = ""
    category: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_correct": self.is_correct,
            "reasoning": self.reasoning,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

@dataclass
class LLMJudge:
    """LLM-based evaluator for comparing AI answers against gold answers."""

    backend: InferenceBackend
    system_prompt: str = _JUDGE_SYSTEM_PROMPT
    user_template: str = _JUDGE_USER_TEMPLATE
    max_retries: int = 2

    def judge(
        self,
        question: str,
        expected_answer: str,
        ai_response: str,
        conversation_id: str = "",
        question_type: str = "",
        category: int = 0,
    ) -> JudgeVerdict:
        """Evaluate a single AI response against the gold answer.

        Args:
            question: The original question text.
            expected_answer: The gold/reference answer.
            ai_response: The AI system's generated answer.
            conversation_id: Optional conversation identifier for metadata.
            question_type: Optional question type label (single_hop, temporal, etc.).
            category: Optional numeric category.

        Returns:
            JudgeVerdict with is_correct, reasoning, and error fields.
        """
        _logger.debug("LLMJudge.judge | question={!r} | expected={!r} | ai={!r}",
                     question[:100], expected_answer[:100], ai_response[:100])

        user_prompt = self.user_template.format(
            question=question,
            expected_answer=expected_answer,
            ai_response=ai_response,
        )

        full_prompt = f"{self.system_prompt}\n\n{user_prompt}"

        for attempt in range(self.max_retries + 1):
            raw = self.backend.generate(full_prompt)
            verdict = self._parse_verdict(raw)

            if verdict is not None:
                verdict.question = question
                verdict.expected_answer = expected_answer
                verdict.ai_response = ai_response
                verdict.conversation_id = conversation_id
                verdict.question_type = question_type
                verdict.category = category
                _logger.debug("LLMJudge.judge → correct={} error={}",
                             verdict.is_correct, verdict.error)
                return verdict

            _logger.warning("LLMJudge.judge | parse failed (attempt {}/{}), raw={!r}",
                           attempt + 1, self.max_retries + 1, raw[:200])

        # Fallback: mark as incorrect with parse error
        fallback = JudgeVerdict(
            is_correct=False,
            reasoning="Judge failed to produce valid JSON after retries.",
            error="vague",
            question=question,
            expected_answer=expected_answer,
            ai_response=ai_response,
            conversation_id=conversation_id,
            question_type=question_type,
            category=category,
        )
        return fallback

    def judge_batch(
        self,
        questions: List[str],
        expected_answers: List[str],
        ai_responses: List[str],
        conversation_ids: Optional[List[str]] = None,
        question_types: Optional[List[str]] = None,
        categories: Optional[List[int]] = None,
    ) -> List[JudgeVerdict]:
        """Evaluate a batch of AI responses.

        Args:
            questions: List of question strings.
            expected_answers: List of gold answers.
            ai_responses: List of AI-generated answers.
            conversation_ids: Optional conversation IDs (parallel to questions).
            question_types: Optional question type labels.
            categories: Optional numeric categories.

        Returns:
            List of JudgeVerdict, one per question.
        """
        n = len(questions)
        if conversation_ids is None:
            conversation_ids = [""] * n
        if question_types is None:
            question_types = [""] * n
        if categories is None:
            categories = [0] * n

        verdicts: List[JudgeVerdict] = []
        for i in range(n):
            v = self.judge(
                question=questions[i],
                expected_answer=expected_answers[i],
                ai_response=ai_responses[i],
                conversation_id=conversation_ids[i],
                question_type=question_types[i],
                category=categories[i],
            )
            verdicts.append(v)
            if (i + 1) % 10 == 0:
                _logger.info("LLMJudge.judge_batch | progress {}/{}", i + 1, n)

        return verdicts

    def _parse_verdict(self, raw: str) -> Optional[JudgeVerdict]:
        """Extract the JSON verdict from the LLM output.

        Handles cases where the model wraps JSON in markdown fences or
        includes extra text before/after the JSON object.
        """
        # Try to extract JSON from markdown code fences first
        cleaned = raw.strip()
        if "```json" in cleaned:
            start = cleaned.find("```json") + 7
            end = cleaned.find("```", start)
            if end > start:
                cleaned = cleaned[start:end].strip()
        elif "```" in cleaned:
            start = cleaned.find("```") + 3
            end = cleaned.find("```", start)
            if end > start:
                cleaned = cleaned[start:end].strip()

        # Try to find a JSON object
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            cleaned = cleaned[brace_start:brace_end + 1]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        is_correct = data.get("is_correct")
        reasoning = data.get("reasoning")
        error = data.get("error")

        if not isinstance(is_correct, bool) or not isinstance(reasoning, str):
            return None

        if error is not None and not isinstance(error, str):
            error = str(error)

        return JudgeVerdict(
            is_correct=is_correct,
            reasoning=reasoning,
            error=error,
        )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class JudgeMetrics:
    """Aggregate metrics from judge verdicts."""

    judge_mean: float  # % of answers marked correct
    judge_pct_perfect: float  # % correct AND no error
    judge_pct_acceptable: float  # % correct (even with minor errors)
    total: int
    correct: int
    perfect: int
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)


def compute_judge_metrics(
    verdicts: List[JudgeVerdict],
    per_category: bool = False,
) -> JudgeMetrics:
    """Compute aggregate judge metrics from a list of verdicts.

    Args:
        verdicts: List of JudgeVerdict objects.
        per_category: If True, compute breakdown by question_type.

    Returns:
        JudgeMetrics with aggregate statistics.
    """
    total = len(verdicts)
    if total == 0:
        return JudgeMetrics(
            judge_mean=0.0,
            judge_pct_perfect=0.0,
            judge_pct_acceptable=0.0,
            total=0,
            correct=0,
            perfect=0,
        )

    correct = sum(1 for v in verdicts if v.is_correct)
    perfect = sum(1 for v in verdicts if v.is_correct and v.error is None)

    by_category: Dict[str, Dict[str, float]] = {}
    if per_category:
        from collections import defaultdict
        cat_verdicts: Dict[str, List[JudgeVerdict]] = defaultdict(list)
        for v in verdicts:
            key = v.question_type or f"cat{v.category}"
            cat_verdicts[key].append(v)

        for cat_name, cat_vs in cat_verdicts.items():
            n = len(cat_vs)
            if n == 0:
                continue
            cat_correct = sum(1 for cv in cat_vs if cv.is_correct)
            cat_perfect = sum(1 for cv in cat_vs if cv.is_correct and cv.error is None)
            by_category[cat_name] = {
                "judge_mean": cat_correct / n,
                "judge_pct_perfect": cat_perfect / n * 100,
                "judge_pct_acceptable": cat_correct / n * 100,
                "total": n,
            }

    return JudgeMetrics(
        judge_mean=correct / total,
        judge_pct_perfect=perfect / total * 100,
        judge_pct_acceptable=correct / total * 100,
        total=total,
        correct=correct,
        perfect=perfect,
        by_category=by_category,
    )


def verdicts_to_list(verdicts: List[JudgeVerdict]) -> List[Dict[str, Any]]:
    """Convert verdicts to a list of dicts for JSON serialization."""
    return [
        {
            "conversation_id": v.conversation_id,
            "question_type": v.question_type,
            "category": v.category,
            "question": v.question,
            "expected_answer": v.expected_answer,
            "ai_response": v.ai_response,
            "evaluation": v.to_dict(),
        }
        for v in verdicts
    ]
