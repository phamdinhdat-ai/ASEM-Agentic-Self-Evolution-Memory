#!/usr/bin/env bash
# Phase 2: Static Memory Evaluation with Full Metrics
set -e

TAG="${1:-thg_exp1}"
EVAL_CONFIG="${2:-configs/models/qwen3_4b_api.yaml}"
JUDGE_CONFIG="${3:-configs/models/judge_api.yaml}"
SYSTEMS="${4:-ASEM-THG FullContext NoMemory}"

echo "================================================================="
echo "  PHASE 2: RETRIEVAL & QA EVALUATION (FULL METRICS)"
echo "  Tag: $TAG"
echo "  Backbone Model Config: $EVAL_CONFIG"
echo "  Judge Model Config: $JUDGE_CONFIG"
echo "  Systems: $SYSTEMS"
echo "  Metrics: EM, F1, ROUGE-L, BERTScore-F1, LLM-as-a-Judge"
echo "================================================================="

python scripts/run_static_eval.py \
    --tag "$TAG" \
    --config "$EVAL_CONFIG" \
    --judge-config "$JUDGE_CONFIG" \
    --systems $SYSTEMS \
    --metrics em f1 rougeL bertscore_f1 judge \
    --per-category \
    --per-conversation

