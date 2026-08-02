#!/usr/bin/env bash
#
# ASEM Experiment Runner
# ======================
# Run experiments on the LoCoMo benchmark datasets.
#
# Usage:
#   bash run.sh locomo-benchmark     # Full LoCoMo benchmark on val.jsonl
#   bash run.sh locomo10-quick       # Smoke test (10 QA pairs, ~2 min)
#   bash run.sh locomo10-small       # Small run (100 QA pairs)
#   bash run.sh locomo10-full        # Full locomo10 run (~1990 QA pairs)
#   bash run.sh locomo10-custom      # Custom: all systems, per-category
#
# Environment variables (can also be set in .env):
#   OPENAI_API_KEY    – API key for the LLM provider
#   OPENAI_BASE_URL   – Base URL for the API endpoint

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


mkdir -p "${RESULTS_DIR}" "${DB_DIR}"
ss
MODE="${1:-help}"

case "${MODE}" in

# ------------------------------------------------------------------
# Original LoCoMo benchmark (on pre-processed val.jsonl)
# ------------------------------------------------------------------
locomo-benchmark)
    echo "=== Running LoCoMo benchmark (val.jsonl) ==="
    OPENAI_API_KEY="${API_KEY}" \
    OPENAI_BASE_URL="${BASE_URL}" \
    python scripts/run_locomo_benchmark.py \
        --val     data/training/val.jsonl \
        --config  "${CONFIG}" \
        --results "${RESULTS_DIR}/locomo_openai.json" \
        --db-dir  data/benchmarks/eval_banks_openai \
        --systems NoMemory FullContext SimRetrieval \
        --metrics em rougeL
    ;;

# ------------------------------------------------------------------
# locomo10 experiments (raw LoCoMo conversations → sequential memory)
# ------------------------------------------------------------------

# Quick smoke test — 10 QA pairs, all 7 systems
locomo10-quick)
    echo "=== locomo10 smoke test (10 QA pairs) ==="
    OPENAI_API_KEY="${API_KEY}" \
    OPENAI_BASE_URL="${BASE_URL}" \
    python scripts/run_locomo10_experiments.py \
        --input    datasets/locomo/locomo10.json \
        --config   "${CONFIG}" \
        --results  "${RESULTS_DIR}/locomo10_quick.json" \
        --db-dir   "${DB_DIR}" \
        --limit    10 \
        --metrics  em rougeL
    ;;

# Small run — 100 QA pairs
locomo10-small)
    echo "=== locomo10 small run (100 QA pairs) ==="
    OPENAI_API_KEY="${API_KEY}" \
    OPENAI_BASE_URL="${BASE_URL}" \
    python scripts/run_locomo10_experiments.py \
        --input    datasets/locomo/locomo10.json \
        --config   "${CONFIG}" \
        --results  "${RESULTS_DIR}/locomo10_small.json" \
        --db-dir   "${DB_DIR}" \
        --limit    100 \
        --metrics  em rougeL \
        --max-history-turns 150
    ;;

# Full run — all ~1990 QA pairs, all 7 systems
locomo10-full)
    echo "=== locomo10 FULL run (~1990 QA pairs) ==="
    echo "WARNING: This will make ~14,000 API calls and may take hours."
    echo "Press Ctrl-C within 5s to abort."
    sleep 5
    OPENAI_API_KEY="${API_KEY}" \
    OPENAI_BASE_URL="${BASE_URL}" \
    python scripts/run_locomo10_experiments.py \
        --input    datasets/locomo/locomo10.json \
        --config   "${CONFIG}" \
        --results  "${RESULTS_DIR}/locomo10_full.json" \
        --db-dir   "${DB_DIR}" \
        --metrics  em rougeL bertscore_f1 \
        --per-category \
        --max-history-turns 150
    ;;

# Custom full run — all systems with per-category breakdown
locomo10-custom)
    echo "=== locomo10 custom run ==="
    LIMIT="${LIMIT:-200}"
    SYSTEMS="${SYSTEMS:-NoMemory FullContext SimRetrieval AtomicLinking RLManagerOnly ValueRetrievalOnly ASEM}"
    echo "  Limit: ${LIMIT}"
    echo "  Systems: ${SYSTEMS}"
    OPENAI_API_KEY="${API_KEY}" \
    OPENAI_BASE_URL="${BASE_URL}" \
    python scripts/run_locomo10_experiments.py \
        --input    datasets/locomo/locomo10.json \
        --config   "${CONFIG}" \
        --results  "${RESULTS_DIR}/locomo10_custom.json" \
        --db-dir   "${DB_DIR}" \
        --limit    "${LIMIT}" \
        --systems  ${SYSTEMS} \
        --metrics  em rougeL \
        --per-category \
        --max-history-turns 150
    ;;

# ------------------------------------------------------------------
# Help
# ------------------------------------------------------------------
help|*)
    echo "ASEM Experiment Runner"
    echo "======================"
    echo ""
    echo "Usage: bash run.sh <mode>"
    echo ""
    echo "Modes:"
    echo "  locomo-benchmark   Original LoCoMo benchmark (val.jsonl)"
    echo "  locomo10-quick     Smoke test: 10 QA pairs, all systems"
    echo "  locomo10-small     Small run: 100 QA pairs, all systems"
    echo "  locomo10-full      Full run: ~1990 QA pairs, all systems + per-category"
    echo "  locomo10-custom    Custom run (set LIMIT and SYSTEMS env vars)"
    echo ""
    echo "Environment:"
    echo "  OPENAI_API_KEY     API key (default: from .env)"
    echo "  OPENAI_BASE_URL    API base URL (default: https://api.deepseek.com)"
    echo "  CONFIG             YAML config (default: configs/locomo_openai.yaml)"
    echo "  LIMIT              Max QA pairs for custom mode (default: 200)"
    echo "  SYSTEMS            Systems to run in custom mode (space-separated)"
    ;;

esac
