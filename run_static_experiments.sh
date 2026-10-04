#!/usr/bin/env bash
# Static banks for LoCoMo — ingest ONCE, then evaluate many.
#
#   Phase A  builds one memory bank per system/conversation into
#            static/memory_banks/<dataset>/<tag>/<System>/<locomo_000N>/<bank>.sqlite
#   Phase B  answers full QA from those frozen banks and compares against
#            FullContext + NoMemory. Metrics: em, f1, rougeL, bertscore_f1, judge.
#
# Both phases RESUME: re-run this file and ingest skips banks that already
# exist, while evaluate answers only the QA pairs it has not answered yet.
# Re-running is always safe — it never starts over.

cd "$(dirname "$0")"
export PYTHONPATH=.
export KMP_DUPLICATE_LIB_OK=TRUE      # Windows + faiss/torch
set -a; . ./.env; set +a              # OPENAI_API_KEY / OPENAI_BASE_URL

SYSTEMS="ASEM ASEMv2 FastASEM SimRetrieval AtomicLinking RLManagerOnly ValueRetrievalOnly"

# ---------------------------------------------------------------------------
# Phase A — build the frozen banks (run ONCE; slow — this is the LLM-heavy part)
# ---------------------------------------------------------------------------
python scripts/build_static_banks.py \
    --input datasets/locomo/locomo10.json \
    --tag deepseek \
    --ingest-config configs/models/deepseek_api.yaml \
    --systems $SYSTEMS

# ---------------------------------------------------------------------------
# Phase B — full LoCoMo QA from those banks vs FullContext + NoMemory
# ---------------------------------------------------------------------------
python scripts/run_static_eval.py \
    --input datasets/locomo/locomo10.json \
    --tag deepseek \
    --config configs/models/deepseek_api.yaml \
    --judge-config configs/models/judge_api.yaml \
    --systems $SYSTEMS FullContext NoMemory \
    --metrics em f1 rougeL bertscore_f1 judge \
    --per-category

# Results -> data/benchmarks/results/static/locomo10/deepseek__<model>.{json,md}

# ---------------------------------------------------------------------------
# Variants — copy/paste one at a time
# ---------------------------------------------------------------------------
#
# Cheap smoke test (1 conversation, one system, SIMPLEST path — do this first):
#   python scripts/build_static_banks.py \
#       --tag smoke --ingest-config configs/models/deepseek_api.yaml \
#       --systems FastASEM --limit-conversations 1
#   cat static/memory_banks/locomo10/smoke/manifest.json   # notes must be > 0
#
# Then the same for ASEM (turn-by-turn, slowest):
#   python scripts/build_static_banks.py \
#       --tag smoke --ingest-config configs/models/deepseek_api.yaml \
#       --systems ASEM --limit-conversations 1
#
# Add a system to an existing tag (its banks already exist -> skipped):
#   python scripts/build_static_banks.py --tag deepseek \
#       --ingest-config configs/models/deepseek_api.yaml --systems ASEM
#
# Rebuild one system from scratch after changing its config:
#   python scripts/build_static_banks.py --tag deepseek \
#       --ingest-config configs/models/deepseek_api.yaml --systems ASEM --force
#
# Reuse the SAME frozen banks with a different backbone:
#   python scripts/run_static_eval.py --tag deepseek \
#       --config configs/models/qwen3_4b_api.yaml \
#       --systems FastASEM ASEM FullContext
#
# Only one conversation (banks are per conversation):
#   python scripts/build_static_banks.py --tag deepseek \
#       --ingest-config configs/models/deepseek_api.yaml \
#       --systems $SYSTEMS --conversations 0
#
# What got built, and where the banks are:
#   cat static/memory_banks/locomo10/deepseek/manifest.json
#   ls  static/memory_banks/locomo10/deepseek/*/*/*.sqlite
#
# Drop saved RESULTS for the tag (keeps the banks):
#   rm -f data/benchmarks/results/static/locomo10/deepseek__*
#   rm -f data/benchmarks/results/static/locomo10/{preds,scores,logs}/deepseek__*
