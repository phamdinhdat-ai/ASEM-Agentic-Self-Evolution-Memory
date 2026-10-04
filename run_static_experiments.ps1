# Static banks for LoCoMo — ingest ONCE, then evaluate many.
#
#   Phase A  builds one memory bank per system/conversation into
#            static/memory_banks/<dataset>/<tag>/<System>/<locomo_000N>/<bank>.sqlite
#   Phase B  answers full QA from those frozen banks and compares against
#            FullContext + NoMemory. Metrics: em, f1, rougeL, bertscore_f1, judge.
#
# Both phases RESUME: re-run this file and ingest skips banks that already
# exist, while evaluate answers only the QA pairs it has not answered yet.
#
# NOTE: the Python scripts load .env themselves (OPENAI_API_KEY / OPENAI_BASE_URL).

Set-Location $PSScriptRoot
$env:PYTHONPATH = "."
$env:KMP_DUPLICATE_LIB_OK = "TRUE"    # Windows + faiss/torch

# ---- Phase A — build the frozen banks (run ONCE; slow) --------------------
python scripts/build_static_banks.py `
    --input datasets/locomo/locomo10.json `
    --tag deepseek `
    --ingest-config configs/models/deepseek_api.yaml `
    --systems ASEM ASEMv2 FastASEM SimRetrieval AtomicLinking RLManagerOnly ValueRetrievalOnly

# ---- Phase B — full LoCoMo QA vs FullContext + NoMemory -------------------
python scripts/run_static_eval.py `
    --input datasets/locomo/locomo10.json `
    --tag deepseek `
    --config configs/models/deepseek_api.yaml `
    --judge-config configs/models/judge_api.yaml `
    --systems ASEM ASEMv2 FastASEM SimRetrieval AtomicLinking RLManagerOnly ValueRetrievalOnly FullContext NoMemory `
    --metrics em f1 rougeL bertscore_f1 judge `
    --per-category

# Results -> data\benchmarks\results\static\locomo10\deepseek__<model>.json / .md
#
# Variants — copy/paste one at a time:
#
#   Cheap smoke test (1 conversation, one system — do this FIRST):
#     python scripts/build_static_banks.py --tag smoke `
#         --ingest-config configs/models/deepseek_api.yaml `
#         --systems FastASEM --limit-conversations 1
#     Get-Content static\memory_banks\locomo10\smoke\manifest.json
#
#   Add a system to an existing tag (existing banks are skipped):
#     python scripts/build_static_banks.py --tag deepseek `
#         --ingest-config configs/models/deepseek_api.yaml --systems ASEM
#
#   Rebuild one system from scratch after changing its config:
#     python scripts/build_static_banks.py --tag deepseek `
#         --ingest-config configs/models/deepseek_api.yaml --systems ASEM --force
#
#   Reuse the SAME frozen banks with a different backbone:
#     python scripts/run_static_eval.py --tag deepseek `
#         --config configs/models/qwen3_4b_api.yaml `
#         --systems FastASEM ASEM FullContext
#
#   What got built, and where the banks are:
#     Get-Content static\memory_banks\locomo10\deepseek\manifest.json
#     Get-ChildItem static\memory_banks\locomo10\deepseek -Recurse -Filter *.sqlite
