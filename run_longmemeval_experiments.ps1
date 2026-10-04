# LongMemEval benchmark — run every ASEM system + baseline on the LongMemEval
# oracle and s_cleaned datasets.
#
# Thin wrapper around scripts/run_longmemeval_eval.py (no architecture changes):
# for each question it ingests that question's haystack sessions into a FRESH
# bank, asks the system, and scores EM / F1 / ROUGE-L / BERTScore-F1 / LLM judge
# plus retrieval recall (session + turn) and abstention accuracy.
#
# Both runs RESUME: re-run this file and each system answers only the questions
# it has not answered yet (per-system JSONL under
# data/benchmarks/results/longmemeval/<tag>/preds/).
#
# NOTE: the Python script loads .env itself (OPENAI_API_KEY / OPENAI_BASE_URL);
#       the model name comes from --config, not from .env.
#
# Datasets:
#   oracle     datasets/longmemeval/longmemeval_oracle.json     (only the gold
#                                                               sessions)
#   s_cleaned  datasets/longmemeval/longmemeval_s_cleaned.json  (full haystack,
#                                                               small/clean)

Set-Location $PSScriptRoot
$env:PYTHONPATH = "."
$env:KMP_DUPLICATE_LIB_OK = "TRUE"    # Windows + faiss/torch

$SYSTEMS = "ASEM-THG ASEM ASEMv2 FastASEM NoMemory FullContext SimRetrieval AtomicLinking RLManagerOnly ValueRetrievalOnly"

# ---- Run 1 — LongMemEval oracle ------------------------------------------
python scripts/run_longmemeval_eval.py `
    --input datasets/longmemeval/longmemeval_oracle.json `
    --tag longmemeval_oracle `
    --config configs/models/deepseek_api.yaml `
    --judge-config configs/models/judge_api.yaml `
    --systems $SYSTEMS `
    --metrics em f1 rougeL bertscore_f1 judge `
    --per-category `
    --recall

# ---- Run 2 — LongMemEval s_cleaned ---------------------------------------
python scripts/run_longmemeval_eval.py `
    --input datasets/longmemeval/longmemeval_s_cleaned.json `
    --tag longmemeval_s_cleaned `
    --config configs/models/deepseek_api.yaml `
    --judge-config configs/models/judge_api.yaml `
    --systems $SYSTEMS `
    --metrics em f1 rougeL bertscore_f1 judge `
    --per-category `
    --recall

# Results -> data\benchmarks\results\longmemeval\<tag>\<tag>__<model>.json / .md
#            data\benchmarks\results\longmemeval\<tag>\preds\    (resumable)
#            data\benchmarks\results\longmemeval\<tag>\scores\   (judge rows)
#
# Variants — copy/paste one at a time:
#
#   Cheap smoke test (5 questions, three systems — do this FIRST):
#     python scripts/run_longmemeval_eval.py --limit 5 `
#         --tag smoke --systems NoMemory FullContext SimRetrieval `
#         --metrics em f1 rougeL --config configs/models/deepseek_api.yaml
#
#   Retrieval-only (no LLM judge — much faster, backbone-agnostic):
#     python scripts/run_longmemeval_eval.py `
#         --input datasets/longmemeval/longmemeval_s_cleaned.json `
#         --tag longmemeval_s_cleaned --config configs/models/qwen3_4b_api.yaml `
#         --systems ASEM FastASEM FullContext `
#         --metrics em f1 rougeL --recall
#
#   Start over for one tag (discard saved predictions):
#     python scripts/run_longmemeval_eval.py `
#         --input datasets/longmemeval/longmemeval_oracle.json `
#         --tag longmemeval_oracle --clean `
#         --config configs/models/deepseek_api.yaml `
#         --systems $SYSTEMS `
#         --metrics em f1 rougeL bertscore_f1 judge --per-category --recall
