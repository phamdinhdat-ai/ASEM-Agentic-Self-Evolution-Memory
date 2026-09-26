# Phase 2: Static Memory Evaluation with Full Metrics
# Runs retrieval over frozen Phase 1 graphs and evaluates using SLM + Independent Judge
# ---------------------------------------------------------------------------

param (
    [string]$Tag = "thg_exp1",
    [string]$EvalConfig = "configs/models/qwen3_4b_api.yaml",
    [string]$JudgeConfig = "configs/models/judge_api.yaml",
    [string]$Systems = "ASEM-THG FullContext NoMemory",
    [int]$Limit = 0
)

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  PHASE 2: RETRIEVAL & QA EVALUATION (FULL METRICS)" -ForegroundColor Cyan
Write-Host "  Tag: $Tag" -ForegroundColor Yellow
Write-Host "  Backbone Model Config: $EvalConfig" -ForegroundColor Yellow
Write-Host "  Judge Model Config: $JudgeConfig" -ForegroundColor Yellow
Write-Host "  Systems: $Systems" -ForegroundColor Yellow
Write-Host "  Metrics: EM, F1, ROUGE-L, BERTScore-F1, LLM-as-a-Judge" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan

$cmd = "python scripts/run_static_eval.py --tag $Tag --config $EvalConfig --judge-config $JudgeConfig --systems $Systems --metrics em f1 rougeL bertscore_f1 judge --per-category --per-conversation"

if ($Limit -gt 0) {
    $cmd += " --limit $Limit"
}

Write-Host "Executing: $cmd" -ForegroundColor Green
Invoke-Expression $cmd

