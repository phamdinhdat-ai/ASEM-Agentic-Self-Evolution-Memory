# Phase 1: Build Static Memory Banks (Offline Graph Ingestion)
# Runs single-pass session ingestion and builds the Temporal Hyper-Graph
# ---------------------------------------------------------------------------

param (
    [string]$Tag = "thg_exp1",
    [string]$IngestConfig = "configs/models/deepseek_api.yaml",
    [string]$Systems = "ASEM-THG",
    [int]$Limit = 0
)

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  PHASE 1: INGESTION & GRAPH BUILDER" -ForegroundColor Cyan
Write-Host "  Tag: $Tag" -ForegroundColor Yellow
Write-Host "  Config: $IngestConfig" -ForegroundColor Yellow
Write-Host "  Systems: $Systems" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan

$cmd = "python scripts/build_static_banks.py --tag $Tag --ingest-config $IngestConfig --systems $Systems"

if ($Limit -gt 0) {
    $cmd += " --limit-conversations $Limit"
}

Write-Host "Executing: $cmd" -ForegroundColor Green
Invoke-Expression $cmd

