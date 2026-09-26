#!/usr/bin/env bash
# Phase 1: Build Static Memory Banks (Offline Graph Ingestion)
set -e

TAG="${1:-thg_exp1}"
INGEST_CONFIG="${2:-configs/models/deepseek_api.yaml}"
SYSTEMS="${3:-ASEM-THG}"

echo "================================================================="
echo "  PHASE 1: INGESTION & GRAPH BUILDER"
echo "  Tag: $TAG"
echo "  Config: $INGEST_CONFIG"
echo "  Systems: $SYSTEMS"
echo "================================================================="

python scripts/build_static_banks.py \
    --tag "$TAG" \
    --ingest-config "$INGEST_CONFIG" \
    --systems $SYSTEMS

