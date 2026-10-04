#!/usr/bin/env bash
# G2 falsification: run the two decisive experiments SEQUENTIALLY so only one
# arm's system is ever in memory (the machine has 7.9 GB total; running the
# arms concurrently is what got the first attempt OOM-killed).
#
#   1. default weights, 3 arms  -> does the walk move evidence recall at all?
#   2. ppr_weight=0.6, 2 arms   -> if not, is it the Phase D weight?
#
# Retrieval only: no generation, no API cost.
set -u
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.

echo "########## RUN 1: default weights (ppr_weight=0.15) ##########"
python analysis/_g2_recall_probe.py --arms dark note-only typed+ent
echo "########## RUN 2: ppr_weight=0.6 base_weight=0.30 ##########"
python analysis/_g2_recall_probe.py --arms dark typed+ent \
    --set ppr_weight=0.6 base_weight=0.30
echo "########## DONE ##########"
