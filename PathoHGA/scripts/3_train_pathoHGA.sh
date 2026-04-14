#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/share/HDD_16T_1/AIFFPE/MediAgent
WT=${1:-$ROOT/worktrees/agent-main}
cd "$WT/PathoHGA"

GRAPH_ROOT=${GRAPH_ROOT:-$WT/PathoHGA/data/smoke_bracs}
OUT_DIR=${OUT_DIR:-$WT/PathoHGA/results/smoke/pathohga}
python3 -m core.train \
  --graph_root "$GRAPH_ROOT" \
  --epochs 1 \
  --use_c1 \
  --use_c2 \
  --lambda_align 0.1 \
  --max_train_steps 40 \
  --max_test_steps 20 \
  --out_dir "$OUT_DIR"
