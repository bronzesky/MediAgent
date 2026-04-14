#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/share/HDD_16T_1/AIFFPE/MediAgent
WT=${1:-$ROOT/worktrees/agent-main}
cd "$WT/PathoHGA"

BRACS_ROOT=${BRACS_ROOT:-$ROOT/data/BRACS}
OUT_DIR=${OUT_DIR:-$WT/PathoHGA/data/smoke_bracs}
MAX_PER_CLASS=${MAX_PER_CLASS:-3}

python3 -m core.preprocessing.graph_builder \
  --bracs_root "$BRACS_ROOT" \
  --out_dir "$OUT_DIR" \
  --split all \
  --max_per_class "$MAX_PER_CLASS" \
  --feature_dim 64

echo "[OK] smoke preprocess done: $OUT_DIR"
