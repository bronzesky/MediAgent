#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/share/HDD_16T_1/AIFFPE/MediAgent
WT=${1:-$ROOT/worktrees/agent-main}
cd "$WT/PathoHGA"

GRAPH_ROOT=${GRAPH_ROOT:-$WT/PathoHGA/data/smoke_bracs}
CKPT=${CKPT:-$WT/PathoHGA/results/smoke/pathohga/model.pt}
INDEX=${INDEX:-$WT/PathoHGA/results/smoke/pathohga/graph_index.json}
OUT=${OUT:-$WT/PathoHGA/results/smoke/reasoning_report.json}

python3 -m core.agent.reasoning \
  --graph_root "$GRAPH_ROOT" \
  --checkpoint "$CKPT" \
  --index_json "$INDEX" \
  --split test \
  --case_idx 0 \
  --topk 3 \
  --use_c1 \
  --use_c2 \
  --out_json "$OUT"
