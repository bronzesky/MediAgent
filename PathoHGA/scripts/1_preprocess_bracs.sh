#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=/media/share/HDD_16T_1/AIFFPE/MediAgent
PATHO_DIR=$BASE_DIR/worktrees/agent-graph/PathoHGA
DATA_DIR=$BASE_DIR/data/BRACS
MANIFEST=$DATA_DIR/manifest.json
OUTPUT_DIR=$DATA_DIR/graphs
PYTHON=/home/rent_user/miniconda3/envs/AIFFPE/bin/python
SUBSET=${1:-smoke}

echo "[1_preprocess_bracs.sh] Starting graph construction"
echo "  Subset: $SUBSET"
echo "  Manifest: $MANIFEST"
echo "  Output: $OUTPUT_DIR"

cd "$PATHO_DIR"
$PYTHON scripts/create_manifest.py
$PYTHON core/preprocessing/graph_builder.py \
  --manifest "$MANIFEST" \
  --output "$OUTPUT_DIR" \
  --subset "$SUBSET" \
  --feature_dim 64

echo "[1_preprocess_bracs.sh] Done"
