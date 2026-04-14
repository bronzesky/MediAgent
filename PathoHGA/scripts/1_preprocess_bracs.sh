#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=/media/share/HDD_16T_1/AIFFPE/MediAgent
PATHO_DIR=$BASE_DIR/worktrees/agent-graph/PathoHGA
DATA_DIR=$BASE_DIR/data/BRACS
TARGET_IMG=$BASE_DIR/data/target.png

MANIFEST=$DATA_DIR/manifest.json
OUTPUT_DIR=$DATA_DIR/graphs

# Use AIFFPE conda environment
PYTHON=/home/rent_user/miniconda3/envs/AIFFPE/bin/python

# Default: smoke test
SUBSET=${1:-smoke}

echo "[1_preprocess_bracs.sh] Starting graph construction"
echo "  Subset: $SUBSET"
echo "  Manifest: $MANIFEST"
echo "  Output: $OUTPUT_DIR"

# Check target image
if [ ! -f "$TARGET_IMG" ]; then
    echo "Warning: Target image not found at $TARGET_IMG"
    echo "Attempting to copy from hact-net..."
    if [ -f "$BASE_DIR/hact-net/data/target.png" ]; then
        cp "$BASE_DIR/hact-net/data/target.png" "$TARGET_IMG"
        echo "Target image copied successfully"
    else
        echo "Error: Cannot find target image for stain normalization"
        exit 1
    fi
fi

# Run graph builder
cd $PATHO_DIR
$PYTHON core/preprocessing/graph_builder.py \
    --manifest "$MANIFEST" \
    --output "$OUTPUT_DIR" \
    --target "$TARGET_IMG" \
    --subset "$SUBSET" \
    --device cuda

echo "[1_preprocess_bracs.sh] Done"
