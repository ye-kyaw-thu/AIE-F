#!/usr/bin/env bash
# =============================================================================
# 05_evaluate.sh  –  Evaluate a trained checkpoint on val and test sets
# =============================================================================
# Usage:
#   bash scripts/05_evaluate.sh bilstm exp_bilstm
#   bash scripts/05_evaluate.sh transformer exp_transformer
# =============================================================================

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
MODEL="${1:-bilstm}"
EXP="${2:-exp_${MODEL}}"
CHECKPOINT="results/${EXP}/checkpoints/best.pth"
CONFIG="config/config.yaml"
OUTPUT_DIR="results/${EXP}/evaluation"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo " Evaluation — ${MODEL} / ${EXP}"
echo "============================================================"
echo " Checkpoint : ${CHECKPOINT}"
echo " Output dir : ${OUTPUT_DIR}"
echo ""

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CHECKPOINT}"
    echo "  → Train first with: bash scripts/04_train.sh ${MODEL} ${EXP}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Evaluate on validation set
echo "── Validation set ──────────────────────────────────────────"
python src/evaluate.py \
    --checkpoint "${CHECKPOINT}" \
    --config     "${CONFIG}" \
    --split      val \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/eval_val.log"

echo ""
echo "── Test set ────────────────────────────────────────────────"
python src/evaluate.py \
    --checkpoint "${CHECKPOINT}" \
    --config     "${CONFIG}" \
    --split      test \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/eval_test.log"

echo ""
echo "============================================================"
echo " Evaluation done!  Results → ${OUTPUT_DIR}/"
echo "  ├── metrics_val.json / metrics_test.json"
echo "  ├── per_class_val.csv / per_class_test.csv"
echo "  ├── confusion_matrix_test.png"
echo "  └── predictions_test.csv"
echo "============================================================"
