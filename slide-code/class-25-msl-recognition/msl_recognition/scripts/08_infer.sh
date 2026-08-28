#!/usr/bin/env bash
# =============================================================================
# 08_infer.sh  –  Run inference on a video or webcam
# =============================================================================
# Usage:
#   bash scripts/08_infer.sh video    data/videos/sample.mp4
#   bash scripts/08_infer.sh webcam   0
#   bash scripts/08_infer.sh batch    data/videos/new_signs/
# =============================================================================

set -euo pipefail

MODE="${1:-video}"         # video | webcam | batch
INPUT="${2:-}"
MODEL="bilstm"             # edit to match your best experiment
EXP="exp_bilstm"        # edit to match your best experiment
#EXP="exp_transformer"
#EXP="exp_stgcn"
CHECKPOINT="results/${EXP}/checkpoints/best.pth"
CONFIG="config/config.yaml"
TOP_K=5

echo "============================================================"
echo " MSL Sign Language Inference"
echo "============================================================"
echo " Mode       : ${MODE}"
echo " Input      : ${INPUT}"
echo " Checkpoint : ${CHECKPOINT}"
echo ""

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CHECKPOINT}"
    echo "  → Train a model first, then update EXP and MODEL in this script."
    exit 1
fi

case "${MODE}" in
    video)
        if [[ -z "${INPUT}" ]]; then
            echo "[ERROR] Provide a video path: bash scripts/08_infer.sh video path/to/video.mp4"
            exit 1
        fi
        python src/infer.py \
            --checkpoint "${CHECKPOINT}" \
            --config     "${CONFIG}" \
            --video      "${INPUT}" \
            --top_k      "${TOP_K}"
        ;;

    webcam)
        CAM="${INPUT:-0}"
        python src/infer.py \
            --checkpoint "${CHECKPOINT}" \
            --config     "${CONFIG}" \
            --webcam     "${CAM}"
        ;;

    batch)
        if [[ -z "${INPUT}" ]]; then
            echo "[ERROR] Provide a directory: bash scripts/08_infer.sh batch path/to/dir/"
            exit 1
        fi
        OUTPUT="results/inference_$(basename ${INPUT}).csv"
        python src/infer.py \
            --checkpoint "${CHECKPOINT}" \
            --config     "${CONFIG}" \
            --batch      "${INPUT}" \
            --output     "${OUTPUT}" \
            --top_k      "${TOP_K}"
        echo "Predictions saved → ${OUTPUT}"
        ;;

    *)
        echo "[ERROR] Unknown mode: ${MODE}. Choose: video | webcam | batch"
        exit 1
        ;;
esac
