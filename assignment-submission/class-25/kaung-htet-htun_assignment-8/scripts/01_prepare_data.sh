#!/usr/bin/env bash
# =============================================================================
# 01_prepare_data.sh  –  Parse annotations, build vocab, create splits
# =============================================================================
# Run ONCE before any training.
# Edit DATA_DIR and ANNOTATION_FILE to match your setup.
# =============================================================================

set -euo pipefail

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
DATA_DIR="data"
VIDEO_DIR="${DATA_DIR}/videos"
ANNOTATION_FILE="${DATA_DIR}/annotations.txt"
CONFIG="config/config.yaml"
LOG_DIR="results/logs"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo " Step 1 — Data Preparation"
echo "============================================================"
echo " Video dir       : ${VIDEO_DIR}"
echo " Annotation file : ${ANNOTATION_FILE}"
echo ""

# Verify annotation file exists
if [[ ! -f "${ANNOTATION_FILE}" ]]; then
    echo "[ERROR] Annotation file not found: ${ANNOTATION_FILE}"
    echo ""
    echo "  Expected format (tab-delimited, one line per sign):"
    echo "  မီးချိတ် ။<TAB>မီးချိတ်"
    echo "  သဲ အိတ် ။<TAB>အိတ် ထဲ သဲ"
    echo ""
    echo "  → Place your annotation file at: ${ANNOTATION_FILE}"
    exit 1
fi

# Verify video directory exists
if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "[WARNING] Video directory not found: ${VIDEO_DIR}"
    echo "  → Download MSL4Emergency videos from GitHub and place them in ${VIDEO_DIR}"
    echo "  → Continuing with annotation-only preparation…"
fi

# Count videos
N_VIDEOS=$(find "${VIDEO_DIR}" -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \
           -o -name "*.MP4" 2>/dev/null | wc -l || echo 0)
N_ANNS=$(grep -c "." "${ANNOTATION_FILE}" || echo 0)

echo " Found ${N_VIDEOS} videos and ${N_ANNS} annotation lines"
echo ""

# Run preparation
python src/prepare_data.py \
    --config "${CONFIG}" \
    --verify_keypoints \
    2>&1 | tee "${LOG_DIR}/prepare_data.log"

echo ""
echo "============================================================"
echo " Data preparation done!"
echo " Outputs:"
echo "   data/label_map.json   ← class vocab"
echo "   data/splits.json      ← train/val/test indices"
echo "   data/kfold_splits.json← 5-fold CV splits"
echo " Next: run 02_extract_keypoints.sh"
echo "============================================================"
