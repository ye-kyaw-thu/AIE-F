#!/usr/bin/env bash
# =============================================================================
# 02_extract_keypoints.sh  –  Extract MediaPipe Holistic keypoints from all videos
# =============================================================================
# Processes all .mp4 / .avi / .mov files under data/videos/
# Output .npy files are saved under data/keypoints/ mirroring the folder structure.
#
# For 558 videos on an RTX 3090 Ti this typically takes ~20-40 minutes.
# Re-run safely: already-processed videos are skipped unless --overwrite is set.
# =============================================================================

set -euo pipefail

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
VIDEO_DIR="data/videos"
KEYPOINT_DIR="data/keypoints"
COMPLEXITY=2          # 0=fastest, 2=most accurate (recommended for research)
OVERWRITE=false       # set to true to re-extract existing files
LOG_DIR="results/logs"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${KEYPOINT_DIR}" "${LOG_DIR}"

echo "============================================================"
echo " Step 2 — MediaPipe Holistic Keypoint Extraction"
echo "============================================================"
echo " Video dir     : ${VIDEO_DIR}"
echo " Output dir    : ${KEYPOINT_DIR}"
echo " Complexity    : ${COMPLEXITY}"
echo " Overwrite     : ${OVERWRITE}"
echo ""

if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "[ERROR] Video directory not found: ${VIDEO_DIR}"
    exit 1
fi

N_VIDEOS=$(find "${VIDEO_DIR}" -type f \( -name "*.mp4" -o -name "*.avi" \
           -o -name "*.mov" -o -name "*.mkv" -o -name "*.MP4" \) | wc -l)
echo " Total videos  : ${N_VIDEOS}"
echo ""

# Build args
EXTRA_ARGS=""
if [[ "${OVERWRITE}" == "true" ]]; then
    EXTRA_ARGS="--overwrite"
fi

# Run extraction
python src/extract_keypoints.py \
    --video_dir  "${VIDEO_DIR}" \
    --output_dir "${KEYPOINT_DIR}" \
    --complexity "${COMPLEXITY}" \
    ${EXTRA_ARGS} \
    --log_file   "${LOG_DIR}/extract_keypoints.log" \
    2>&1 | tee "${LOG_DIR}/extract_keypoints_stdout.log"

echo ""
echo "============================================================"
echo " Keypoint extraction complete!"
echo " Stats saved → ${KEYPOINT_DIR}/extraction_stats.json"
echo " Next: run 03_augment_data.sh"
echo "============================================================"
