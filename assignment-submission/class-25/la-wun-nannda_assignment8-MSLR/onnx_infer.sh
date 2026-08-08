#!/usr/bin/env bash
# =============================================================================
# onnx_infer.sh  –  Run ONNX inference on video, webcam, or a batch of videos
# =============================================================================
# Prerequisites:
#   python src/export_to_onnx.py --all --output_dir onnx_models
#
# Usage (run from repo root; console output only):
#   bash onnx_infer.sh video      transformer data/sample4infer/idx20-101.mp4
#   bash onnx_infer.sh webcam     transformer 0
#   bash onnx_infer.sh batch      bilstm      data/sample4infer/
#   bash onnx_infer.sh batch-all  transformer data/sample4infer/
#
# Models: bilstm | transformer | stgcn  (default: transformer)
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f "${ROOT_DIR}/src/infer_onnx.py" ]]; then
    echo "[ERROR] src/infer_onnx.py not found under ${ROOT_DIR}"
    echo "  → Run this script from the repo root (where onnx_infer.sh lives)."
    exit 1
fi

MODE="${1:-}"
MODEL="${2:-transformer}"
INPUT="${3:-}"

ONNX_DIR="onnx_models"
TOP_K=5
BUF_FRAMES=100

declare -A ONNX_NAMES=(
    [bilstm]="bilstm_msl.onnx"
    [transformer]="transformer_msl.onnx"
    [stgcn]="stgcn_msl.onnx"
)

usage() {
    sed -n '5,14p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit 1
}

validate_model() {
    local model="$1"
    if [[ -z "${ONNX_NAMES[${model}]+x}" ]]; then
        echo "[ERROR] Unknown model: ${model}. Choose: bilstm | transformer | stgcn"
        exit 1
    fi
}

check_onnx_prereqs() {
    local model="$1"
    local onnx_path="${ONNX_DIR}/${ONNX_NAMES[${model}]}"

    if [[ ! -f "${onnx_path}" ]]; then
        echo "[ERROR] ONNX model not found: ${onnx_path}"
        echo "  → Run: python src/export_to_onnx.py --model ${model} --output_dir ${ONNX_DIR}"
        exit 1
    fi
    if [[ ! -f "${ONNX_DIR}/label_map.json" ]]; then
        echo "[ERROR] label_map.json not found in ${ONNX_DIR}/"
        echo "  → Re-export models or copy data/label_map.json to ${ONNX_DIR}/"
        exit 1
    fi
}

run_video() {
    local model="$1"
    local video="$2"

    echo "------------------------------------------------------------"
    echo " Video  : ${video}"
    echo " Model  : ${model}"
    echo " Output : console only (no log or video files)"
    echo "------------------------------------------------------------"

    python src/infer_onnx.py \
        --onnx_model "${ONNX_DIR}/${ONNX_NAMES[${model}]}" \
        --video      "${video}" \
        --top_k      "${TOP_K}" \
        --output_dir ""
}

run_webcam() {
    local model="$1"
    local cam="${2:-0}"

    echo "------------------------------------------------------------"
    echo " Webcam : device ${cam}"
    echo " Model  : ${model}"
    echo " Output : live window only (no log or video files)"
    echo " Keys   : Q = quit | C = clear buffer"
    echo "------------------------------------------------------------"

    python src/infer_onnx.py \
        --onnx_model  "${ONNX_DIR}/${ONNX_NAMES[${model}]}" \
        --webcam      "${cam}" \
        --no_mirror \
        --buf_frames  "${BUF_FRAMES}" \
        --zero_z \
        --top_k       "${TOP_K}" \
        --output_dir  ""
}

run_batch() {
    local model="$1"
    local dir="$2"

    if [[ ! -d "${dir}" ]]; then
        echo "[ERROR] Directory not found: ${dir}"
        exit 1
    fi

    mapfile -t videos < <(find "${dir}" -maxdepth 1 -type f -name '*.mp4' | sort)
    if [[ ${#videos[@]} -eq 0 ]]; then
        echo "[ERROR] No .mp4 files in ${dir}"
        exit 1
    fi

    echo "============================================================"
    echo " ONNX batch inference"
    echo " Model  : ${model}"
    echo " Videos : ${#videos[@]} in ${dir}"
    echo "============================================================"

    for video in "${videos[@]}"; do
        run_video "${model}" "${video}"
    done

    echo ""
    echo "Done."
}

run_batch_all() {
    local dir="$1"

    for model in bilstm transformer stgcn; do
        check_onnx_prereqs "${model}"
        run_batch "${model}" "${dir}"
    done
}

# ── Parse args ────────────────────────────────────────────────────────────────

if [[ -z "${MODE}" ]]; then
    usage
fi

case "${MODE}" in
    -h|--help|help)
        usage
        ;;
esac

case "${MODE}" in
    video)
        if [[ -z "${INPUT}" ]]; then
            echo "[ERROR] Provide a video path."
            echo "  bash onnx_infer.sh video transformer data/sample4infer/idx20-101.mp4"
            exit 1
        fi
        validate_model "${MODEL}"
        check_onnx_prereqs "${MODEL}"
        run_video "${MODEL}" "${INPUT}"
        ;;

    webcam)
        CAM="${INPUT:-0}"
        validate_model "${MODEL}"
        check_onnx_prereqs "${MODEL}"
        run_webcam "${MODEL}" "${CAM}"
        ;;

    batch)
        if [[ -z "${INPUT}" ]]; then
            echo "[ERROR] Provide a directory of .mp4 files."
            echo "  bash onnx_infer.sh batch bilstm data/sample4infer/"
            exit 1
        fi
        validate_model "${MODEL}"
        check_onnx_prereqs "${MODEL}"
        run_batch "${MODEL}" "${INPUT}"
        ;;

    batch-all)
        if [[ -z "${INPUT}" ]]; then
            echo "[ERROR] Provide a directory of .mp4 files."
            echo "  bash onnx_infer.sh batch-all transformer data/sample4infer/"
            exit 1
        fi
        run_batch_all "${INPUT}"
        ;;

    *)
        echo "[ERROR] Unknown mode: ${MODE}. Choose: video | webcam | batch | batch-all"
        usage
        ;;
esac
