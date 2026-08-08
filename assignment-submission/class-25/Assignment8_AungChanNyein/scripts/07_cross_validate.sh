#!/usr/bin/env bash
# =============================================================================
# 07_cross_validate.sh  –  K-Fold Cross-Validation (augmented manifest design)
# =============================================================================
# Folds over the augmented training pool so all 558 classes appear in every
# train and val split within each fold. Final test uses original keypoints.
#
# Usage:
#   bash scripts/07_cross_validate.sh bilstm      cv_bilstm      5
#   bash scripts/07_cross_validate.sh transformer cv_transformer 5
#   bash scripts/07_cross_validate.sh stgcn       cv_stgcn       5
# =============================================================================

set -euo pipefail

MODEL="${1:-bilstm}"
EXP="${2:-cv_${MODEL}}"
FOLDS="${3:-5}"
CONFIG="config/config.yaml"
AUG_MANIFEST="data/augmented/augmented_manifest.json"

echo "============================================================"
echo " K-Fold Cross-Validation"
echo "============================================================"
echo " Model  : ${MODEL}"
echo " Exp    : ${EXP}"
echo " Folds  : ${FOLDS}"
echo ""

# ── Prerequisites ─────────────────────────────────────────────────────────────
if [[ ! -f "data/label_map.json" ]]; then
    echo "[ERROR] data/label_map.json not found. Run 01_prepare_data.sh first."
    exit 1
fi

# Verify all-class manifest exists and is in the correct format
if [[ ! -f "${AUG_MANIFEST}" ]]; then
    echo "[INFO] Augmented manifest not found — running augmentation first..."
    bash scripts/03_augment_data.sh
fi

FORMAT=$(python3 -c "
import json
d = json.load(open('${AUG_MANIFEST}'))
print('new' if isinstance(d, dict) and 'train' in d else 'old')
")
if [[ "${FORMAT}" == "old" ]]; then
    echo "[INFO] Old manifest format — regenerating with all-class design..."
    rm -f "${AUG_MANIFEST}"
    bash scripts/03_augment_data.sh
fi

N_TRAIN=$(python3 -c "import json; d=json.load(open('${AUG_MANIFEST}')); print(len(d['train']))")
N_TEST=$(python3 -c "import json; d=json.load(open('${AUG_MANIFEST}')); print(len(d['test']))")
echo " Aug train pool : ${N_TRAIN} samples (all classes, folded into ${FOLDS} folds)"
echo " Test set       : ${N_TEST} originals (fixed across all folds)"
echo ""

# ── Create experiment directory BEFORE tee tries to write to it ──────────────
mkdir -p "results/${EXP}"

python src/cross_validate.py \
    --config "${CONFIG}" \
    --model  "${MODEL}" \
    --exp    "${EXP}" \
    --folds  "${FOLDS}" \
    2>&1 | tee "results/${EXP}/cv.log"

echo ""
echo "============================================================"
echo " CV done!  Summary → results/${EXP}/cv_summary.json"
echo "============================================================"

# Plot fold distribution
python src/plot_results.py --exp "results/${EXP}"

