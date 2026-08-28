#!/usr/bin/env bash
# =============================================================================
# 03_augment_data.sh  –  All-class augmentation for 1-sample-per-class MSL data
# =============================================================================
#
# KEY DESIGN: With 558 videos and ~556 unique classes, a train/val/test split
# produces ZERO class overlap — making evaluation permanently 0%.
#
# Correct fix: augment ALL 558 videos and split BY AUGMENTATION TYPE:
#   Train : aug copies 1..N-1   (all classes, N-1 augmented copies each)
#   Val   : aug copy  0          (all classes, 1 augmented copy each)
#   Test  : original .npy        (all classes, 1 original each = strictest eval)
#
# Result: 100% class overlap across train/val/test.
# =============================================================================

set -euo pipefail

CONFIG="config/config.yaml"
AUG_FACTOR=20
FORCE=false
LOG_DIR="results/logs"
MANIFEST="data/augmented/augmented_manifest.json"

# Parse optional --force flag
for arg in "$@"; do
    [[ "$arg" == "--force" ]] && FORCE=true
done

mkdir -p "data/augmented" "${LOG_DIR}"

echo "============================================================"
echo " Step 3 — All-Class Offline Data Augmentation"
echo "============================================================"
echo " Strategy    : all 558 videos → train/val/test by aug type"
echo " Aug factor  : ${AUG_FACTOR}x"
echo "   Train     : $(( AUG_FACTOR - 1 )) aug copies × 558 = $(( (AUG_FACTOR-1)*558 )) samples"
echo "   Val       : 1 aug copy × 558 = 558 samples"
echo "   Test      : 558 originals (no augmentation)"
echo " Class overlap: 100% across all splits"
echo ""

if [[ -f "${MANIFEST}" && "${FORCE}" != "true" ]]; then
    # Check if it's the new dict format with train/val/test keys
    FORMAT=$(python3 -c "
import json
d = json.load(open('${MANIFEST}'))
print('new' if isinstance(d, dict) and 'train' in d else 'old')
")
    if [[ "${FORMAT}" == "new" ]]; then
        N_TRAIN=$(python3 -c "import json; d=json.load(open('${MANIFEST}')); print(len(d['train']))")
        N_VAL=$(python3 -c "import json; d=json.load(open('${MANIFEST}')); print(len(d['val']))")
        N_TEST=$(python3 -c "import json; d=json.load(open('${MANIFEST}')); print(len(d['test']))")
        echo " [SKIP] All-class manifest exists:"
        echo "   train=${N_TRAIN}, val=${N_VAL}, test=${N_TEST}"
        echo "   Use --force to regenerate."
        exit 0
    else
        echo " [INFO] Old manifest format detected — regenerating with all-class design..."
        rm -f "${MANIFEST}"
    fi
fi

python src/augment.py \
    --config     "${CONFIG}" \
    --aug_factor "${AUG_FACTOR}" \
    2>&1 | tee "${LOG_DIR}/augment.log"

echo ""
echo "============================================================"
echo " Augmentation done!  Manifest → ${MANIFEST}"
echo " Next: bash scripts/04_train.sh"
echo "============================================================"

