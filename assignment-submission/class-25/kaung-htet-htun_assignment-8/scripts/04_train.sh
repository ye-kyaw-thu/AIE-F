#!/usr/bin/env bash
# =============================================================================
# 04_train.sh  –  Train one model
# =============================================================================
# Usage:
#   bash scripts/04_train.sh                       # BiLSTM (default)
#   bash scripts/04_train.sh transformer exp_transformer
#   bash scripts/04_train.sh stgcn       exp_stgcn
# =============================================================================

set -euo pipefail

MODEL="${1:-bilstm}"
EXP="${2:-exp_${MODEL}}"
EXTRA_ARGS="${3:-}"
CONFIG="config/config.yaml"
LOG_DIR="results/${EXP}/logs"
AUG_MANIFEST="data/augmented/augmented_manifest.json"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo " Training MSL Recognition Model"
echo "============================================================"
echo " Model      : ${MODEL}"
echo " Experiment : ${EXP}"
echo " Config     : ${CONFIG}"
echo ""

# Verify prerequisites
if [[ ! -f "data/label_map.json" ]]; then
    echo "[ERROR] data/label_map.json not found. Run 01_prepare_data.sh first."
    exit 1
fi

# Ensure all-class augmented manifest exists
if [[ ! -f "${AUG_MANIFEST}" ]]; then
    echo "[INFO] Augmented manifest not found — running augmentation first..."
    bash scripts/03_augment_data.sh
fi

# Verify it's the new dict format
FORMAT=$(python3 -c "
import json, sys
d = json.load(open('${AUG_MANIFEST}'))
print('new' if isinstance(d, dict) and 'train' in d else 'old')
")
if [[ "${FORMAT}" == "old" ]]; then
    echo "[INFO] Old manifest format — regenerating with all-class design..."
    rm -f "${AUG_MANIFEST}"
    bash scripts/03_augment_data.sh
fi

N_TRAIN=$(python3 -c "import json; d=json.load(open('${AUG_MANIFEST}')); print(len(d['train']))")
N_VAL=$(python3 -c "import json; d=json.load(open('${AUG_MANIFEST}')); print(len(d['val']))")
N_TEST=$(python3 -c "import json; d=json.load(open('${AUG_MANIFEST}')); print(len(d['test']))")
echo " Data: train=${N_TRAIN}, val=${N_VAL}, test=${N_TEST} (all-class design)"
echo ""

echo "[GPU]"
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu \
    --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
echo ""

START=$(date +%s)

python src/train.py \
    --config  "${CONFIG}" \
    --model   "${MODEL}" \
    --exp     "${EXP}" \
    --weighted_loss \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_DIR}/train_stdout.log"

END=$(date +%s)
ELAPSED=$(( END - START ))
echo ""
echo "  Total training time: $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"
echo ""
echo "============================================================"
echo " Training done!  → results/${EXP}/"
echo " Next: bash scripts/05_evaluate.sh ${MODEL} ${EXP}"
echo "============================================================"

