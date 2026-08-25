#!/usr/bin/env bash
# =============================================================================
# 06_run_all_experiments.sh  –  Full reproducible experiment suite
# =============================================================================
# Experiments:
#   A: BiLSTM + Attention        (all-class augmented data)
#   B: Transformer Encoder       (all-class augmented data)
#   C: ST-GCN                    (all-class augmented data)
#   D: 5-Fold CV on best model   (for paper reporting)
# =============================================================================

set -euo pipefail

CONFIG="config/config.yaml"

run_experiment() {
    local model="$1"
    local exp="$2"
    local extra="${3:-}"
    echo ""
    echo "████████████████████████████████████████████████████████"
    echo "  EXPERIMENT: ${exp}  (${model})"
    echo "████████████████████████████████████████████████████████"
    START=$(date +%s)
    bash scripts/04_train.sh   "${model}" "${exp}" "${extra}"
    bash scripts/05_evaluate.sh "${model}" "${exp}"
    END=$(date +%s)
    echo "  ✓ ${exp} done in $(( (END-START)/60 )) min"
}

# Prerequisites check
echo "============================================================"
echo " MSL Recognition — Full Experiment Suite"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo ""

if [[ ! -f "data/label_map.json" ]]; then
    echo "[ERROR] Run 01_prepare_data.sh and 02_extract_keypoints.sh first."
    exit 1
fi

# Ensure all-class augmented manifest exists and is correct format
AUG_MANIFEST="data/augmented/augmented_manifest.json"
if [[ ! -f "${AUG_MANIFEST}" ]]; then
    bash scripts/03_augment_data.sh
else
    FORMAT=$(python3 -c "
import json; d=json.load(open('${AUG_MANIFEST}'))
print('new' if isinstance(d,dict) and 'train' in d else 'old')")
    if [[ "${FORMAT}" == "old" ]]; then
        echo "[INFO] Old manifest — regenerating..."
        rm -f "${AUG_MANIFEST}"
        bash scripts/03_augment_data.sh
    fi
fi

# Experiments A, B, C
run_experiment bilstm       exp_A_bilstm
run_experiment transformer  exp_B_transformer
run_experiment stgcn        exp_C_stgcn

# Experiment D: 5-Fold CV on transformer (best candidate)
echo ""
echo "████████████████████████████████████████████████████████"
echo "  EXPERIMENT D: 5-Fold Cross-Validation (transformer)"
echo "████████████████████████████████████████████████████████"
python src/cross_validate.py \
    --config "${CONFIG}" \
    --model  transformer \
    --exp    exp_D_cv_transformer \
    --folds  5 \
    2>&1 | tee results/logs/cv_transformer.log

# Plot comparison
echo ""
echo "Generating comparison plots..."
python src/plot_results.py \
    --exp results/exp_A_bilstm \
          results/exp_B_transformer \
          results/exp_C_stgcn \
    --output results/model_comparison.png

# Final summary table
echo ""
echo "============================================================"
echo " FINAL RESULTS SUMMARY"
echo "============================================================"
python3 - << 'PYEOF'
import json
from pathlib import Path

exps = [
    ("A", "exp_A_bilstm",       "BiLSTM + Attention"),
    ("B", "exp_B_transformer",  "Transformer Encoder"),
    ("C", "exp_C_stgcn",        "ST-GCN"),
    ("D", "exp_D_cv_transformer","Transformer 5-Fold CV"),
]
print(f"{'Exp':<4} {'Model':<28} {'Val Top-1':>10} {'Test Top-1':>11} {'Test F1':>9}")
print("─" * 66)
for exp_id, exp_dir, name in exps:
    base  = Path("results") / exp_dir
    val_m = base / "evaluation" / "metrics_val.json"
    tst_m = base / "evaluation" / "metrics_test.json"
    cv_m  = base / "cv_summary.json"
    val_top1 = test_top1 = test_f1 = "—"
    if val_m.exists():
        d = json.load(open(val_m))
        val_top1 = f"{d['top1_accuracy']*100:.2f}%"
    if tst_m.exists():
        d = json.load(open(tst_m))
        test_top1 = f"{d['top1_accuracy']*100:.2f}%"
        test_f1   = f"{d['f1_macro']*100:.2f}%"
    if cv_m.exists():
        d = json.load(open(cv_m))
        val_top1  = f"{d['mean_top1']:.2f}±{d['std_top1']:.2f}%"
    print(f" {exp_id:<3} {name:<28} {val_top1:>10} {test_top1:>11} {test_f1:>9}")
print("─" * 66)
PYEOF

echo ""
echo " Comparison chart → results/model_comparison.png"
echo " TensorBoard      → tensorboard --logdir results/"
echo "============================================================"

