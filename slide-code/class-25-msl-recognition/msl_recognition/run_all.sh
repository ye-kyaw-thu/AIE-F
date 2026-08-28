#!/usr/bin/env bash
# =============================================================================
# run_all.sh  –  Full pipeline re-run after numeric sort fix
# =============================================================================
# Saves all output to run2.log (append mode so nothing is overwritten).
# Run from the project root:
#   bash run_all.sh
# =============================================================================

set -euo pipefail

LOG="run2.log"

# Start fresh log
echo "============================================================" | tee "${LOG}"
echo " MSL Recognition — Full Pipeline (numeric sort fix applied)" | tee -a "${LOG}"
echo " Started: $(date)" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"

# ── Step 1: Data preparation (rebuilds correct video→annotation mapping) ──────
echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## STEP 1: Data Preparation" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
time bash scripts/01_prepare_data.sh 2>&1 | tee -a "${LOG}"

# ── Step 2: Augmentation (delete old wrong-label data first) ──────────────────
echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## STEP 2: Data Augmentation (deleting old wrong-label data)" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "[INFO] Removing old augmented data..." | tee -a "${LOG}"
rm -rf data/augmented/
time bash scripts/03_augment_data.sh 2>&1 | tee -a "${LOG}"

# ── Step 3: Delete old results ────────────────────────────────────────────────
echo "" | tee -a "${LOG}"
echo "[INFO] Removing old experiment results..." | tee -a "${LOG}"
rm -rf results/exp_bilstm results/exp_transformer results/exp_stgcn
rm -rf results/cv_bilstm results/cv_transformer results/cv_stgcn
echo "[INFO] Old results removed." | tee -a "${LOG}"

# ── Step 4: BiLSTM ────────────────────────────────────────────────────────────
echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## STEP 4A: Train BiLSTM" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
time bash scripts/04_train.sh bilstm exp_bilstm 2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## STEP 4B: Evaluate BiLSTM" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
time bash scripts/05_evaluate.sh bilstm exp_bilstm 2>&1 | tee -a "${LOG}"

# ── Step 5: Transformer ───────────────────────────────────────────────────────
echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## STEP 5A: Train Transformer" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
time bash scripts/04_train.sh transformer exp_transformer 2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## STEP 5B: Evaluate Transformer" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
time bash scripts/05_evaluate.sh transformer exp_transformer 2>&1 | tee -a "${LOG}"

# ── Step 6: ST-GCN ───────────────────────────────────────────────────────────
echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## STEP 6A: Train ST-GCN" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
time bash scripts/04_train.sh stgcn exp_stgcn 2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## STEP 6B: Evaluate ST-GCN" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
time bash scripts/05_evaluate.sh stgcn exp_stgcn 2>&1 | tee -a "${LOG}"

# ── Step 7: Cross-Validation ─────────────────────────────────────────────────
echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## STEP 7A: 5-Fold CV — BiLSTM" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
time bash scripts/07_cross_validate.sh bilstm cv_bilstm 5 2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## STEP 7B: 5-Fold CV — Transformer" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
time bash scripts/07_cross_validate.sh transformer cv_transformer 5 2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## STEP 7C: 5-Fold CV — ST-GCN" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
time bash scripts/07_cross_validate.sh stgcn cv_stgcn 5 2>&1 | tee -a "${LOG}"

# ── Step 8: Comparison plots ─────────────────────────────────────────────────
echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## STEP 8: Generate comparison plots" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
python src/plot_results.py \
    --exp results/exp_bilstm \
          results/exp_transformer \
          results/exp_stgcn \
    --output results/model_comparison.png 2>&1 | tee -a "${LOG}"

# ── Final summary ─────────────────────────────────────────────────────────────
echo "" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
echo "## FINAL RESULTS SUMMARY" | tee -a "${LOG}"
echo "##############################################################" | tee -a "${LOG}"
python3 - 2>&1 | tee -a "${LOG}" << 'PYEOF'
import json
from pathlib import Path

exps = [
    ("BiLSTM",      "exp_bilstm",      "cv_bilstm"),
    ("Transformer", "exp_transformer", "cv_transformer"),
    ("ST-GCN",      "exp_stgcn",       "cv_stgcn"),
]
print(f"\n{'Model':<14} {'Val Top-1':>10} {'Test Top-1':>11} {'Test F1':>9} {'CV Val Mean±Std':>18} {'CV Test Mean±Std':>19}")
print("─" * 85)
for name, exp_dir, cv_dir in exps:
    base   = Path("results") / exp_dir
    cv_b   = Path("results") / cv_dir
    val_m  = base / "evaluation" / "metrics_val.json"
    tst_m  = base / "evaluation" / "metrics_test.json"
    cv_f   = cv_b / "cv_summary.json"
    val_t1 = test_t1 = test_f1 = cv_val = cv_tst = "—"
    if val_m.exists():
        d      = json.load(open(val_m))
        val_t1 = f"{d['top1_accuracy']*100:.2f}%"
    if tst_m.exists():
        d       = json.load(open(tst_m))
        test_t1 = f"{d['top1_accuracy']*100:.2f}%"
        test_f1 = f"{d['f1_macro']*100:.2f}%"
    if cv_f.exists():
        d      = json.load(open(cv_f))
        cv_val = f"{d['mean_val_top1']:.2f}±{d['std_val_top1']:.2f}%"
        cv_tst = f"{d['mean_test_top1']:.2f}±{d['std_test_top1']:.2f}%"
    print(f"{name:<14} {val_t1:>10} {test_t1:>11} {test_f1:>9} {cv_val:>18} {cv_tst:>19}")
print("─" * 85)
PYEOF

echo "" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"
echo " All steps completed!" | tee -a "${LOG}"
echo " Finished: $(date)" | tee -a "${LOG}"
echo " Full log saved → ${LOG}" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"
