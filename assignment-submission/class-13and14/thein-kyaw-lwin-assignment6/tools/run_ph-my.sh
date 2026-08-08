#!/bin/bash
set -e

# Resolve the absolute path of the project root
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)

# Verify that moses binary exists in the tools folder before running
if [ ! -x "$PROJECT_ROOT/tools/mosesdecoder/bin/moses" ]; then
  echo "Error: Moses decoder binary not found or not executable at $PROJECT_ROOT/tools/mosesdecoder/bin/moses."
  echo "Please make sure you are running this script inside the configured Docker container."
  exit 1
fi

echo "======================================================================"
echo "          Myanmar SMT P2G (ph -> my) Pipeline Automation Script       "
echo "======================================================================"
echo "Starting P2G pipeline from scratch under project root: $PROJECT_ROOT..."
echo ""

echo "[Step 1/7] Running setup_exp.sh..."
"$PROJECT_ROOT/tools/setup_exp.sh"
echo ""

echo "[Step 2/7] Running clean_data.sh..."
"$PROJECT_ROOT/tools/clean_data.sh"
echo ""

echo "[Step 3/7] Generating SGM files..."
cd "$PROJECT_ROOT/exp/scripts"
perl ./generate_sgms.pl
cd "$PROJECT_ROOT"
echo ""

echo "[Step 4/7] Building P2G Language Model (ph-my)..."
"$PROJECT_ROOT/tools/build_lm_ph-my.sh"
echo ""

echo "[Step 5/7] Training P2G SMT Model (ph-my)..."
"$PROJECT_ROOT/tools/train_ph-my.sh"
echo ""

echo "[Step 6/7] Tuning P2G SMT Model (ph-my)..."
"$PROJECT_ROOT/tools/tune_ph-my.sh"
echo ""

echo "[Step 7/7] Evaluating P2G SMT Model (ph-my)..."
"$PROJECT_ROOT/tools/evaluate_ph-my.sh"
echo ""

echo "======================================================================"
echo "                   P2G Pipeline Completed Successfully                "
echo "======================================================================"
echo "Final P2G (ph -> my) BLEU Score:"
if [ -f "$PROJECT_ROOT/exp/ph-my/test.output" ]; then
  perl "$PROJECT_ROOT/tools/mosesdecoder/scripts/generic/multi-bleu.perl" \
    "$PROJECT_ROOT/exp/clean-data/test.my" \
    < "$PROJECT_ROOT/exp/ph-my/test.output"
else
  echo "P2G evaluation output not found."
fi
echo "======================================================================"
