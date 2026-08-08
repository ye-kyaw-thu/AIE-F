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
echo "          Myanmar SMT G2P (my -> ph) Pipeline Automation Script       "
echo "======================================================================"
echo "Starting G2P pipeline from scratch under project root: $PROJECT_ROOT..."
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

echo "[Step 4/7] Building G2P Language Model (my-ph)..."
"$PROJECT_ROOT/tools/build_lm_my-ph.sh"
echo ""

echo "[Step 5/7] Training G2P SMT Model (my-ph)..."
"$PROJECT_ROOT/tools/train_my-ph.sh"
echo ""

echo "[Step 6/7] Tuning G2P SMT Model (my-ph)..."
"$PROJECT_ROOT/tools/tune_my-ph.sh"
echo ""

echo "[Step 7/7] Evaluating G2P SMT Model (my-ph)..."
"$PROJECT_ROOT/tools/evaluate_my-ph.sh"
echo ""

echo "======================================================================"
echo "                   G2P Pipeline Completed Successfully                "
echo "======================================================================"
echo "Final G2P (my -> ph) BLEU Score:"
if [ -f "$PROJECT_ROOT/exp/my-ph/test.output" ]; then
  perl "$PROJECT_ROOT/tools/mosesdecoder/scripts/generic/multi-bleu.perl" \
    "$PROJECT_ROOT/exp/clean-data/test.ph" \
    < "$PROJECT_ROOT/exp/my-ph/test.output"
else
  echo "G2P evaluation output not found."
fi
echo "======================================================================"
