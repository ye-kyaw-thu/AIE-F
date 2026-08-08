#!/bin/bash
set -e

# Resolve the absolute path of the project root
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)

echo "Starting Moses decoding on the test set (P2G: ph -> my)..."
echo "Translating test.ph -> test.output using tuned moses.ini..."
echo "Running Moses decoder on 4 threads..."
echo "----------------------------------------------------------------------"

# Run the decoder
"$PROJECT_ROOT/tools/mosesdecoder/bin/moses" \
  -f "$PROJECT_ROOT/exp/ph-my/mert-work/moses.ini" \
  -threads 4 \
  < "$PROJECT_ROOT/exp/clean-data/test.ph" \
  > "$PROJECT_ROOT/exp/ph-my/test.output" \
  2> "$PROJECT_ROOT/exp/ph-my/test.decode.log"

echo "Decoding complete! Output saved to $PROJECT_ROOT/exp/ph-my/test.output"
echo "----------------------------------------------------------------------"
echo "Evaluating translation quality using BLEU..."

# Calculate BLEU score
perl "$PROJECT_ROOT/tools/mosesdecoder/scripts/generic/multi-bleu.perl" \
  "$PROJECT_ROOT/exp/clean-data/test.my" \
  < "$PROJECT_ROOT/exp/ph-my/test.output"

echo "----------------------------------------------------------------------"
