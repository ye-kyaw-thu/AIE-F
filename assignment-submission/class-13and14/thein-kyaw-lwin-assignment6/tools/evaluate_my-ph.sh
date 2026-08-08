#!/bin/bash
set -e

# Resolve the absolute path of the project root
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)

echo "Starting Moses decoding on the test set (G2P: my -> ph)..."
echo "Translating test.my -> test.output using tuned moses.ini..."
echo "Running Moses decoder on 4 threads..."
echo "----------------------------------------------------------------------"

# Run the decoder
"$PROJECT_ROOT/tools/mosesdecoder/bin/moses" \
  -f "$PROJECT_ROOT/exp/my-ph/mert-work/moses.ini" \
  -threads 4 \
  < "$PROJECT_ROOT/exp/clean-data/test.my" \
  > "$PROJECT_ROOT/exp/my-ph/test.output" \
  2> "$PROJECT_ROOT/exp/my-ph/test.decode.log"

echo "Decoding complete! Output saved to $PROJECT_ROOT/exp/my-ph/test.output"
echo "----------------------------------------------------------------------"
echo "Evaluating translation quality using BLEU..."

# Calculate BLEU score
perl "$PROJECT_ROOT/tools/mosesdecoder/scripts/generic/multi-bleu.perl" \
  "$PROJECT_ROOT/exp/clean-data/test.ph" \
  < "$PROJECT_ROOT/exp/my-ph/test.output"

echo "----------------------------------------------------------------------"
