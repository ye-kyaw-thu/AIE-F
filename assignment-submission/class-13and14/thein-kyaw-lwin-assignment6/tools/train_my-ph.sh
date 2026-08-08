#!/bin/bash
set -e

# Resolve the absolute path of the project root
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)

# Make sure the model directory exists
mkdir -p "$PROJECT_ROOT/exp/my-ph/model"

echo "Starting Moses translation model training (G2P: my -> ph)..."
echo "This runs SMT training with a 5-gram phoneme LM and MGIZA (4 CPUs)."
echo "----------------------------------------------------------------------"

# Run train-model.perl
perl "$PROJECT_ROOT/tools/mosesdecoder/scripts/training/train-model.perl" \
  -root-dir "$PROJECT_ROOT/exp/my-ph/model" \
  -corpus "$PROJECT_ROOT/exp/clean-data/clean-train" \
  -f my -e ph \
  -alignment grow-diag-final-and \
  -reordering msd-bidirectional-fe \
  -lm "0:5:$PROJECT_ROOT/exp/my-ph/lm/train.ph.blm:8" \
  -mgiza \
  -mgiza-cpus 4 \
  -external-bin-dir "$PROJECT_ROOT/tools/training-tools" \
  2>&1 | tee "$PROJECT_ROOT/exp/my-ph/model/train.log"

echo "----------------------------------------------------------------------"
echo "Moses training complete!"
echo "Model config file created at: $PROJECT_ROOT/exp/my-ph/model/model/moses.ini"
