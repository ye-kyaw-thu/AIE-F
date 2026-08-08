#!/bin/bash
set -e

# Resolve the absolute path of the project root
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)

echo "Running clean-corpus-n.perl..."
perl "$PROJECT_ROOT/tools/mosesdecoder/scripts/training/clean-corpus-n.perl" \
  "$PROJECT_ROOT/exp/clean-data/train" my ph \
  "$PROJECT_ROOT/exp/clean-data/clean-train" 1 80

echo "Corpus cleaning complete!"
echo "Original counts:"
wc -l "$PROJECT_ROOT/exp/clean-data/train.my" "$PROJECT_ROOT/exp/clean-data/train.ph"

echo "Cleaned counts:"
wc -l "$PROJECT_ROOT/exp/clean-data/clean-train.my" "$PROJECT_ROOT/exp/clean-data/clean-train.ph"
