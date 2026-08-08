#!/bin/bash
set -e

# Resolve the absolute path of the project root
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)

# Make sure the target directory exists
mkdir -p "$PROJECT_ROOT/exp/ph-my/lm"

echo "Building language model using lmplz (target side: Myanmar graphemes)..."
"$PROJECT_ROOT/tools/mosesdecoder/bin/lmplz" \
  -o 5 \
  < "$PROJECT_ROOT/exp/clean-data/clean-train.my" \
  > "$PROJECT_ROOT/exp/ph-my/lm/train.my.arpa"

echo "Compiling language model to binary format..."
"$PROJECT_ROOT/tools/mosesdecoder/bin/build_binary" \
  "$PROJECT_ROOT/exp/ph-my/lm/train.my.arpa" \
  "$PROJECT_ROOT/exp/ph-my/lm/train.my.blm"

echo "Language model build complete!"
ls -lh "$PROJECT_ROOT/exp/ph-my/lm"
