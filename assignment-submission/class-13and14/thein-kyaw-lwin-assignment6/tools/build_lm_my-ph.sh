#!/bin/bash
set -e

# Resolve the absolute path of the project root
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)

# Make sure the target directory exists
mkdir -p "$PROJECT_ROOT/exp/my-ph/lm"

echo "Building language model using lmplz (target side: phonemes)..."
"$PROJECT_ROOT/tools/mosesdecoder/bin/lmplz" \
  -o 5 \
  < "$PROJECT_ROOT/exp/clean-data/clean-train.ph" \
  > "$PROJECT_ROOT/exp/my-ph/lm/train.ph.arpa"

echo "Compiling language model to binary format..."
"$PROJECT_ROOT/tools/mosesdecoder/bin/build_binary" \
  "$PROJECT_ROOT/exp/my-ph/lm/train.ph.arpa" \
  "$PROJECT_ROOT/exp/my-ph/lm/train.ph.blm"

echo "Language model build complete!"
ls -lh "$PROJECT_ROOT/exp/my-ph/lm"
