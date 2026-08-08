#!/bin/bash
set -e

# Resolve the absolute path of the project root
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)

# Make sure the tuning work directory exists
mkdir -p "$PROJECT_ROOT/exp/my-ph/mert-work"

echo "Starting Moses MERT tuning (G2P: my -> ph)..."
echo "This will iteratively decode the dev set (dev.my -> dev.ph) to optimize weights."
echo "Running Moses decoder on 4 threads..."
echo "Tuning logs are saved to $PROJECT_ROOT/exp/my-ph/mert-work/mert.log"
echo "----------------------------------------------------------------------"

# Run MERT tuning
perl "$PROJECT_ROOT/tools/mosesdecoder/scripts/training/mert-moses.pl" \
  "$PROJECT_ROOT/exp/clean-data/dev.my" \
  "$PROJECT_ROOT/exp/clean-data/dev.ph" \
  "$PROJECT_ROOT/tools/mosesdecoder/bin/moses" \
  "$PROJECT_ROOT/exp/my-ph/model/model/moses.ini" \
  --mertdir "$PROJECT_ROOT/tools/mosesdecoder/bin/" \
  --working-dir "$PROJECT_ROOT/exp/my-ph/mert-work" \
  --no-filter-phrase-table \
  --decoder-flags="-threads 4" \
  2>&1 | tee "$PROJECT_ROOT/exp/my-ph/mert-work/mert.log"

echo "----------------------------------------------------------------------"
echo "Moses MERT tuning complete!"
echo "Tuned config file created at: $PROJECT_ROOT/exp/my-ph/mert-work/moses.ini"
