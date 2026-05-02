#!/bin/bash

set -e

echo "=== BUILD: Banking SCRDR ==="
python3 ../scrdr_interactive.py \
  --input ./data/train.csv \
  --target deposit \
  --tree bank_rules.json \
  --mode build | tee build.log

echo
echo "=== TEST: Banking SCRDR ==="
python3 ../scrdr_interactive.py \
  --input ./data/test.csv \
  --target deposit \
  --tree bank_rules.json \
  --mode test | tee test.log