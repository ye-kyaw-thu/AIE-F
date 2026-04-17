#!/bin/bash

set -e

echo "=== BUILD: Mental Health SCRDR ==="
python3 ../scrdr_interactive.py \
  --input ./data/train.csv \
  --target Depression \
  --exclude Person_ID \
  --tree depression_rules.json \
  --mode build | tee build.log

echo
echo "=== TEST: Mental Health SCRDR ==="
python3 ../scrdr_interactive.py \
  --input ./data/test.csv \
  --target Depression \
  --exclude Person_ID \
  --tree depression_rules.json \
  --mode test | tee test.log