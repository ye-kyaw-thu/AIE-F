#!/bin/bash
set -e

# Resolve the absolute path of the project root (one level up from this script's directory)
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)

echo "Creating experiment directories under $PROJECT_ROOT/exp..."
mkdir -p "$PROJECT_ROOT/exp/clean-data"
mkdir -p "$PROJECT_ROOT/exp/my-ph/lm"
mkdir -p "$PROJECT_ROOT/exp/my-ph/model"
mkdir -p "$PROJECT_ROOT/exp/my-ph/mert-work"
mkdir -p "$PROJECT_ROOT/exp/ph-my/lm"
mkdir -p "$PROJECT_ROOT/exp/ph-my/model"
mkdir -p "$PROJECT_ROOT/exp/ph-my/mert-work"
mkdir -p "$PROJECT_ROOT/exp/scripts"

echo "Copying data files to $PROJECT_ROOT/exp/clean-data..."
cp "$PROJECT_ROOT/data/g2p-par/"* "$PROJECT_ROOT/exp/clean-data/"

echo "Copying SGM scripts to $PROJECT_ROOT/exp/scripts..."
cp "$PROJECT_ROOT/tools/scripts/"* "$PROJECT_ROOT/exp/scripts/"
chmod +x "$PROJECT_ROOT/exp/scripts/"*

echo "Experiment directory setup complete!"
ls -lh "$PROJECT_ROOT/exp/clean-data"
