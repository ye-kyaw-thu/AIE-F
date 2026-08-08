#!/bin/bash
set -e

# Resolve script directory and change working directory to it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Redirect all stdout and stderr to pipeline.log inside the script directory
exec > >(tee -ia "$SCRIPT_DIR/pipeline.log") 2>&1

echo "=================================================="
echo "🚀 Starting assignment5 End-to-End Language Model Pipeline"
echo "=================================================="

# 1. Compile KenLM and install Python bindings
echo ""
echo "=== Step 0: Compiling KenLM and installing Python bindings ==="
if [ -d "./kenlm_src" ]; then
    echo "Using existing kenlm_src directory..."
else
    if [ -d "../LM-Tutorial/kenlm_src" ]; then
        echo "Copying KenLM source from ../LM-Tutorial/kenlm_src..."
        cp -r ../LM-Tutorial/kenlm_src ./kenlm_src
    else
        echo "Cloning KenLM source from GitHub..."
        git clone https://github.com/kpu/kenlm.git kenlm_src
    fi
fi

echo "Building KenLM tool binaries..."
cd kenlm_src
rm -rf build && mkdir -p build && cd build
cmake ..
make -j$(nproc 2>/dev/null || echo 4)
echo "KenLM tools successfully built!"

echo "Installing Python bindings..."
cd ..
pip install . --break-system-packages 2>/dev/null || pip install .
cd ..
echo "Python bindings successfully installed!"

# Verify imports
python3 -c "import kenlm; print('KenLM binding import successful from:', kenlm.__file__)"

# 2. Run Downloader
echo ""
python3 src/download.py

# 3. Run Segmenter
echo ""
python3 src/tokenize_data.py

# 4. Run Trainer
echo ""
python3 src/train.py

# 5. Run Evaluator and Adapter
echo ""
python3 src/evaluate.py

echo "=================================================="
echo "🎉 Pipeline finished successfully! README.md generated."
echo "=================================================="
