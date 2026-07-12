#!/usr/bin/env bash
# =============================================================================
# 00_setup_env.sh  –  Create venv and install all dependencies
# =============================================================================
# Usage: bash scripts/00_setup_env.sh
# =============================================================================

set -euo pipefail

ENV_NAME="msl_recog"
VENV_DIR=".venv"
PYTHON_VERSION="3.10"

echo "============================================================"
echo " MSL Recognition — Environment Setup"
echo "============================================================"

# ── Check Python version ─────────────────────────────────────────────────────
echo "[INFO] Checking Python version..."
PYTHON_CMD=""

if command -v python${PYTHON_VERSION} &> /dev/null; then
    PYTHON_CMD="python${PYTHON_VERSION}"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "[ERROR] Python 3.10 not found. Please install Python ${PYTHON_VERSION} first."
    exit 1
fi

ACTUAL_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[INFO] Found Python: ${ACTUAL_VERSION}"

if [[ "$ACTUAL_VERSION" != "${PYTHON_VERSION}" ]]; then
    echo "[WARN] Expected Python ${PYTHON_VERSION}, found ${ACTUAL_VERSION}"
    echo "[WARN] Proceeding anyway, but compatibility issues may occur."
fi

# ── Virtual environment ──────────────────────────────────────────────────────
if [[ -d "${VENV_DIR}" ]]; then
    echo "[INFO] Virtual environment '${VENV_DIR}' already exists."
else
    echo "[INFO] Creating virtual environment '${VENV_DIR}' (Python ${ACTUAL_VERSION})…"
    $PYTHON_CMD -m venv "${VENV_DIR}"
fi

# Activate
source "${VENV_DIR}/bin/activate"

echo "[INFO] Python: $(python --version)"
echo "[INFO] Virtual env: ${VIRTUAL_ENV}"

# ── Upgrade pip ──────────────────────────────────────────────────────────────
echo "[INFO] Upgrading pip…"
pip install --upgrade pip

# ── PyTorch (CUDA 11.8 for RTX 3090 Ti) ─────────────────────────────────────
echo "[INFO] Installing PyTorch with CUDA support…"
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# ── Other dependencies ───────────────────────────────────────────────────────
echo "[INFO] Installing remaining dependencies…"
pip install -r requirements.txt

# ── Verify GPU ───────────────────────────────────────────────────────────────
echo ""
echo "[CHECK] GPU detection:"
python -c "
import torch
print(f'  CUDA available : {torch.cuda.is_available()}')
print(f'  Device count   : {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  GPU name       : {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU memory     : {mem:.1f} GB')
"

echo ""
echo "[CHECK] MediaPipe:"
python -c "import mediapipe; print(f'  mediapipe {mediapipe.__version__} OK')"

echo ""
echo "============================================================"
echo " Setup complete!  Activate with:"
echo "   source ${VENV_DIR}/bin/activate"
echo "============================================================"

