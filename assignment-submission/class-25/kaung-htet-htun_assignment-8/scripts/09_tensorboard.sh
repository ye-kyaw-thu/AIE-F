#!/usr/bin/env bash
# =============================================================================
# 09_tensorboard.sh  –  Launch TensorBoard to monitor training
# =============================================================================

set -euo pipefail

LOGDIR="${1:-results}"
PORT="${2:-6006}"

echo "============================================================"
echo " Starting TensorBoard"
echo "============================================================"
echo " Log dir : ${LOGDIR}"
echo " URL     : http://localhost:${PORT}"
echo " Press Ctrl+C to stop"
echo ""

tensorboard --logdir "${LOGDIR}" --port "${PORT}" --bind_all
