#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source BTCenv/bin/activate

export CUDA_VISIBLE_DEVICES=0
RUN_TS="$(date -u +'%Y%m%d_%H%M%S')"
export BTCA_RESULTS_DIR="$ROOT_DIR/results/$RUN_TS"
mkdir -p "$BTCA_RESULTS_DIR"
echo "[EMERGENCY] Results dir: $BTCA_RESULTS_DIR"

mkdir -p .cuda_stub/nvvm/libdevice
ln -sf "$ROOT_DIR/BTCenv/lib/python3.10/site-packages/triton/backends/nvidia/lib/libdevice.10.bc" \
  .cuda_stub/nvvm/libdevice/libdevice.10.bc
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$ROOT_DIR/.cuda_stub"

python main_emergency_2h.py --results-dir "$BTCA_RESULTS_DIR" "$@"
