#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source BTCenv/bin/activate

# Pick the freest GPU by memory.used then utilization.
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ID="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | sort -t, -k2,2n -k3,3n | head -n1 | awk -F',' '{gsub(/ /, "", $1); print $1}')"
  if [[ ! "${GPU_ID:-}" =~ ^[0-9]+$ ]]; then
    GPU_ID="0"
  fi
else
  GPU_ID="0"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
echo "[AGGRESSIVE] Using freest GPU index: ${GPU_ID:-0}"
RUN_TS="$(date -u +'%Y%m%d_%H%M%S')"
export BTCA_RESULTS_DIR="$ROOT_DIR/results/$RUN_TS"
mkdir -p "$BTCA_RESULTS_DIR"
echo "[AGGRESSIVE] Results dir: $BTCA_RESULTS_DIR"

mkdir -p .cuda_stub/nvvm/libdevice
ln -sf "$ROOT_DIR/BTCenv/lib/python3.10/site-packages/triton/backends/nvidia/lib/libdevice.10.bc" \
  .cuda_stub/nvvm/libdevice/libdevice.10.bc
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$ROOT_DIR/.cuda_stub"

# Aggressive TF knobs (single GPU).
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
export TF_GPU_ALLOCATOR="cuda_malloc_async"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

python main_emergency_2h.py \
  --results-dir "$BTCA_RESULTS_DIR" \
  --target-mode core \
  --aggressive-single-gpu \
  --aggressive-workers 2 \
  --force-retrain \
  --force-fresh-main \
  --per-job-minutes 6 \
  --episode-cap 3 \
  --episode-cap-fallback 4 \
  "$@"
