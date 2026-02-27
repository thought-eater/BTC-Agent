#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source BTCenv/bin/activate

RESUME_FROM=""
RESTART_MAIN_FROM_B=0
RESTART_PREDICTIVE=0
RESTART_TRADE=0
RESTART_B2E=0
MAIN_RESUME_MODE="off"
FORCE_PREP_FLAG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume-from)
      RESUME_FROM="${2:-}"
      shift 2
      ;;
    --restart-main-from-b)
      RESTART_MAIN_FROM_B=1
      shift
      ;;
    --restart-predictive)
      RESTART_PREDICTIVE=1
      shift
      ;;
    --restart-trade)
      RESTART_TRADE=1
      shift
      ;;
    --restart-b2e)
      RESTART_B2E=1
      shift
      ;;
    --force-prep-rebuild)
      FORCE_PREP_FLAG="--force-prep-rebuild"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

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
echo "[10H] Using freest GPU index: ${GPU_ID:-0}"

if [[ -n "$RESUME_FROM" ]]; then
  if [[ "$RESUME_FROM" = /* ]]; then
    export BTCA_RESULTS_DIR="$RESUME_FROM"
  else
    export BTCA_RESULTS_DIR="$ROOT_DIR/$RESUME_FROM"
  fi
  MAIN_RESUME_MODE="auto"
else
  RUN_TS="$(date -u +'%Y%m%d_%H%M%S')"
  export BTCA_RESULTS_DIR="$ROOT_DIR/results/$RUN_TS"
fi

mkdir -p "$BTCA_RESULTS_DIR"
echo "[10H] Results dir: $BTCA_RESULTS_DIR"

if [[ "$RESTART_B2E" -eq 1 ]]; then
  RESTART_TRADE=1
  RESTART_PREDICTIVE=1
  RESTART_MAIN_FROM_B=1
fi

if [[ "$RESTART_MAIN_FROM_B" -eq 1 ]]; then
  if [[ "$MAIN_RESUME_MODE" != "auto" ]]; then
    echo "[10H][ERROR] --restart-main-from-b requires --resume-from <results_dir>."
    exit 1
  fi
  python - "$BTCA_RESULTS_DIR" <<'PY'
import json
import pathlib
import sys

results_dir = pathlib.Path(sys.argv[1])
manifest_path = results_dir / "run_manifest.json"
state_path = results_dir / "run_state.json"

if not manifest_path.exists() or not state_path.exists():
    raise SystemExit("run_manifest.json or run_state.json not found in resume directory.")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
state = json.loads(state_path.read_text(encoding="utf-8"))

nodes = manifest.get("nodes", {})
jobs = state.get("jobs", {})
removed_nodes = [k for k in list(nodes.keys()) if k.startswith("main_")]
removed_jobs = [k for k in list(jobs.keys()) if k.startswith("main_")]

for k in removed_nodes:
    nodes.pop(k, None)
for k in removed_jobs:
    jobs.pop(k, None)

# Also clear main checkpoints to avoid carrying collapsed policies.
ckpt_dir = pathlib.Path("utils/checkpoints")
removed_ckpts = 0
for p in ckpt_dir.glob("main_dqn_*"):
    try:
        p.unlink()
        removed_ckpts += 1
    except Exception:
        pass

manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
print(f"[10H] Cleared Main phase state: nodes={len(removed_nodes)} jobs={len(removed_jobs)} checkpoints={removed_ckpts}")
PY
fi

if [[ "$RESTART_PREDICTIVE" -eq 1 ]]; then
  if [[ "$MAIN_RESUME_MODE" != "auto" ]]; then
    echo "[10H][ERROR] --restart-predictive requires --resume-from <results_dir>."
    exit 1
  fi
  python - "$BTCA_RESULTS_DIR" <<'PY'
import json
import pathlib
import sys

results_dir = pathlib.Path(sys.argv[1])
manifest_path = results_dir / "run_manifest.json"
state_path = results_dir / "run_state.json"

if not manifest_path.exists() or not state_path.exists():
    raise SystemExit("run_manifest.json or run_state.json not found in resume directory.")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
state = json.loads(state_path.read_text(encoding="utf-8"))

nodes = manifest.get("nodes", {})
jobs = state.get("jobs", {})
removed_nodes = [k for k in list(nodes.keys()) if k.startswith("pred_")]
removed_jobs = [k for k in list(jobs.keys()) if k.startswith("pred_")]

for k in removed_nodes:
    nodes.pop(k, None)
for k in removed_jobs:
    jobs.pop(k, None)

ckpt_dir = pathlib.Path("utils/checkpoints")
removed_ckpts = 0
for p in ckpt_dir.glob("predictive_dqn*"):
    try:
        p.unlink()
        removed_ckpts += 1
    except Exception:
        pass

manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
print(f"[10H] Cleared Predictive phase state: nodes={len(removed_nodes)} jobs={len(removed_jobs)} checkpoints={removed_ckpts}")
PY
fi

if [[ "$RESTART_TRADE" -eq 1 ]]; then
  if [[ "$MAIN_RESUME_MODE" != "auto" ]]; then
    echo "[10H][ERROR] --restart-trade requires --resume-from <results_dir>."
    exit 1
  fi
  python - "$BTCA_RESULTS_DIR" <<'PY'
import json
import pathlib
import sys

results_dir = pathlib.Path(sys.argv[1])
manifest_path = results_dir / "run_manifest.json"
state_path = results_dir / "run_state.json"

if not manifest_path.exists() or not state_path.exists():
    raise SystemExit("run_manifest.json or run_state.json not found in resume directory.")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
state = json.loads(state_path.read_text(encoding="utf-8"))

nodes = manifest.get("nodes", {})
jobs = state.get("jobs", {})
removed_nodes = [k for k in list(nodes.keys()) if k == "trade"]
removed_jobs = [k for k in list(jobs.keys()) if k == "trade"]

for k in removed_nodes:
    nodes.pop(k, None)
for k in removed_jobs:
    jobs.pop(k, None)

ckpt_dir = pathlib.Path("utils/checkpoints")
removed_ckpts = 0
for p in ckpt_dir.glob("trade_dqn*"):
    try:
        p.unlink()
        removed_ckpts += 1
    except Exception:
        pass

manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
print(f"[10H] Cleared Trade phase state: nodes={len(removed_nodes)} jobs={len(removed_jobs)} checkpoints={removed_ckpts}")
PY
fi

# TensorFlow/XLA CUDA libdevice path fix.
mkdir -p .cuda_stub/nvvm/libdevice
ln -sf "$ROOT_DIR/BTCenv/lib/python3.10/site-packages/triton/backends/nvidia/lib/libdevice.10.bc" \
  .cuda_stub/nvvm/libdevice/libdevice.10.bc
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$ROOT_DIR/.cuda_stub"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
PREDICTIVE_ACTION_STEP="${PREDICTIVE_ACTION_STEP:-0.2}"
TRADE_MAX_STEPS="${TRADE_MAX_STEPS:-4096}"
PREDICTIVE_MAX_STEPS="${PREDICTIVE_MAX_STEPS:-4096}"
echo "[10H] Predictive action step: ${PREDICTIVE_ACTION_STEP}"
echo "[10H] Trade max steps: ${TRADE_MAX_STEPS}"
echo "[10H] Predictive max steps: ${PREDICTIVE_MAX_STEPS}"

python main.py \
  --results-dir "$BTCA_RESULTS_DIR" \
  --gpu-mode single \
  --parallel-policy safe_adaptive \
  --max-workers 2 \
  --total-budget-hours 10 \
  --deadline-hours 10 \
  --stage-budget "prep=30,trade=90,pred=90,main=360,eval=30" \
  --resume "$MAIN_RESUME_MODE" \
  --main-variant-list "paper,policy_gradient" \
  --main-omega-list "16" \
  --predictive-variant-list "paper" \
  --predictive-action-step "$PREDICTIVE_ACTION_STEP" \
  --min-improvements 1 \
  --trade-episode-cap 220 \
  --trade-max-steps "$TRADE_MAX_STEPS" \
  --predictive-episode-cap 220 \
  --predictive-max-steps "$PREDICTIVE_MAX_STEPS" \
  --main-episode-cap 140 \
  --early-stop-metric roi \
  --early-stop-patience 20 \
  --eval-every-n-episodes 1 \
  --min-episodes-before-early-stop 6 \
  --min-trades-for-best 20 \
  $FORCE_PREP_FLAG \
  "${EXTRA_ARGS[@]}"
