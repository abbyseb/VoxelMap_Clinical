#!/usr/bin/env bash
# No-FiLM FDK sweep eval only (metrics, no videos). Writes eval_sweep_nofilm/.
set -euo pipefail

ROOT="${VOXELMAP_CLINICAL_ROOT:-/home/abhishek/Documents/VoxelMap_Clinical}"
LEARN="${LEARN_GUI_ROOT:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python}"
PY="${LEARN}/.venv/bin/python"
GPU="${FDK_GPU:-1}"

export VOXELMAP_CLINICAL_ROOT="$ROOT"
export LEARN_GUI_ROOT="$LEARN"
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ $# -gt 0 ]]; then
  SCANS=("$@")
else
  SCANS=(
    CE_P1_V_01_fdk CE_P2_V_01_fdk CE_P3_V_01_fdk CE_P4_V_01_fdk CE_P5_V_01_fdk
    CV_P1_V_01_fdk CV_P2_V_01_fdk CV_P3_V_01_fdk CV_P4_V_01_fdk CV_P5_V_01_fdk
  )
fi

LOG="${FDK_LOG:-$ROOT/runs/fdk_nofilm_eval.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== FDK no-FiLM sweep eval $(date) | GPU=$GPU ==="

for fdk in "${SCANS[@]}"; do
  metrics="$ROOT/runs/${fdk}/eval_sweep_nofilm/metrics.json"
  ckpt="$ROOT/runs/${fdk}/checkpoints_nofilm/best.pt"
  echo "--- [$fdk] $(date) ---"
  if [[ ! -f "$ckpt" ]]; then
    echo "[$fdk] missing nofilm checkpoint — skip"
    continue
  fi
  if [[ -f "$metrics" ]]; then
    echo "[$fdk] sweep metrics exist — skip"
    continue
  fi
  "$PY" "$ROOT/scripts/run_sweep_eval_masked.py" \
    --scan-id "$fdk" --gpu "$GPU" --no-film --skip-video
  # also copy into results for publishing
  dst="$ROOT/results/${fdk}"
  mkdir -p "$dst"
  cp -f "$metrics" "$dst/sweep_metrics_nofilm.json"
  echo "[$fdk] DONE $(date)"
done

echo "=== FDK no-FiLM sweep eval finished $(date) ==="
