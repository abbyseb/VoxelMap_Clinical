#!/usr/bin/env bash
# Export No-FiLM 5×2 full-res upscale MP4s on the breathing sweep for FDK scans.
# Body-masks DVF panels. Copies into results/<scan>/videos_nofilm/.
set -euo pipefail

ROOT="${VOXELMAP_CLINICAL_ROOT:-/home/abhishek/Documents/VoxelMap_Clinical}"
LEARN="${LEARN_GUI_ROOT:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python}"
PY="${LEARN}/.venv/bin/python"
GPU="${FDK_GPU:-0}"

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

LOG="${FDK_LOG:-$ROOT/runs/fdk_nofilm_sweep_upscale_viz.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== No-FiLM sweep upscale 5x2 MP4s $(date) | GPU=$GPU ==="

for fdk in "${SCANS[@]}"; do
  ckpt="$ROOT/runs/${fdk}/checkpoints_nofilm/best.pt"
  echo "--- [$fdk] $(date) ---"
  if [[ ! -f "$ckpt" ]]; then
    echo "[$fdk] missing nofilm ckpt — skip"
    continue
  fi
  # Skip if a full sweep video already exists (340 Elekta / 680 Varian)
  if ls "$ROOT/results/${fdk}/videos_nofilm/"*_fullres_upscale_*_sweep{340,680}_nofilm.mp4 >/dev/null 2>&1; then
    echo "[$fdk] sweep upscale video exists — skip"
    continue
  fi
  "$PY" "$ROOT/scripts/export_fullres_upscale_mp4.py" \
    --scan-id "$fdk" --gpu "$GPU" --source sweep --plane sagittal \
    --tile-px 220 --fps 8
  echo "[$fdk] DONE $(date)"
done

echo "=== finished $(date) ==="
