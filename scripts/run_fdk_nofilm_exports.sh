#!/usr/bin/env bash
# Export No-FiLM FDK plots + DVF warp MP4s into results/<scan>_fdk/{plots,videos}_nofilm/.
# Does not overwrite FiLM artifacts. Checkpoints stay under runs/ (gitignored).
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

LOG="${FDK_LOG:-$ROOT/runs/fdk_nofilm_exports.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== FDK no-FiLM exports $(date) | GPU=$GPU ==="

peak_ptv_slice() {
  local fdk="$1"
  "$PY" - "$fdk" <<'PY'
import sys, numpy as np
sys.path.insert(0, "/home/abhishek/Documents/VoxelMap_Clinical")
from ml.volume_view import VolumeViewConfig
scan = sys.argv[1]
m = np.load(
    f"/home/abhishek/Documents/VoxelMap_Clinical/runs/{scan}/ModelTraining/test/{scan}/Masks/Mask_PTV_mha.npy"
).astype("float32").squeeze()
cfg = VolumeViewConfig(scan_id=scan, plane="sagittal").resolve()
axis = int(cfg.slice_axis)
counts = [(i, int((np.take(m, i, axis=axis) > 0.5).sum())) for i in range(m.shape[axis])]
print(max(counts, key=lambda x: x[1])[0])
PY
}

for fdk in "${SCANS[@]}"; do
  run="$ROOT/runs/${fdk}"
  ckpt="$run/checkpoints_nofilm/best.pt"
  dst="$ROOT/results/${fdk}"
  echo "--- [$fdk] $(date) ---"
  if [[ ! -f "$ckpt" ]]; then
    echo "[$fdk] missing nofilm checkpoint — skip"
    continue
  fi

  mkdir -p "$dst/plots_nofilm" "$dst/videos_nofilm" "$run/videos"

  # Training + sweep plots
  cp -f "$run/plots_nofilm/loss_curves.png" "$dst/plots_nofilm/" 2>/dev/null || true
  cp -f "$run/plots_nofilm/loss_history.json" "$dst/plots_nofilm/" 2>/dev/null || true
  if [[ -f "$run/eval_sweep_nofilm/metrics.json" ]]; then
    "$PY" "$ROOT/scripts/regenerate_sweep_trace.py" \
      --scan-id "$fdk" \
      --metrics "$run/eval_sweep_nofilm/metrics.json" \
      --out "$run/eval_sweep_nofilm/Performance_Trace_by_index.png"
    cp -f "$run/eval_sweep_nofilm/"*.png "$dst/plots_nofilm/" 2>/dev/null || true
    cp -f "$run/eval_sweep_nofilm/metrics.json" "$dst/sweep_metrics_nofilm.json"
  fi

  panels="$run/videos/${fdk}_dvf_warp_panels_nofilm.mp4"
  if [[ ! -f "$panels" ]]; then
    echo "[$fdk] panels MP4"
    "$PY" "$ROOT/scripts/export_dvf_warp_mp4.py" \
      --scan-id "$fdk" --no-film --body-mask --ptv-mask --device cuda
  else
    echo "[$fdk] panels exist — skip"
  fi

  SLICE="$(peak_ptv_slice "$fdk")"
  sagittal="$run/videos/${fdk}_dvf_warp_sagittal${SLICE}_ptv_body_nofilm.mp4"
  if [[ ! -f "$sagittal" ]]; then
    echo "[$fdk] sagittal${SLICE} MP4"
    "$PY" "$ROOT/scripts/export_dvf_warp_mp4.py" \
      --scan-id "$fdk" --no-film --body-mask --ptv-mask \
      --plane sagittal --slice-index "$SLICE" --device cuda
  else
    echo "[$fdk] sagittal exist — skip"
  fi

  cp -f "$run/videos/"*_nofilm.mp4 "$dst/videos_nofilm/" 2>/dev/null || true
  echo "[$fdk] DONE $(date)"
done

echo "=== FDK no-FiLM exports finished $(date) ==="
