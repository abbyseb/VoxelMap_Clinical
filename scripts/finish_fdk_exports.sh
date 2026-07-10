#!/usr/bin/env bash
# Finish FDK runs that already have train + sweep metrics but failed on video export.
set -euo pipefail

ROOT="${VOXELMAP_CLINICAL_ROOT:-/home/abhishek/Documents/VoxelMap_Clinical}"
LEARN="${LEARN_GUI_ROOT:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python}"
PY="${LEARN}/.venv/bin/python"
GPU="${FDK_GPU:-0}"

export VOXELMAP_CLINICAL_ROOT="$ROOT"
export LEARN_GUI_ROOT="$LEARN"
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ $# -lt 1 ]]; then
  echo "Usage: FDK_GPU=0 $0 CV_P3_V_01_fdk [CV_P4_V_01_fdk ...]"
  exit 1
fi

for fdk in "$@"; do
  run="$ROOT/runs/${fdk}"
  sweep="$run/eval_sweep/metrics.json"
  ckpt="$run/checkpoints/${fdk}_concat_film.pt"
  if [[ ! -f "$ckpt" || ! -f "$sweep" ]]; then
    echo "[$fdk] skip: need checkpoint + sweep metrics"
    continue
  fi

  echo "=== [$fdk] export-only $(date) ==="
  "$PY" "$ROOT/scripts/run_sweep_eval_masked.py" --scan-id "$fdk" --export-only --gpu "$GPU"

  echo "[$fdk] sagittal PTV + body-masked DVF video"
  SLICE=$("$PY" - "$fdk" <<'PY'
import sys, numpy as np
sys.path.insert(0, "/home/abhishek/Documents/VoxelMap_Clinical")
from ml.volume_view import VolumeViewConfig
scan = sys.argv[1]
m = np.load(f"/home/abhishek/Documents/VoxelMap_Clinical/runs/{scan}/ModelTraining/test/{scan}/Masks/Mask_PTV_mha.npy").astype("float32").squeeze()
cfg = VolumeViewConfig(scan_id=scan, plane="sagittal").resolve()
axis = int(cfg.slice_axis)
counts = [(i, int((np.take(m, i, axis=axis) > 0.5).sum())) for i in range(m.shape[axis])]
print(max(counts, key=lambda x: x[1])[0])
PY
)
  "$PY" "$ROOT/scripts/export_dvf_warp_mp4.py" --scan-id "$fdk" \
    --plane sagittal --slice-index "$SLICE" --ptv-mask --body-mask \
    --out "$run/videos/${fdk}_dvf_warp_sagittal${SLICE}_ptv_body.mp4"

  dst="$ROOT/results/${fdk}"
  mkdir -p "$dst/plots" "$dst/videos"
  cp -f "$run/plots/loss_curves.png" "$dst/plots/" 2>/dev/null || true
  cp -f "$run/plots/loss_history.json" "$dst/loss_history.json" 2>/dev/null || true
  cp -f "$run/eval_sweep/metrics.json" "$dst/sweep_metrics.json" 2>/dev/null || true
  cp -f "$run/eval/metrics.json" "$dst/train_pair_metrics.json" 2>/dev/null || true
  cp -f "$run/eval_sweep/"*.png "$dst/plots/" 2>/dev/null || true
  cp -f "$run/eval/"*.png "$dst/plots/" 2>/dev/null || true
  cp -f "$run/videos/"*.mp4 "$dst/videos/" 2>/dev/null || true

  echo "[$fdk] DONE $(date)"
done
