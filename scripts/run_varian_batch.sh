#!/usr/bin/env bash
# Full Clinical Varian pipeline for CV_P2–CV_P5: stage → phase2 → train → eval → export.
# Runs sequentially on a single GPU (default 1). Skips steps whose outputs already exist.
set -euo pipefail

ROOT="${VOXELMAP_CLINICAL_ROOT:-/home/abhishek/Documents/VoxelMap_Clinical}"
LEARN="${LEARN_GUI_ROOT:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python}"
PY="${LEARN}/.venv/bin/python"
GPU="${VARIAN_GPU:-1}"

export VOXELMAP_CLINICAL_ROOT="$ROOT"
export LEARN_GUI_ROOT="$LEARN"
export CUDA_VISIBLE_DEVICES="$GPU"

SCANS=(${VARIAN_SCANS:-CV_P2_V_01 CV_P3_V_01 CV_P4_V_01 CV_P5_V_01})

BATCH_LOG="$ROOT/runs/varian_batch.log"
mkdir -p "$(dirname "$BATCH_LOG")"
exec > >(tee -a "$BATCH_LOG") 2>&1

echo "=== Varian batch start $(date) | GPU=$GPU | scans=${SCANS[*]} ==="

pnum_of() { echo "$1" | sed -E 's/CV_P([0-9]+)_.*/\1/'; }

for scan in "${SCANS[@]}"; do
  pnum=$(pnum_of "$scan")
  run_root="$ROOT/runs/$scan"
  log_dir="$run_root/logs"
  mkdir -p "$log_dir"
  final_ckpt="$run_root/checkpoints/${scan}_concat_film.pt"
  train_dir="$run_root/ModelTraining/train/$scan"

  echo "--- [$scan] (P$pnum) $(date) ---"

  # 1. Stage
  if [[ ! -d "$ROOT/data/staged/P$pnum/$scan/Proj" ]]; then
    echo "[$scan] staging"
    "$PY" "$ROOT/scripts/stage_varian_scan.py" "$scan"
  else
    echo "[$scan] already staged"
  fi

  # 2. Phase 2 preprocessing (DRR → compress → DVF → prep_train, with test sweep)
  if [[ ! -d "$train_dir/SourceProjections" ]]; then
    echo "[$scan] phase2 preprocessing"
    "$PY" "$ROOT/scripts/run_varian_phase2.py" --scan-id "$scan" --with-test \
      2>&1 | tee "$log_dir/phase2.log"
  else
    echo "[$scan] phase2 outputs present, skipping"
  fi

  # 3. Train (50 epochs)
  if [[ -f "$final_ckpt" ]]; then
    echo "[$scan] training complete, skipping"
  else
    echo "[$scan] training on GPU $GPU"
    "$PY" "$ROOT/scripts/run_elekta_phase3_train.py" \
      --scan-id "$scan" --epochs 50 --gpu "$GPU" \
      2>&1 | tee "$log_dir/phase3_train_wrapper.log"
  fi

  # 4. Train-pair eval
  echo "[$scan] train-pair eval"
  "$PY" "$ROOT/scripts/run_elekta_phase4_eval.py" --scan-id "$scan" --gpu "$GPU" \
    2>&1 | tee "$log_dir/phase4_eval.log"

  # 5. Breathing sweep eval (+ default MP4 post-processing)
  echo "[$scan] sweep eval"
  "$PY" "$ROOT/scripts/run_elekta_sweep_eval.py" --scan-id "$scan" --gpu "$GPU" \
    2>&1 | tee "$log_dir/phase4_sweep_eval.log"

  # 6. Sagittal DVF warp video at peak-PTV slice
  echo "[$scan] sagittal PTV DVF video"
  SLICE=$("$PY" - "$scan" <<'PY'
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
  echo "[$scan] peak PTV sagittal slice = $SLICE"
  "$PY" "$ROOT/scripts/export_dvf_warp_mp4.py" --scan-id "$scan" \
    --plane sagittal --slice-index "$SLICE" --ptv-mask \
    --out "$run_root/videos/${scan}_dvf_warp_sagittal${SLICE}_ptv.mp4"

  # 7. Export results/ summaries
  dst="$ROOT/results/$scan"
  mkdir -p "$dst/plots" "$dst/videos"
  "$PY" - "$scan" "$SLICE" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/home/abhishek/Documents/VoxelMap_Clinical")
from ml.volume_view import VolumeViewConfig
scan, sl = sys.argv[1], int(sys.argv[2])
v = VolumeViewConfig(scan_id=scan, plane="sagittal", slice_index=sl).resolve()
v.save_json(Path(f"/home/abhishek/Documents/VoxelMap_Clinical/results/{scan}/dvf_view_config.json"))
PY
  cp -f "$run_root/plots/loss_curves.png" "$dst/plots/" 2>/dev/null || true
  cp -f "$run_root/plots/loss_history.json" "$dst/loss_history.json" 2>/dev/null || true
  cp -f "$run_root/eval_sweep/metrics.json" "$dst/sweep_metrics.json" 2>/dev/null || true
  cp -f "$run_root/eval_sweep/Performance_Trace"*.png "$dst/plots/" 2>/dev/null || true
  cp -f "$run_root/eval/metrics.json" "$dst/train_pair_metrics.json" 2>/dev/null || true
  cp -f "$run_root/eval/"*.png "$dst/plots/" 2>/dev/null || true
  cp -f "$run_root/videos/"*.mp4 "$dst/videos/" 2>/dev/null || true

  echo "[$scan] DONE $(date)"
done

echo "=== Varian batch finished $(date) ==="
