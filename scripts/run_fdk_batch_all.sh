#!/usr/bin/env bash
# FDK batch: Elekta + Varian P1–P5 (CE_Pn_V_01 / CV_Pn_V_01 → *_fdk).
# Body-masked PSNR/SSIM in eval; body+PTV masked DVF warp videos.
set -euo pipefail

ROOT="${VOXELMAP_CLINICAL_ROOT:-/home/abhishek/Documents/VoxelMap_Clinical}"
LEARN="${LEARN_GUI_ROOT:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python}"
PY="${LEARN}/.venv/bin/python"
GPU="${FDK_GPU:-1}"
EPOCHS="${FDK_EPOCHS:-50}"

export VOXELMAP_CLINICAL_ROOT="$ROOT"
export LEARN_GUI_ROOT="$LEARN"
export CUDA_VISIBLE_DEVICES="$GPU"

# Scans to process: pass as CLI args to override, else default to all P1–P5.
if [[ $# -gt 0 ]]; then
  SCANS=("$@")
else
  SCANS=(
    CE_P1_V_01 CE_P2_V_01 CE_P3_V_01 CE_P4_V_01 CE_P5_V_01
    CV_P1_V_01 CV_P2_V_01 CV_P3_V_01 CV_P4_V_01 CV_P5_V_01
  )
fi

LOG="${FDK_LOG:-$ROOT/runs/fdk_batch_all.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== FDK batch all patients $(date) | GPU=$GPU | epochs=$EPOCHS ==="

pnum_of() { echo "$1" | sed -E 's/(CE|CV)_P([0-9]+)_.*/\2/'; }
vendor_of() { echo "$1" | sed -E 's/(CE|CV)_.*/\1/'; }

for source in "${SCANS[@]}"; do
  fdk="${source}_fdk"
  pnum=$(pnum_of "$source")
  vendor=$(vendor_of "$source")
  staged="$ROOT/data/staged_fdk/P${pnum}/${fdk}"
  run="$ROOT/runs/${fdk}"
  log_dir="$run/logs"
  final_ckpt="$run/checkpoints/${fdk}_concat_film.pt"
  train_dir="$run/ModelTraining/train/${fdk}"
  mkdir -p "$log_dir"

  echo "--- [$fdk] from $source $(date) ---"

  if [[ ! -d "$staged/Proj" ]]; then
    echo "[$fdk] stage FDK4D"
    "$PY" "$ROOT/scripts/stage_fdk_scan.py" "$source" --staged-scan-id "$fdk"
  fi

  if [[ ! -d "$train_dir/SourceProjections" ]]; then
    echo "[$fdk] phase2"
    if [[ "$vendor" == "CE" ]]; then
      "$PY" "$ROOT/scripts/run_elekta_phase2.py" --scan-id "$fdk" --run-root "$run" \
        --staged "$staged" --with-test 2>&1 | tee "$log_dir/phase2.log"
    else
      "$PY" "$ROOT/scripts/run_varian_phase2.py" --scan-id "$fdk" --run-root "$run" \
        --staged "$staged" --with-test 2>&1 | tee "$log_dir/phase2.log"
    fi
  fi

  if [[ ! -f "$final_ckpt" ]]; then
    echo "[$fdk] train ($EPOCHS epochs)"
    "$PY" "$ROOT/scripts/run_elekta_phase3_train.py" --scan-id "$fdk" --epochs "$EPOCHS" --gpu "$GPU" \
      2>&1 | tee "$log_dir/phase3_train_wrapper.log"
  else
    echo "[$fdk] training done, skipping"
  fi

  echo "[$fdk] train-pair eval (body-masked metrics)"
  "$PY" "$ROOT/scripts/run_elekta_phase4_eval.py" --scan-id "$fdk" --gpu "$GPU" \
    2>&1 | tee "$log_dir/phase4_eval.log"

  echo "[$fdk] sweep eval (body-masked) + videos"
  "$PY" "$ROOT/scripts/run_sweep_eval_masked.py" --scan-id "$fdk" --gpu "$GPU" \
    2>&1 | tee "$log_dir/phase4_sweep_eval.log"

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

echo "=== FDK batch all finished $(date) ==="
