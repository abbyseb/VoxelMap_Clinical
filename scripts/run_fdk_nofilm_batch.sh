#!/usr/bin/env bash
# FDK no-FiLM training batch. Reuses existing *_fdk ModelTraining data.
# Writes to checkpoints_nofilm/ / plots_nofilm/ (does not overwrite FiLM runs).
set -euo pipefail

ROOT="${VOXELMAP_CLINICAL_ROOT:-/home/abhishek/Documents/VoxelMap_Clinical}"
LEARN="${LEARN_GUI_ROOT:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python}"
PY="${LEARN}/.venv/bin/python"
GPU="${FDK_GPU:-1}"
EPOCHS="${FDK_EPOCHS:-50}"

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

LOG="${FDK_LOG:-$ROOT/runs/fdk_nofilm_batch.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== FDK no-FiLM batch $(date) | GPU=$GPU | epochs=$EPOCHS ==="
echo "Scans: ${SCANS[*]}"

for fdk in "${SCANS[@]}"; do
  run="$ROOT/runs/${fdk}"
  train_dir="$run/ModelTraining/train/${fdk}"
  final_ckpt="$run/checkpoints_nofilm/${fdk}_concat_nofilm.pt"
  log_dir="$run/logs"
  mkdir -p "$log_dir"

  echo "--- [$fdk] $(date) ---"

  if [[ ! -d "$train_dir/SourceProjections" ]]; then
    echo "[$fdk] ERROR: ModelTraining missing at $train_dir — skip"
    continue
  fi

  if [[ -f "$final_ckpt" ]]; then
    echo "[$fdk] no-FiLM training done, skipping ($final_ckpt)"
    continue
  fi

  echo "[$fdk] train no-FiLM ($EPOCHS epochs) on GPU $GPU"
  "$PY" "$ROOT/scripts/run_elekta_phase3_train.py" \
    --scan-id "$fdk" --epochs "$EPOCHS" --gpu "$GPU" --no-film \
    2>&1 | tee "$log_dir/phase3_train_nofilm_wrapper.log"

  echo "[$fdk] DONE $(date)"
done

echo "=== FDK no-FiLM batch finished $(date) ==="
