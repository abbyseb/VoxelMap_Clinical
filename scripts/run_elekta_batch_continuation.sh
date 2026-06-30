#!/usr/bin/env bash
# Wait for in-flight CE_P2 training, then train+eval P2–P5.
set -euo pipefail

ROOT="${VOXELMAP_CLINICAL_ROOT:-/home/abhishek/Documents/VoxelMap_Clinical}"
LEARN="${LEARN_GUI_ROOT:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python}"
PY="${LEARN}/.venv/bin/python"
export VOXELMAP_CLINICAL_ROOT="$ROOT"
export LEARN_GUI_ROOT="$LEARN"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG="$ROOT/runs/batch_train_eval.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== batch continuation $(date) ==="

# Wait for CE_P2 if training still running
P2_FINAL="$ROOT/runs/CE_P2_V_01/checkpoints/CE_P2_V_01_concat_film.pt"
if [[ ! -f "$P2_FINAL" ]]; then
  echo "Waiting for CE_P2_V_01 training to finish..."
  while [[ ! -f "$P2_FINAL" ]]; do
    if pgrep -f "run_elekta_phase3_train.py --scan-id CE_P2_V_01" >/dev/null; then
      sleep 120
    else
      sleep 30
    fi
  done
  echo "CE_P2_V_01 training complete."
fi

bash "$ROOT/scripts/run_elekta_batch_train_eval.sh"
echo "=== batch finished $(date) ==="
