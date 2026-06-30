#!/usr/bin/env bash
# Train + evaluate CE_P2–CE_P5 sequentially (skip patients already finished).
set -euo pipefail

ROOT="${VOXELMAP_CLINICAL_ROOT:-/home/abhishek/Documents/VoxelMap_Clinical}"
LEARN="${LEARN_GUI_ROOT:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python}"
PY="${LEARN}/.venv/bin/python"
export VOXELMAP_CLINICAL_ROOT="$ROOT"
export LEARN_GUI_ROOT="$LEARN"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCANS=(CE_P2_V_01 CE_P3_V_01 CE_P4_V_01 CE_P5_V_01)

for scan in "${SCANS[@]}"; do
  log_dir="$ROOT/runs/$scan/logs"
  mkdir -p "$log_dir"
  final_ckpt="$ROOT/runs/$scan/checkpoints/${scan}_concat_film.pt"

  if [[ -f "$final_ckpt" ]]; then
    echo "[$scan] training complete ($final_ckpt), skipping training"
  elif pgrep -f "run_elekta_phase3_train.py --scan-id $scan" >/dev/null; then
    echo "[$scan] training already running, waiting..."
    while pgrep -f "run_elekta_phase3_train.py --scan-id $scan" >/dev/null; do
      sleep 60
    done
    if [[ ! -f "$final_ckpt" ]]; then
      echo "[$scan] training process ended without final checkpoint" >&2
      exit 1
    fi
  else
    echo "[$scan] training..."
    "$PY" "$ROOT/scripts/run_elekta_phase3_train.py" \
      --scan-id "$scan" --epochs 50 --gpu 0 \
      2>&1 | tee "$log_dir/phase3_train.log"
  fi

  echo "[$scan] train-pair eval..."
  "$PY" "$ROOT/scripts/run_elekta_phase4_eval.py" --scan-id "$scan" --gpu 0 \
    2>&1 | tee "$log_dir/phase4_eval.log"

  echo "[$scan] sweep eval..."
  "$PY" "$ROOT/scripts/run_elekta_sweep_eval.py" --scan-id "$scan" --gpu 0 \
    2>&1 | tee "$log_dir/phase4_sweep_eval.log"

  echo "[$scan] export results..."
  "$PY" - <<PY
import shutil
from pathlib import Path
root = Path("$ROOT")
scan = "$scan"
dst = root / "results" / scan
dst.mkdir(parents=True, exist_ok=True)
run_root = root / "runs" / scan
copies = [
    (run_root / "plots/loss_curves.png", dst / "plots/loss_curves.png"),
    (run_root / "plots/loss_history.json", dst / "loss_history.json"),
    (run_root / "eval_sweep/metrics.json", dst / "sweep_metrics.json"),
    (run_root / "eval_sweep/Performance_Trace.png", dst / "plots/Performance_Trace.png"),
    (run_root / "eval_sweep/Performance_Trace_by_index.png", dst / "plots/Performance_Trace_by_index.png"),
]
for src, out in copies:
    if src.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        print(f"  {out}")
PY
  echo "[$scan] done"
done
echo "Batch complete."
