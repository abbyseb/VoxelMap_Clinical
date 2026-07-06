#!/usr/bin/env bash
# Pilot: train/eval CE_P1 using FDKRecon/FDK4D_* instead of Evaluation GTVol_*.
# Isolated run id CE_P1_V_01_fdk — does not touch CE_P1_V_01 or existing scripts.
set -euo pipefail

ROOT="${VOXELMAP_CLINICAL_ROOT:-/home/abhishek/Documents/VoxelMap_Clinical}"
LEARN="${LEARN_GUI_ROOT:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python}"
PY="${LEARN}/.venv/bin/python"
SOURCE_SCAN="${SOURCE_SCAN:-CE_P1_V_01}"
FDK_SCAN="${FDK_SCAN:-CE_P1_V_01_fdk}"
GPU="${FDK_GPU:-1}"
EPOCHS="${FDK_EPOCHS:-50}"

export VOXELMAP_CLINICAL_ROOT="$ROOT"
export LEARN_GUI_ROOT="$LEARN"
export CUDA_VISIBLE_DEVICES="$GPU"

PNUM=$(echo "$SOURCE_SCAN" | sed -E 's/CE_P([0-9]+)_.*/\1/')
STAGED="$ROOT/data/staged_fdk/P${PNUM}/${FDK_SCAN}"
RUN="$ROOT/runs/${FDK_SCAN}"
LOG="$RUN/logs/fdk_pilot.log"
mkdir -p "$RUN/logs"
exec > >(tee -a "$LOG") 2>&1

echo "=== Elekta FDK pilot $(date) | source=$SOURCE_SCAN run=$FDK_SCAN GPU=$GPU ==="

echo "[$FDK_SCAN] stage FDK4D as GTVol"
"$PY" "$ROOT/scripts/stage_elekta_fdk_scan.py" \
  --source-scan-id "$SOURCE_SCAN" \
  --staged-scan-id "$FDK_SCAN"

echo "[$FDK_SCAN] phase2 (existing script, separate paths)"
"$PY" "$ROOT/scripts/run_elekta_phase2.py" \
  --scan-id "$FDK_SCAN" \
  --run-root "$RUN" \
  --staged "$STAGED" \
  --with-test

echo "[$FDK_SCAN] train ($EPOCHS epochs)"
"$PY" "$ROOT/scripts/run_elekta_phase3_train.py" \
  --scan-id "$FDK_SCAN" --epochs "$EPOCHS" --gpu "$GPU"

echo "[$FDK_SCAN] train-pair eval"
"$PY" "$ROOT/scripts/run_elekta_phase4_eval.py" --scan-id "$FDK_SCAN" --gpu "$GPU"

echo "[$FDK_SCAN] sweep eval"
"$PY" "$ROOT/scripts/run_elekta_sweep_eval.py" --scan-id "$FDK_SCAN" --gpu "$GPU"

echo "[$FDK_SCAN] export results summary"
DST="$ROOT/results/${FDK_SCAN}"
mkdir -p "$DST/plots"
cp -f "$RUN/plots/loss_curves.png" "$DST/plots/" 2>/dev/null || true
cp -f "$RUN/plots/loss_history.json" "$DST/loss_history.json" 2>/dev/null || true
cp -f "$RUN/eval_sweep/metrics.json" "$DST/sweep_metrics.json" 2>/dev/null || true
cp -f "$RUN/eval/metrics.json" "$DST/train_pair_metrics.json" 2>/dev/null || true
cp -f "$RUN/eval_sweep/"*.png "$DST/plots/" 2>/dev/null || true
cp -f "$RUN/eval/"*.png "$DST/plots/" 2>/dev/null || true

echo "=== Comparison vs GTVol baseline ==="
"$PY" "$ROOT/scripts/compare_fdk_vs_gtvol_results.py" \
  --gtvol-scan "$SOURCE_SCAN" --fdk-scan "$FDK_SCAN"

echo "=== FDK pilot finished $(date) ==="
