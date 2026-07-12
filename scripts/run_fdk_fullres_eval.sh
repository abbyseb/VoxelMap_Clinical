#!/usr/bin/env bash
# Full-res DVF upsample eval for FDK FiLM and/or no-FiLM checkpoints.
set -euo pipefail

ROOT="${VOXELMAP_CLINICAL_ROOT:-/home/abhishek/Documents/VoxelMap_Clinical}"
LEARN="${LEARN_GUI_ROOT:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python}"
PY="${LEARN}/.venv/bin/python"
GPU="${FDK_GPU:-1}"
MAX_SAMPLES="${FDK_FULLRES_SAMPLES:-90}"
MODE="${FDK_FULLRES_MODE:-both}"   # film | nofilm | both

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

LOG="${FDK_LOG:-$ROOT/runs/fdk_fullres_eval.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== FDK full-res DVF eval $(date) | GPU=$GPU | mode=$MODE | samples=$MAX_SAMPLES ==="

run_one() {
  local fdk="$1"
  local nofilm_flag="$2"
  local out_name metrics dst
  if [[ "$nofilm_flag" == "--no-film" ]]; then
    out_name="eval_fullres_nofilm"
    metrics="$ROOT/runs/${fdk}/${out_name}/fullres_vs_128_metrics.json"
    dst="$ROOT/results/${fdk}/fullres_vs_128_metrics_nofilm.json"
  else
    out_name="eval_fullres"
    metrics="$ROOT/runs/${fdk}/${out_name}/fullres_vs_128_metrics.json"
    dst="$ROOT/results/${fdk}/fullres_vs_128_metrics_film.json"
  fi
  if [[ -f "$metrics" ]]; then
    echo "[$fdk] ${out_name} exists — skip"
    mkdir -p "$(dirname "$dst")"
    cp -f "$metrics" "$dst"
    return 0
  fi
  echo "[$fdk] fullres ${out_name} $(date)"
  "$PY" "$ROOT/scripts/run_fullres_dvf_eval.py" \
    --scan-id "$fdk" --gpu "$GPU" --max-samples "$MAX_SAMPLES" $nofilm_flag
  mkdir -p "$(dirname "$dst")"
  cp -f "$metrics" "$dst"
}

for fdk in "${SCANS[@]}"; do
  echo "--- [$fdk] $(date) ---"
  case "$MODE" in
    film) run_one "$fdk" "" ;;
    nofilm) run_one "$fdk" "--no-film" ;;
    both)
      run_one "$fdk" ""
      run_one "$fdk" "--no-film"
      ;;
    *) echo "Unknown MODE=$MODE"; exit 1 ;;
  esac
done

echo "=== FDK full-res DVF eval finished $(date) ==="
