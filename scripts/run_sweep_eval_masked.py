#!/usr/bin/env python3
"""Breathing sweep eval with body-masked PSNR/SSIM + masked DVF warp video."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python"))
DEFAULT_SCAN = os.environ.get("SPARE_SCAN_ID", "CE_P1_V_01")


def main() -> int:
    ap = argparse.ArgumentParser(description="Masked sweep eval + export")
    ap.add_argument("--scan-id", default=DEFAULT_SCAN)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument(
        "--no-film",
        action="store_true",
        help="Evaluate no-FiLM checkpoint under checkpoints_nofilm/ → eval_sweep_nofilm/",
    )
    ap.add_argument(
        "--export-only",
        action="store_true",
        help="Skip sweep eval; only run video exports (for already-evaluated runs).",
    )
    args = ap.parse_args()

    scan_id = args.scan_id
    if args.no_film:
        ckpt = REPO / "runs" / scan_id / "checkpoints_nofilm" / "best.pt"
        out = REPO / "runs" / scan_id / "eval_sweep_nofilm"
        log = REPO / "runs" / scan_id / "logs/phase4_sweep_eval_nofilm.log"
    else:
        ckpt = REPO / "runs" / scan_id / "checkpoints" / "best.pt"
        out = REPO / "runs" / scan_id / "eval_sweep"
        log = REPO / "runs" / scan_id / "logs/phase4_sweep_eval.log"
    data = REPO / "runs" / scan_id / "ModelTraining/test" / scan_id

    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    if not (data / "TestProjections").is_dir():
        raise SystemExit(f"Test sweep not found: {data}")

    python = LEARN / ".venv/bin/python"
    if not python.is_file():
        python = Path(sys.executable)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = [
        str(python),
        str(REPO / "ml/sweep_evaluator.py"),
        "--checkpoint",
        str(ckpt),
        "--data_dir",
        str(data),
        "--output_dir",
        str(out),
        "--scan-id",
        scan_id,
        "--device",
        "cuda",
    ]
    if not args.no_film:
        cmd.append("--use_film")

    if not args.export_only:
        out.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, env=env)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
    elif not out.is_dir():
        raise SystemExit(f"Sweep eval output not found: {out}")

    if args.skip_video:
        return 0

    post = [
        [str(python), str(REPO / "scripts/export_test_sweep_mp4.py"), "--scan-id", scan_id],
        [str(python), str(REPO / "scripts/regenerate_sweep_trace.py"), "--scan-id", scan_id],
        [
            str(python),
            str(REPO / "scripts/export_dvf_warp_mp4.py"),
            "--scan-id",
            scan_id,
            "--body-mask",
            "--ptv-mask",
        ],
        [
            str(python),
            str(REPO / "scripts/export_sweep_drr_gt_mp4.py"),
            "--scan-id",
            scan_id,
            "--sim-label",
            "Sim DRR (FDK)" if "_fdk" in scan_id else "Sim DRR",
        ],
    ]
    for post_cmd in post:
        subprocess.run(post_cmd, cwd=str(REPO), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
