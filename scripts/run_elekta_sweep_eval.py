#!/usr/bin/env python3
"""Run LEARN-GUI tester.py on the synthetic breathing sweep (ModelTraining/test)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/Volumes/T7 Shield/DENNIS_BACKUP/LEARN-GUI/LEARN-GUI-Python"))
DEFAULT_SCAN = os.environ.get("SPARE_SCAN_ID", "CE_P1_V_01")


def main() -> int:
    ap = argparse.ArgumentParser(description="Breathing sweep evaluation via LEARN-GUI tester.py")
    ap.add_argument("--scan-id", default=DEFAULT_SCAN)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    scan_id = args.scan_id
    ckpt = REPO / "runs" / scan_id / "checkpoints" / "best.pt"
    data = REPO / "runs" / scan_id / "ModelTraining/test"
    out = REPO / "runs" / scan_id / "eval_sweep"
    log = REPO / "runs" / scan_id / "logs/phase4_sweep_eval.log"

    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    if not (data / scan_id / "TestProjections").is_dir():
        raise SystemExit(
            f"Test sweep not found: {data / scan_id}\nBuild with prep_train test split (NOTE §6)."
        )

    python = LEARN / ".venv/bin/python"
    if not python.is_file():
        python = Path(sys.executable)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = [
        str(python),
        str(LEARN / "ml/tester.py"),
        "--model_path",
        str(ckpt),
        "--data_dir",
        str(data),
        "--output_dir",
        str(out),
        "--architecture",
        "concatenated",
        "--use_film",
        "--device",
        "cuda",
    ]

    out.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    print("Scan:", scan_id)
    print("Checkpoint:", ckpt)
    print("Test data:", data)
    print("Output:", out)
    print("Log:", log)

    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(LEARN), stdout=fh, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        with log.open(encoding="utf-8") as fh:
            print(fh.read()[-5000:])
        raise SystemExit(proc.returncode)

    with log.open(encoding="utf-8") as fh:
        print(fh.read()[-4000:])

    post = [
        [
            str(python),
            str(REPO / "scripts/export_test_sweep_mp4.py"),
            "--scan-id",
            scan_id,
        ],
        [
            str(python),
            str(REPO / "scripts/regenerate_sweep_trace.py"),
            "--scan-id",
            scan_id,
        ],
        [
            str(python),
            str(REPO / "scripts/export_dvf_warp_mp4.py"),
            "--scan-id",
            scan_id,
        ],
    ]
    for post_cmd in post:
        subprocess.run(post_cmd, cwd=str(REPO), env=env, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
