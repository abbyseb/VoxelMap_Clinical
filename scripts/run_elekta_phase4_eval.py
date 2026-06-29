#!/usr/bin/env python3
"""Phase 4: evaluate best checkpoint (Dice, PSNR, SSIM, centroid shift)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/Volumes/T7 Shield/DENNIS_BACKUP/LEARN-GUI/LEARN-GUI-Python"))
SCAN_ID = os.environ.get("SPARE_SCAN_ID", "CE_P1_V_01")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run VoxelMap evaluation")
    ap.add_argument("--scan-id", default=SCAN_ID)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    data = REPO / "runs" / args.scan_id / "ModelTraining/train" / args.scan_id
    ckpt = args.checkpoint or (REPO / "runs" / args.scan_id / "checkpoints/best.pt")
    out = REPO / "runs" / args.scan_id / "eval"
    log = REPO / "runs" / args.scan_id / "logs/phase4_eval.log"

    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    if not data.is_dir():
        raise SystemExit(f"ModelTraining not found: {data}")

    python = LEARN / ".venv/bin/python"
    if not python.is_file():
        python = Path(sys.executable)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = [
        str(python),
        str(REPO / "ml/evaluator.py"),
        "--checkpoint",
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
    if args.max_samples > 0:
        cmd.extend(["--max_samples", str(args.max_samples)])

    out.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    print("Checkpoint:", ckpt)
    print("Output:", out)
    print("Log:", log)

    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        with log.open(encoding="utf-8") as fh:
            print(fh.read()[-4000:])
        raise SystemExit(proc.returncode)

    with log.open(encoding="utf-8") as fh:
        print(fh.read()[-3000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
