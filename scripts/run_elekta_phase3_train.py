#!/usr/bin/env python3
"""Phase 3: concatenated + FiLM training for CE_P1_V_01."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/Volumes/T7 Shield/DENNIS_BACKUP/LEARN-GUI/LEARN-GUI-Python"))
SCAN_ID = os.environ.get("SPARE_SCAN_ID", "CE_P1_V_01")


def default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run VoxelMap concatenated+FiLM training")
    ap.add_argument("--scan-id", default=SCAN_ID)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index (sets CUDA_VISIBLE_DEVICES)")
    args = ap.parse_args()

    device = args.device or default_device()
    if device.startswith("cuda") and device == "cuda":
        device = "cuda:0"

    data = REPO / "runs" / args.scan_id / "ModelTraining/train" / args.scan_id
    ckpt_dir = REPO / "runs" / args.scan_id / "checkpoints"
    plots_dir = REPO / "runs" / args.scan_id / "plots"
    final_ckpt = ckpt_dir / f"{args.scan_id}_concat_film.pt"
    log = REPO / "runs" / args.scan_id / "logs/phase3_train.log"

    if not data.is_dir():
        raise SystemExit(f"ModelTraining not found: {data}\nRun phase 2 or rsync runs/ from T7.")

    python = LEARN / ".venv/bin/python"
    if not python.is_file():
        python = Path(sys.executable)

    env = os.environ.copy()
    if device.startswith("cuda"):
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = [
        str(python),
        str(REPO / "ml/trainer.py"),
        "--data_dirs",
        str(data),
        "--architecture",
        "concatenated",
        "--use_film",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        "1e-5",
        "--val_split",
        "0.1",
        "--device",
        "cuda" if device.startswith("cuda") else device,
        "--num_workers",
        str(args.num_workers),
        "--checkpoint_dir",
        str(ckpt_dir),
        "--plots_dir",
        str(plots_dir),
        "--save_path",
        str(final_ckpt),
    ]

    print("VOXELMAP_CLINICAL_ROOT:", REPO)
    print("LEARN_GUI_ROOT:", LEARN)
    print("CUDA_VISIBLE_DEVICES:", env.get("CUDA_VISIBLE_DEVICES", "(unset)"))
    print("Device:", device)
    print("Checkpoints:", ckpt_dir)
    print("Plots:", plots_dir)
    print("Command:", " ".join(cmd))
    print("Log:", log)

    log.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
