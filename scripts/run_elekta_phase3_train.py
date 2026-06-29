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
    args = ap.parse_args()

    device = args.device or default_device()
    data = REPO / "runs" / args.scan_id / "ModelTraining/train" / args.scan_id
    ckpt = REPO / "runs" / args.scan_id / "checkpoints" / f"{args.scan_id}_concat_film.pt"
    log = REPO / "runs" / args.scan_id / "logs/phase3_train.log"

    if not data.is_dir():
        raise SystemExit(f"ModelTraining not found: {data}\nRun phase 2 or rsync runs/ from T7.")

    cmd = [
        sys.executable,
        str(LEARN / "ml/trainer.py"),
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
        device,
        "--num_workers",
        str(args.num_workers),
        "--save_path",
        str(ckpt),
    ]

    print("LEARN_GUI_ROOT:", LEARN)
    print("Device:", device)
    print("Command:", " ".join(cmd))
    print("Log:", log)
    log.parent.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        subprocess.run(cmd, cwd=str(LEARN), stdout=fh, stderr=subprocess.STDOUT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
