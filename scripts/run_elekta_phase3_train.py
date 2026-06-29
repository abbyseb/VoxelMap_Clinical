#!/usr/bin/env python3
"""Phase 3: concatenated + FiLM training for CE_P1_V_01."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEARN = Path("/Volumes/T7 Shield/DENNIS_BACKUP/LEARN-GUI/LEARN-GUI-Python")
DATA = REPO / "runs/CE_P1_V_01/ModelTraining/train/CE_P1_V_01"
CKPT = REPO / "runs/CE_P1_V_01/checkpoints/CE_P1_V_01_concat_film.pt"
LOG = REPO / "runs/CE_P1_V_01/logs/phase3_train.log"

cmd = [
    sys.executable.replace("python3", "/opt/anaconda3/bin/python")
    if "python3" in sys.executable
    else "/opt/anaconda3/bin/python",
    str(LEARN / "ml/trainer.py"),
    "--data_dirs", str(DATA),
    "--architecture", "concatenated",
    "--use_film",
    "--epochs", "50",
    "--batch_size", "4",
    "--lr", "1e-5",
    "--val_split", "0.1",
    "--device", "cpu",
    "--num_workers", "0",
    "--save_path", str(CKPT),
]

if __name__ == "__main__":
    print("Command:", " ".join(cmd))
    print("Log:", LOG)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(LEARN), stdout=log, stderr=subprocess.STDOUT, check=True)
