#!/usr/bin/env python3
"""Regenerate sweep trace plots vs projection index (fixes angle-sorted flat segments)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
SCAN_ID = os.environ.get("SPARE_SCAN_ID", "CE_P1_V_01")


def load_resp_bins(test_dir: Path) -> np.ndarray:
    import pandas as pd

    p = test_dir / "RespBin.csv"
    if not p.is_file():
        return np.array([])
    return np.atleast_1d(pd.read_csv(p, header=None).values.squeeze()).astype(int)


def generate_sweep_trace(res: dict, save_path: Path, resp_bins: np.ndarray | None = None) -> None:
    """
    Plot metrics vs projection index (sweep order).

    Sorting by gantry angle creates horizontal plateaus: only 10 respiratory phases
    exist, so angle-sorted GT curves look like straight segments.
    """
    n = len(res["angles"])
    x = np.arange(1, n + 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        "Breathing Sweep — metrics vs projection index\n"
        "(use index order, not sorted gantry angle, to avoid flat GT segments)",
        fontsize=12,
        fontweight="bold",
    )

    labels = ["LR displacement (mm)", "SI displacement (mm)", "AP displacement (mm)"]
    keys = ["lr", "si", "ap"]

    for i, ax in enumerate(axes[:3]):
        gt = np.array(res["gt_shifts_mm"][keys[i]])
        pred = np.array(res["shifts_mm"][keys[i]])
        ax.plot(x, gt, color="black", lw=1.5, label="Ground-truth (GT DVF)", marker=".", ms=2)
        ax.plot(x, pred, "--", color="red", lw=1.0, label="Prediction", alpha=0.85, marker=".", ms=2)
        ax.set_ylabel(labels[i])
        if resp_bins is not None and len(resp_bins) == n:
            p6 = resp_bins == 6
            ax.scatter(x[p6], gt[p6], c="#2ca02c", s=12, zorder=5, label="phase 06 (ref)" if i == 0 else None)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    psnr = np.array(res.get("psnr", []), dtype=float)
    if len(psnr) == n:
        axes[3].plot(x, psnr, color="#e6550d", lw=1.2, label="PSNR (dB)")
        axes[3].set_ylabel("PSNR (dB)")
        axes[3].legend(loc="lower right", fontsize=8)

    ssim = np.array(res.get("ssim", []), dtype=float)
    if len(ssim) == n and np.any(np.isfinite(ssim)):
        axes[4].plot(x, ssim, color="#2c7fb8", lw=1.2, label="SSIM")
        axes[4].set_ylabel("SSIM")
        axes[4].set_ylim(0.0, 1.05)
        axes[4].legend(loc="lower right", fontsize=8)

    detj = np.array(res.get("det_j_neg_fraction", []), dtype=float)
    if len(detj) == n and np.any(np.isfinite(detj)):
        axes[5].plot(x, detj, color="#7b3294", lw=1.2, label="neg det(J) frac.")
        axes[5].set_ylabel("Neg. det(J)")
        axes[5].legend(loc="upper right", fontsize=8)

    axes[5].set_xlabel("Projection index (sweep order, 1–340)")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-id", default=SCAN_ID)
    ap.add_argument("--metrics", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    metrics_path = args.metrics or (REPO / "runs" / args.scan_id / "eval_sweep/metrics.json")
    out_path = args.out or (REPO / "runs" / args.scan_id / "eval_sweep/Performance_Trace_by_index.png")
    test_dir = REPO / "runs" / args.scan_id / "ModelTraining/test" / args.scan_id

    res = json.loads(metrics_path.read_text(encoding="utf-8"))
    resp = load_resp_bins(test_dir)
    generate_sweep_trace(res, out_path, resp if len(resp) else None)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
