#!/usr/bin/env python3
"""Compare sweep metrics: FDK-volume pilot vs standard GTVol run."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_summary(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {
        "3d_error_mm": statistics.mean(d["shifts_mm"]["3d"]),
        "ssim": statistics.mean(d["ssim"]),
        "psnr_db": statistics.mean(d["psnr"]),
        "dice": statistics.mean(d["dice"]),
        "neg_det_j": statistics.mean(d["det_j_neg_fraction"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtvol-scan", default="CE_P1_V_01")
    ap.add_argument("--fdk-scan", default="CE_P1_V_01_fdk")
    ap.add_argument("--gtvol-metrics", type=Path, default=None)
    ap.add_argument("--fdk-metrics", type=Path, default=None)
    args = ap.parse_args()

    gt_path = args.gtvol_metrics or (
        REPO / "results" / args.gtvol_scan / "sweep_metrics.json"
    )
    fdk_path = args.fdk_metrics or (
        REPO / "runs" / args.fdk_scan / "eval_sweep" / "metrics.json"
    )
    if not fdk_path.is_file():
        fdk_path = REPO / "results" / args.fdk_scan / "sweep_metrics.json"

    gt = load_summary(gt_path)
    fdk = load_summary(fdk_path)

    rows = [
        ("3D error (mm)", "3d_error_mm", ".3f"),
        ("SSIM", "ssim", ".4f"),
        ("PSNR (dB)", "psnr_db", ".2f"),
        ("Dice", "dice", ".4f"),
        ("neg det(J) frac", "neg_det_j", ".4f"),
    ]
    print(f"GTVol metrics: {gt_path}")
    print(f"FDK metrics:   {fdk_path}")
    print()
    print(f"| Metric | {args.gtvol_scan} (GTVol) | {args.fdk_scan} (FDK4D) | Delta (FDK−GTVol) |")
    print("|--------|---------------------|----------------------|---------------------|")
    for label, key, fmt in rows:
        a, b = gt[key], fdk[key]
        delta = b - a
        sign = "+" if delta >= 0 else ""
        print(f"| {label} | {a:{fmt}} | {b:{fmt}} | {sign}{delta:{fmt}} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
