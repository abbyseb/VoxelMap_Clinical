#!/usr/bin/env python3
"""Run full Elekta pipeline: stage → phase2 → train → eval for one or more scans."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
SCRIPTS = REPO / "scripts"


def patient_num(scan_id: str) -> str:
    m = re.match(r"CE_P(\d+)_", scan_id)
    if not m:
        raise ValueError(scan_id)
    return m.group(1)


def run(cmd: list[str], *, env: dict | None = None) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO), env=env or os.environ, check=True)


def export_results(scan_id: str) -> None:
    """Copy key artifacts to results/ for git tracking."""
    dst = REPO / "results" / scan_id
    dst.mkdir(parents=True, exist_ok=True)
    run_root = REPO / "runs" / scan_id

    copies = [
        (run_root / "plots/loss_curves.png", dst / "plots/loss_curves.png"),
        (run_root / "plots/loss_history.json", dst / "loss_history.json"),
        (run_root / "eval_sweep/metrics.json", dst / "sweep_metrics.json"),
        (run_root / "eval_sweep/Performance_Trace.png", dst / "plots/Performance_Trace.png"),
        (
            run_root / "eval_sweep/Performance_Trace_by_index.png",
            dst / "plots/Performance_Trace_by_index.png",
        ),
    ]
    for src, out in copies:
        if src.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            print(f"  copied {src.name} -> {out}")


def process_scan(
    scan_id: str,
    *,
    skip_stage: bool,
    skip_train: bool,
    skip_eval: bool,
    epochs: int,
    gpu: int,
) -> None:
    pnum = patient_num(scan_id)
    staged = REPO / "data/staged" / f"P{pnum}" / scan_id
    run_root = REPO / "runs" / scan_id
    python = sys.executable

    env = os.environ.copy()
    env["VOXELMAP_CLINICAL_ROOT"] = str(REPO)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    t0 = time.perf_counter()
    print(f"\n{'=' * 60}\n{scan_id}\n{'=' * 60}")

    if not skip_stage:
        run([python, str(SCRIPTS / "stage_elekta_scan.py"), scan_id], env=env)

    run(
        [
            python,
            str(SCRIPTS / "run_elekta_phase2.py"),
            "--scan-id",
            scan_id,
            "--run-root",
            str(run_root),
            "--staged",
            str(staged),
            "--with-test",
        ],
        env=env,
    )

    if not skip_train:
        run(
            [
                python,
                str(SCRIPTS / "run_elekta_phase3_train.py"),
                "--scan-id",
                scan_id,
                "--epochs",
                str(epochs),
                "--gpu",
                str(gpu),
            ],
            env=env,
        )

    if not skip_eval:
        run(
            [python, str(SCRIPTS / "run_elekta_phase4_eval.py"), "--scan-id", scan_id, "--gpu", str(gpu)],
            env=env,
        )
        run(
            [python, str(SCRIPTS / "run_elekta_sweep_eval.py"), "--scan-id", scan_id, "--gpu", str(gpu)],
            env=env,
        )
        export_results(scan_id)

    print(f"{scan_id} done in {(time.perf_counter() - t0) / 60:.1f} min")


def main() -> int:
    ap = argparse.ArgumentParser(description="Full Elekta SPARE pipeline")
    ap.add_argument(
        "scan_ids",
        nargs="*",
        default=["CE_P2_V_01", "CE_P3_V_01", "CE_P4_V_01", "CE_P5_V_01"],
    )
    ap.add_argument("--skip-stage", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    for scan_id in args.scan_ids:
        process_scan(
            scan_id,
            skip_stage=args.skip_stage,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
            epochs=args.epochs,
            gpu=args.gpu,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
