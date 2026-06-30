#!/usr/bin/env python3
"""Symlink Elekta SPARE participant + evaluation data into data/staged/."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPARE = Path(
    "/home/abhishek/research-data/2RESEARCH/1_ClinicalData/SPAREChallenge"
)


def patient_num(scan_id: str) -> str:
    m = re.match(r"CE_P(\d+)_", scan_id)
    if not m:
        raise ValueError(f"Cannot parse patient from scan id: {scan_id}")
    return m.group(1)


def symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    dst.symlink_to(src)


def stage_scan(
    scan_id: str,
    spare_root: Path,
    staged_root: Path | None = None,
) -> Path:
    pnum = patient_num(scan_id)
    participant = (
        spare_root
        / "Participant_Datasets/ClinicalElektaDatasets"
        / f"P{pnum}"
        / scan_id
    )
    evaluation = (
        spare_root / "Evaluation/ClinicalElektaDatasets" / f"P{pnum}" / scan_id
    )
    out = (staged_root or REPO / "data/staged") / f"P{pnum}" / scan_id
    out.mkdir(parents=True, exist_ok=True)

    proj_src = participant / "Proj"
    if not (proj_src / "Geometry.xml").is_file():
        raise FileNotFoundError(f"Missing Geometry.xml: {proj_src}")

    symlink(proj_src, out / "Proj")

    for pat in ("GTVol_*.mha", "Mask_*.mha"):
        for src in sorted(evaluation.glob(pat)):
            symlink(src, out / src.name)

    n_gt = len(list(out.glob("GTVol_*.mha")))
    n_mask = len(list(out.glob("Mask_*.mha")))
    if n_gt == 0:
        raise FileNotFoundError(f"No GTVol files in {evaluation}")
    print(f"Staged {scan_id} -> {out} ({n_gt} GTVol, {n_mask} masks)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage Elekta scan via symlinks")
    ap.add_argument("scan_ids", nargs="+", help="e.g. CE_P2_V_01 CE_P3_V_01")
    ap.add_argument("--spare-root", type=Path, default=DEFAULT_SPARE)
    ap.add_argument("--staged-root", type=Path, default=REPO / "data/staged")
    args = ap.parse_args()

    for scan_id in args.scan_ids:
        stage_scan(scan_id, args.spare_root.resolve(), args.staged_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
