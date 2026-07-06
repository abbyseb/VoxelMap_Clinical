#!/usr/bin/env python3
"""Stage Elekta scan using FDKRecon/FDK4D_* as volume GT (symlinked as GTVol_*).

Does not modify data/staged/ used by the standard GTVol pipeline.
"""
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


def stage_fdk_scan(
    source_scan_id: str,
    spare_root: Path,
    staged_root: Path | None = None,
    staged_scan_id: str | None = None,
) -> Path:
    """Stage Proj + FDK4D (as GTVol_*) + Evaluation masks."""
    pnum = patient_num(source_scan_id)
    staged_scan_id = staged_scan_id or f"{source_scan_id}_fdk"
    participant = (
        spare_root
        / "Participant_Datasets/ClinicalElektaDatasets"
        / f"P{pnum}"
        / source_scan_id
    )
    evaluation = (
        spare_root / "Evaluation/ClinicalElektaDatasets" / f"P{pnum}" / source_scan_id
    )
    fdk_dir = participant / "FDKRecon"
    out = (staged_root or REPO / "data/staged_fdk") / f"P{pnum}" / staged_scan_id
    out.mkdir(parents=True, exist_ok=True)

    proj_src = participant / "Proj"
    if not (proj_src / "Geometry.xml").is_file():
        raise FileNotFoundError(f"Missing Geometry.xml: {proj_src}")
    symlink(proj_src, out / "Proj")

    fdk_vols = sorted(fdk_dir.glob("FDK4D_*.mha"))
    if len(fdk_vols) != 10:
        raise FileNotFoundError(
            f"Expected 10 FDK4D_*.mha in {fdk_dir}, found {len(fdk_vols)}"
        )
    for src in fdk_vols:
        m = re.search(r"FDK4D_(\d+)", src.name)
        if not m:
            continue
        symlink(src, out / f"GTVol_{m.group(1)}.mha")

    for pat in ("Mask_*.mha",):
        for src in sorted(evaluation.glob(pat)):
            symlink(src, out / src.name)

    n_gt = len(list(out.glob("GTVol_*.mha")))
    n_mask = len(list(out.glob("Mask_*.mha")))
    print(
        f"Staged FDK volumes as GTVol_*: {staged_scan_id} -> {out} "
        f"({n_gt} volumes from {fdk_dir.name}, {n_mask} masks from Evaluation)"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage Elekta FDK4D volumes (as GTVol_*) + masks"
    )
    ap.add_argument(
        "--source-scan-id",
        default="CE_P1_V_01",
        help="Participant scan with FDKRecon/ (default CE_P1_V_01)",
    )
    ap.add_argument(
        "--staged-scan-id",
        default=None,
        help="Folder name under staged_fdk/ (default {source}_fdk)",
    )
    ap.add_argument("--spare-root", type=Path, default=DEFAULT_SPARE)
    ap.add_argument("--staged-root", type=Path, default=REPO / "data/staged_fdk")
    args = ap.parse_args()

    stage_fdk_scan(
        args.source_scan_id,
        args.spare_root.resolve(),
        args.staged_root.resolve(),
        args.staged_scan_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
