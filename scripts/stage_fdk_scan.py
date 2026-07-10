#!/usr/bin/env python3
"""Stage Elekta or Varian scan using FDKRecon/FDK4D_* as GTVol_* symlinks."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPARE = Path(
    "/home/abhishek/research-data/2RESEARCH/1_ClinicalData/SPAREChallenge"
)

VENDOR = {
    "CE": ("ClinicalElektaDatasets", re.compile(r"CE_P(\d+)_")),
    "CV": ("ClinicalVarianDatasets", re.compile(r"CV_P(\d+)_")),
}


def parse_scan(scan_id: str) -> tuple[str, str, str]:
    prefix = scan_id[:2]
    if prefix not in VENDOR:
        raise ValueError(f"Unsupported scan id: {scan_id} (expected CE_* or CV_*)")
    dataset, pat = VENDOR[prefix]
    m = pat.match(scan_id)
    if not m:
        raise ValueError(f"Cannot parse patient from: {scan_id}")
    return prefix, dataset, m.group(1)


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
    _prefix, dataset, pnum = parse_scan(source_scan_id)
    staged_scan_id = staged_scan_id or f"{source_scan_id}_fdk"
    participant = (
        spare_root / "Participant_Datasets" / dataset / f"P{pnum}" / source_scan_id
    )
    evaluation = spare_root / "Evaluation" / dataset / f"P{pnum}" / source_scan_id
    fdk_dir = participant / "FDKRecon"
    out = (staged_root or REPO / "data/staged_fdk") / f"P{pnum}" / staged_scan_id
    out.mkdir(parents=True, exist_ok=True)

    proj_src = participant / "Proj"
    if not (proj_src / "Geometry.xml").is_file():
        raise FileNotFoundError(f"Missing Geometry.xml: {proj_src}")
    symlink(proj_src, out / "Proj")

    fdk_vols = sorted(fdk_dir.glob("FDK4D_*.mha"))
    if len(fdk_vols) != 10:
        raise FileNotFoundError(f"Expected 10 FDK4D_*.mha in {fdk_dir}, found {len(fdk_vols)}")
    for src in fdk_vols:
        m = re.search(r"FDK4D_(\d+)", src.name)
        if m:
            symlink(src, out / f"GTVol_{m.group(1)}.mha")

    for src in sorted(evaluation.glob("Mask_*.mha")):
        symlink(src, out / src.name)

    n_gt = len(list(out.glob("GTVol_*.mha")))
    n_mask = len(list(out.glob("Mask_*.mha")))
    print(f"Staged FDK {staged_scan_id} -> {out} ({n_gt} FDK4D volumes, {n_mask} masks)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage FDK4D volumes as GTVol_* + masks")
    ap.add_argument("source_scan_ids", nargs="+", help="e.g. CE_P1_V_01 CV_P2_V_01")
    ap.add_argument("--staged-scan-id", default=None, help="Override folder name (single scan only)")
    ap.add_argument("--spare-root", type=Path, default=DEFAULT_SPARE)
    ap.add_argument("--staged-root", type=Path, default=REPO / "data/staged_fdk")
    args = ap.parse_args()

    if len(args.source_scan_ids) > 1 and args.staged_scan_id:
        raise SystemExit("--staged-scan-id only valid with one source scan")

    for scan_id in args.source_scan_ids:
        sid = args.staged_scan_id if len(args.source_scan_ids) == 1 else None
        stage_fdk_scan(scan_id, args.spare_root.resolve(), args.staged_root.resolve(), sid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
