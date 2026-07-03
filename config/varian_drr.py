"""
Clinical Varian SPARE (CV_*) DRR settings for LEARN-GUI generate_drrs.

Half-fan geometry — compatible with MC_VARIAN_DRR_OPTS in elekta_drr.py.
P1/P2 use 1024×768; P3–P5 use 1008×752 (see ClinicalVarianDatasets/README_DataInfo.txt).
"""
from __future__ import annotations

import re
from pathlib import Path

VARIAN_DETECTOR_SPACING_MM = (0.388, 0.388, 1.0)
VARIAN_DETECTOR_ORIGIN_MM = (-200.0, -150.0, 0.0)

VARIAN_DRR_P1_P2 = {
    "geometry_source": "xml",
    "detector_origin": VARIAN_DETECTOR_ORIGIN_MM,
    "detector_spacing": VARIAN_DETECTOR_SPACING_MM,
    "detector_size_xy": (1024, 768),
}

VARIAN_DRR_P3_P5 = {
    "geometry_source": "xml",
    "detector_origin": VARIAN_DETECTOR_ORIGIN_MM,
    "detector_spacing": VARIAN_DETECTOR_SPACING_MM,
    "detector_size_xy": (1008, 752),
}


def patient_num_from_scan_id(scan_id: str) -> str:
    m = re.match(r"CV_P(\d+)_", scan_id)
    if not m:
        raise ValueError(f"Cannot parse patient from Varian scan id: {scan_id}")
    return m.group(1)


def varian_drr_opts_for_patient(patient_num: str | int) -> dict:
    p = int(patient_num)
    if p in (1, 2):
        return dict(VARIAN_DRR_P1_P2)
    if p in (3, 4, 5):
        return dict(VARIAN_DRR_P3_P5)
    raise ValueError(f"Unsupported Varian patient number: {patient_num}")


def varian_drr_opts_for_scan(scan_dir: Path, scan_id: str | None = None) -> dict:
    """Return DRR kwargs with geometry_path set to the scan's Proj/Geometry.xml."""
    scan_dir = Path(scan_dir)
    geom = scan_dir / "Proj" / "Geometry.xml"
    if not geom.is_file():
        geom = scan_dir / "train" / "Proj" / "Geometry.xml"
    if not geom.is_file():
        raise FileNotFoundError(f"Missing Varian geometry: {scan_dir}")

    if scan_id is None:
        parent = scan_dir.name
        if parent == "train":
            parent = scan_dir.parent.name
        scan_id = parent

    pnum = patient_num_from_scan_id(scan_id)
    return {**varian_drr_opts_for_patient(pnum), "geometry_path": str(geom.resolve())}
