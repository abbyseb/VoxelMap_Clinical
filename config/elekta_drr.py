"""
Elekta Clinical SPARE (CE_*) DRR settings for LEARN-GUI generate_drrs.

Use the per-scan Proj/Geometry.xml from SPARE — never Geometry_SPARE.xml (MC/Varian half-fan).

Verified against CE_P1_V_01 (2026-06-29). See ELEKTA_DRR_VERIFICATION.md.
"""
from __future__ import annotations

from pathlib import Path

# SPARE Clinical Elekta (full-fan) — Shieh et al. / SPARE README §13
ELEKTA_SID_MM = 1000.0
ELEKTA_SDD_MM = 1536.0
ELEKTA_DETECTOR_SPACING_MM = (0.8, 0.8, 1.0)
ELEKTA_DETECTOR_SIZE_XY = (512, 512)
# Lower-left detector corner (mm): center 512×512 panel at isocenter (IEC, RTK convention)
ELEKTA_DETECTOR_ORIGIN_MM = (-204.8, -204.8, 0.0)

# Monte Carlo / Varian half-fan gold-batch defaults (do NOT use for Elekta)
MC_VARIAN_DRR_OPTS = {
    "geometry_source": "xml",
    "detector_origin": (-200.0, -150.0, 0.0),
    "detector_spacing": (0.388, 0.388, 1.0),
    "detector_size_xy": (1024, 768),
}

ELEKTA_DRR_OPTS = {
    "geometry_source": "xml",
    "detector_origin": ELEKTA_DETECTOR_ORIGIN_MM,
    "detector_spacing": ELEKTA_DETECTOR_SPACING_MM,
    "detector_size_xy": ELEKTA_DETECTOR_SIZE_XY,
}


def elekta_drr_opts_for_scan(scan_dir: Path) -> dict:
    """Return DRR kwargs with geometry_path set to the scan's Proj/Geometry.xml."""
    geom = Path(scan_dir) / "Proj" / "Geometry.xml"
    if not geom.is_file():
        raise FileNotFoundError(f"Missing Elekta geometry: {geom}")
    return {**ELEKTA_DRR_OPTS, "geometry_path": str(geom.resolve())}


def mc_varian_drr_opts_for_scan(scan_dir: Path) -> dict:
    """MC/CV gold-batch style (half-fan 1024×768)."""
    geom = Path(scan_dir) / "Proj" / "Geometry.xml"
    if not geom.is_file():
        geom = Path(scan_dir) / "train" / "Proj" / "Geometry.xml"
    if not geom.is_file():
        raise FileNotFoundError(f"Missing MC/Varian geometry: {scan_dir}")
    return {**MC_VARIAN_DRR_OPTS, "geometry_path": str(geom.resolve())}
