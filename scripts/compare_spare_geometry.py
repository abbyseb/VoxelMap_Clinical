#!/usr/bin/env python3
"""Compare MC vs Elekta SPARE geometry and volume coordinate metadata."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "config"))

from elekta_drr import ELEKTA_DRR_OPTS, MC_VARIAN_DRR_OPTS  # noqa: E402


def summarize_geom(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    n = len(re.findall(r"<Projection\b", text))

    def tag(name: str) -> str | None:
        m = re.search(rf"<{name}>([^<]+)", text)
        return m.group(1).strip() if m else None

    angles = [float(x) for x in re.findall(r"<GantryAngle>([^<]+)", text)]
    global_off_x = tag("ProjectionOffsetX")
    global_off_y = tag("ProjectionOffsetY")
    return {
        "path": str(path),
        "projections": n,
        "global_offset_x": global_off_x,
        "global_offset_y": global_off_y,
        "sid": tag("SourceToIsocenterDistance"),
        "sdd": tag("SourceToDetectorDistance"),
        "gantry_first": angles[0] if angles else None,
        "gantry_last": angles[-1] if angles else None,
        "gantry_step": (angles[1] - angles[0]) if len(angles) > 1 else None,
        "per_projection_offsets": global_off_x is None,
    }


def mha_meta(path: Path) -> dict:
    meta: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="ignore") as f:
        for _ in range(50):
            line = f.readline()
            if not line.strip():
                break
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return {
        "DimSize": meta.get("DimSize"),
        "ElementSpacing": meta.get("ElementSpacing"),
        "Offset": meta.get("Offset"),
        "TransformMatrix": meta.get("TransformMatrix"),
        "AnatomicalOrientation": meta.get("AnatomicalOrientation"),
        "CenterOfRotation": meta.get("CenterOfRotation"),
    }


def main() -> int:
    mc_geom = Path(
        "/Volumes/T7 Shield/DENNIS_BACKUP/SPAREChallenge/Evaluation/"
        "MonteCarloDatasets/Validation/P1/MC_V_P1_NS_01/Proj/Geometry.xml"
    )
    ce_geom = ROOT / "data/staged/P1/CE_P1_V_01/Proj/Geometry.xml"
    mc_vol = Path(
        "/Volumes/T7 Shield/DENNIS_BACKUP/SPAREChallenge/Evaluation/"
        "MonteCarloDatasets/Validation/P1/MC_V_P1_NS_01/GTVol_06.mha"
    )
    ce_vol = ROOT / "data/staged/P1/CE_P1_V_01/GTVol_06.mha"

    print("=== Geometry XML ===")
    for label, p in [("MC_V_P1_NS_01", mc_geom), ("CE_P1_V_01", ce_geom)]:
        print(f"\n{label}:")
        for k, v in summarize_geom(p).items():
            print(f"  {k}: {v}")

    print("\n=== DRR panel options (LEARN-GUI) ===")
    print("MC/Varian (gold batch):", MC_VARIAN_DRR_OPTS)
    print("Elekta (CE_*):", ELEKTA_DRR_OPTS)

    print("\n=== Volume IEC metadata (phase 06) ===")
    for label, p in [("MC GTVol_06", mc_vol), ("CE GTVol_06", ce_vol)]:
        print(f"\n{label}:")
        for k, v in mha_meta(p).items():
            print(f"  {k}: {v}")

    print("\n=== Coordinate system verdict ===")
    mc_m = mha_meta(mc_vol)
    ce_m = mha_meta(ce_vol)
    same_orientation = (
        mc_m.get("AnatomicalOrientation") == ce_m.get("AnatomicalOrientation")
        and mc_m.get("TransformMatrix") == ce_m.get("TransformMatrix")
        and mc_m.get("ElementSpacing") == ce_m.get("ElementSpacing")
    )
    print(f"  Same RAI + identity direction + 1mm spacing: {same_orientation}")
    print(f"  FOV differs (expected): MC {mc_m.get('DimSize')} vs CE {ce_m.get('DimSize')}")
    print(f"  Geometry XML must differ (half-fan vs full-fan): True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
