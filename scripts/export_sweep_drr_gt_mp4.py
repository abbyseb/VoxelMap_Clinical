#!/usr/bin/env python3
"""Export breathing-sweep MP4: synthetic DRR (from CT volumes) vs acquired onboard CBCT.

Left panel  = TestProjections (DRRs from staged volumes, e.g. FDK4D or GTVol).
Right panel = acquired Proj_*.bin at the same gantry index (clinical reference).
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
SCAN_ID = os.environ.get("SPARE_SCAN_ID", "CE_P1_V_01")


def load_acquired(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    n = int(len(data))
    # Elekta: 512x512. Varian: 1024x768 (P1/P2), 1006x750 acquired (P3-P5), 1008x752 DRR.
    known_shapes = (
        (512, 512),
        (768, 1024),
        (750, 1006),
        (1006, 750),
        (752, 1008),
        (1008, 752),
        (128, 128),
    )
    for h, w in known_shapes:
        if h * w == n:
            return data.reshape((h, w), order="F")
    side = int(round(n ** 0.5))
    if side * side == n:
        return data.reshape((side, side), order="F")
    raise ValueError(f"{path.name}: unexpected size {n}")


def resize_to(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img.astype(np.float32)
    from scipy.ndimage import zoom

    h, w = img.shape
    return zoom(img, (size / h, size / w), order=1)[:size, :size].astype(np.float32)


def normalize_stack(frames: list[np.ndarray], p_lo: float = 1.0, p_hi: float = 99.0) -> list[np.ndarray]:
    stack = np.stack(frames, axis=0)
    vmin = float(np.percentile(stack, p_lo))
    vmax = float(np.percentile(stack, p_hi))
    if vmax <= vmin:
        vmax = vmin + 1.0
    out = []
    for f in frames:
        g = np.clip((f - vmin) / (vmax - vmin), 0.0, 1.0)
        out.append((g * 255.0).astype(np.uint8))
    return out


def annotate_panel(
    frame: np.ndarray,
    title: str,
    proj_idx: int,
    phase: int | None,
    angle: float | None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    rgb = np.stack([frame, frame, frame], axis=-1)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 32), fill=(0, 0, 0))
    draw.text((4, 2), title, fill=(0, 255, 255))
    parts = [f"proj {proj_idx:03d}"]
    if phase is not None:
        parts.append(f"phase {phase:02d}")
    if angle is not None:
        parts.append(f"{angle:.1f}°")
    draw.text((4, 16), "  ".join(parts), fill=(255, 255, 0))
    return np.array(img)


def export_sweep_drr_gt_mp4(
    test_dir: Path,
    proj_dir: Path,
    out_path: Path,
    *,
    sim_label: str = "Sim DRR",
    acq_label: str = "Acquired CBCT",
    tile_px: int = 256,
    fps: int = 10,
    gap_px: int = 6,
) -> int:
    import imageio

    sim_files = sorted(
        test_dir.glob("TestProjections/Proj_*_bin.npy"),
        key=lambda p: int(re.search(r"Proj_(\d+)", p.name).group(1)),
    )
    if not sim_files:
        raise FileNotFoundError(f"No TestProjections in {test_dir}")

    resp_bins: list[int] = []
    resp_path = test_dir / "RespBin.csv"
    if resp_path.is_file():
        import pandas as pd

        resp_bins = [int(v) for v in np.atleast_1d(pd.read_csv(resp_path, header=None).values.squeeze()).ravel()]

    angles = None
    ang_path = test_dir / "Angles.csv"
    if ang_path.is_file():
        import pandas as pd

        angles = np.atleast_1d(pd.read_csv(ang_path, header=None).values.squeeze()).ravel()

    sim_raw: list[np.ndarray] = []
    acq_raw: list[np.ndarray] = []
    meta: list[tuple[int, int | None, float | None]] = []

    for sim_file in sim_files:
        m = re.search(r"Proj_(\d+)", sim_file.name)
        if not m:
            continue
        proj_idx = int(m.group(1))
        acq_file = proj_dir / f"Proj_{proj_idx:05d}.bin"
        if not acq_file.is_file():
            continue
        sim_raw.append(resize_to(np.load(sim_file).astype(np.float32), tile_px))
        acq_raw.append(resize_to(load_acquired(acq_file), tile_px))
        phase = resp_bins[proj_idx - 1] if proj_idx - 1 < len(resp_bins) else None
        angle = float(angles[proj_idx - 1]) if angles is not None and proj_idx - 1 < len(angles) else None
        meta.append((proj_idx, phase, angle))

    if not sim_raw:
        raise FileNotFoundError(f"No overlapping sim/acquired pairs under {test_dir} and {proj_dir}")

    sim_u8 = normalize_stack(sim_raw)
    acq_u8 = normalize_stack(acq_raw)
    gap = np.zeros((tile_px, gap_px, 3), dtype=np.uint8)

    frames = []
    for i, (proj_idx, phase, angle) in enumerate(meta):
        left = annotate_panel(sim_u8[i], sim_label, proj_idx, phase, angle)
        right = annotate_panel(acq_u8[i], acq_label, proj_idx, phase, angle)
        frames.append(np.hstack([left, gap, right]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, codec="libx264")
    return len(frames)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep DRR vs acquired CBCT MP4")
    ap.add_argument("--scan-id", default=SCAN_ID)
    ap.add_argument("--test-dir", type=Path, default=None)
    ap.add_argument("--proj-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sim-label", default=None, help="Left panel title (default: Sim DRR (FDK) if scan_id has _fdk)")
    ap.add_argument("--acq-label", default="Acquired CBCT")
    ap.add_argument("--tile-px", type=int, default=256)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    run_root = REPO / "runs" / args.scan_id
    test_dir = args.test_dir or (run_root / "ModelTraining/test" / args.scan_id)
    proj_dir = args.proj_dir or (run_root / args.scan_id / "train" / "Proj")
    if not proj_dir.is_file() and not (proj_dir / "Geometry.xml").is_file():
        alt = run_root / args.scan_id / "train" / "Proj"
        if (alt / "Geometry.xml").is_file():
            proj_dir = alt

    sim_label = args.sim_label
    if sim_label is None:
        sim_label = "Sim DRR (FDK)" if "_fdk" in args.scan_id.lower() else "Sim DRR (GTVol)"

    out = args.out or (
        run_root / "videos" / f"{args.scan_id}_sweep_drr_vs_acquired.mp4"
    )

    n = export_sweep_drr_gt_mp4(
        test_dir,
        proj_dir,
        out,
        sim_label=sim_label,
        acq_label=args.acq_label,
        tile_px=args.tile_px,
        fps=args.fps,
    )
    print(f"Wrote {n} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
