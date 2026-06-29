#!/usr/bin/env python3
"""Export ModelTraining test sweep projections to MP4 (with phase labels)."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
SCAN_ID = os.environ.get("SPARE_SCAN_ID", "CE_P1_V_01")


def normalize_frames(frames: list[np.ndarray], p_lo: float = 1.0, p_hi: float = 99.0) -> list[np.ndarray]:
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


def load_resp_bins(test_dir: Path) -> list[int]:
    resp_path = test_dir / "RespBin.csv"
    if not resp_path.is_file():
        return []
    import pandas as pd

    vals = pd.read_csv(resp_path, header=None).values.squeeze()
    return [int(v) for v in np.atleast_1d(vals).ravel()]


def annotate_frame(frame: np.ndarray, proj_idx: int, phase: int | None, angle: float | None) -> np.ndarray:
    """Burn in projection index + breathing phase (requires PIL)."""
    from PIL import Image, ImageDraw, ImageFont

    rgb = np.stack([frame, frame, frame], axis=-1)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    parts = [f"proj {proj_idx:03d}"]
    if phase is not None:
        parts.append(f"phase {phase:02d}")
    if angle is not None:
        parts.append(f"angle {angle:.1f}°")
    draw.rectangle((0, 0, img.width, 18), fill=(0, 0, 0))
    draw.text((4, 2), "  ".join(parts), fill=(255, 255, 0))
    return np.array(img)


def export_mp4(
    test_dir: Path,
    out_path: Path,
    fps: int = 10,
    annotate: bool = True,
) -> int:
    import imageio

    proj_dir = test_dir / "TestProjections"
    files = sorted(proj_dir.glob("Proj_*_bin.npy"), key=lambda p: int(re.search(r"Proj_(\d+)", p.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No TestProjections in {proj_dir}")

    resp_bins = load_resp_bins(test_dir)
    angles = None
    ang_path = test_dir / "Angles.csv"
    if ang_path.is_file():
        import pandas as pd

        angles = pd.read_csv(ang_path, header=None).values.squeeze()
        angles = np.atleast_1d(angles).ravel()

    raw_frames = [np.load(f).astype(np.float32) for f in files]
    frames = normalize_frames(raw_frames)

    if annotate:
        annotated = []
        for i, frame in enumerate(frames):
            proj_idx = i + 1
            phase = resp_bins[i] if i < len(resp_bins) else None
            angle = float(angles[i]) if angles is not None and i < len(angles) else None
            annotated.append(annotate_frame(frame, proj_idx, phase, angle))
        frames = annotated

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, codec="libx264")
    return len(frames)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export test breathing sweep MP4")
    ap.add_argument("--scan-id", default=SCAN_ID)
    ap.add_argument("--test-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--no-annotate", action="store_true")
    args = ap.parse_args()

    test_dir = args.test_dir or (REPO / "runs" / args.scan_id / "ModelTraining/test" / args.scan_id)
    out = args.out or (REPO / "runs" / args.scan_id / "videos" / f"{args.scan_id}_test_sweep.mp4")

    n = export_mp4(test_dir, out, fps=args.fps, annotate=not args.no_annotate)
    print(f"Wrote {n} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
