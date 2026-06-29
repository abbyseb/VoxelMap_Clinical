#!/usr/bin/env python3
"""Export Elekta DRR verification MP4s from Phase 2 train/*.bin projections."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def load_bin(path: Path, size: int = 128) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    n = size * size
    if len(data) < n:
        raise ValueError(f"{path.name}: expected >={n} floats, got {len(data)}")
    return data[:n].reshape((size, size), order="F")


def load_acquired(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if len(data) == 512 * 512:
        return data.reshape((512, 512), order="F")
    if len(data) == 128 * 128:
        return data.reshape((128, 128), order="F")
    side = int(round(len(data) ** 0.5))
    if side * side == len(data):
        return data.reshape((side, side), order="F")
    raise ValueError(f"{path.name}: unexpected size {len(data)}")


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


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
        isColor=len(frames[0].shape) == 3,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for f in frames:
        if f.ndim == 2:
            writer.write(f)
        else:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def phase_gantry_sweep(train_dir: Path, phase: str, out: Path, fps: int) -> int:
    pat = re.compile(rf"^{re.escape(phase)}_Proj_(\d+)\.bin$")
    files = sorted(
        (f for f in train_dir.glob(f"{phase}_Proj_*.bin") if pat.match(f.name)),
        key=lambda p: int(pat.match(p.name).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No DRR bins for phase {phase} in {train_dir}")
    frames = normalize_frames([load_bin(f) for f in files])
    write_mp4(out, frames, fps)
    return len(frames)


def acquired_vs_simulated(train_dir: Path, phase: str, out: Path, fps: int) -> int:
    proj_dir = train_dir / "Proj"
    pat_sim = re.compile(rf"^{re.escape(phase)}_Proj_(\d+)\.bin$")
    sim_files = sorted(
        (f for f in train_dir.glob(f"{phase}_Proj_*.bin") if pat_sim.match(f.name)),
        key=lambda p: int(pat_sim.match(p.name).group(1)),
    )
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for sim in sim_files:
        idx = int(pat_sim.match(sim.name).group(1))
        acq = proj_dir / f"Proj_{idx:05d}.bin"
        if not acq.is_file():
            continue
        sim_img = load_bin(sim)
        acq_img = load_acquired(acq)
        # Resize acquired to 128 for side-by-side
        from scipy.ndimage import zoom

        h, w = acq_img.shape
        acq_small = zoom(acq_img, (128 / h, 128 / w), order=1)[:128, :128]
        pairs.append((acq_small, sim_img))

    if not pairs:
        raise FileNotFoundError("No overlapping acquired/simulated projection pairs")

    acq_frames = normalize_frames([p[0] for p in pairs])
    sim_frames = normalize_frames([p[1] for p in pairs])
    combined = []
    for a, s in zip(acq_frames, sim_frames):
        gap = np.zeros((128, 4), dtype=np.uint8)
        combined.append(np.hstack([a, gap, s]))
    write_mp4(out, combined, fps)
    return len(combined)


def breathing_at_angle(train_dir: Path, proj_idx: int, out: Path, fps: int) -> int:
    frames = []
    for phase in range(1, 11):
        p = train_dir / f"{phase:02d}_Proj_{proj_idx:03d}.bin"
        if p.is_file():
            frames.append(load_bin(p))
    if not frames:
        raise FileNotFoundError(f"No phase bins for projection {proj_idx:03d}")
    write_mp4(out, normalize_frames(frames), fps)
    return len(frames)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export DRR verification MP4s")
    ap.add_argument(
        "--train-dir",
        type=Path,
        default=Path(
            "/Volumes/T7 Shield/DENNIS_BACKUP/VoxelMap-SPARE-Clinical/"
            "runs/CE_P1_V_01/CE_P1_V_01/train"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/Volumes/T7 Shield/DENNIS_BACKUP/VoxelMap-SPARE-Clinical/"
            "runs/CE_P1_V_01/videos"
        ),
    )
    ap.add_argument("--phase", default="06", help="Respiratory phase (06 = exhale)")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    train = args.train_dir.resolve()
    out_dir = args.out_dir.resolve()

    n1 = phase_gantry_sweep(
        train, args.phase, out_dir / f"CE_P1_V_01_DRR_phase{args.phase}_gantry_sweep.mp4", args.fps
    )
    print(f"gantry sweep: {n1} frames -> {out_dir / f'CE_P1_V_01_DRR_phase{args.phase}_gantry_sweep.mp4'}")

    n2 = acquired_vs_simulated(
        train, args.phase, out_dir / f"CE_P1_V_01_DRR_phase{args.phase}_acquired_vs_simulated.mp4", args.fps
    )
    print(f"acquired vs simulated: {n2} frames")

    n3 = breathing_at_angle(train, 1, out_dir / "CE_P1_V_01_DRR_breathing_proj001.mp4", fps=2)
    print(f"breathing cycle @ proj 001: {n3} frames")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
