#!/usr/bin/env python3
"""Export DVF warp visualization MP4: 2x2 panel on white background.

Panels:
  [Source projection]  [Target projection]
  [Source volume + DVF arrows]  [Warped volume]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
SCAN_ID = os.environ.get("SPARE_SCAN_ID", "CE_P1_V_01")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.utilities import networksFiLM, spatialTransform
from ml.volume_view import VolumeViewConfig, extract_flow_uv, extract_slice


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def build_sweep_index(test_dir: Path) -> list[dict]:
    import pandas as pd

    tgt_dir = test_dir / "TestProjections"
    src_dir = test_dir / "SourceTestProjections"
    if not src_dir.is_dir():
        src_dir = test_dir / "SourceProjections"
    angles_path = test_dir / "Angles.csv"
    resp_path = test_dir / "RespBin.csv"

    angles = None
    if angles_path.is_file():
        angles = np.atleast_1d(pd.read_csv(angles_path, header=None).values.squeeze()).ravel()
    resp_bins = None
    if resp_path.is_file():
        resp_bins = np.atleast_1d(pd.read_csv(resp_path, header=None).values.squeeze()).ravel()

    samples = []
    for tgt_file in sorted(tgt_dir.glob("Proj_*_bin.npy"), key=lambda p: int(re.search(r"Proj_(\d+)", p.name).group(1))):
        m = re.search(r"Proj_(\d+)", tgt_file.name)
        if not m:
            continue
        proj_idx = int(m.group(1))
        proj_str = f"{proj_idx:03d}"
        src_file = src_dir / f"06_Proj_{proj_str}_bin.npy"
        if not src_file.is_file():
            continue
        phase = int(resp_bins[proj_idx - 1]) if resp_bins is not None and proj_idx - 1 < len(resp_bins) else -1
        angle = float(angles[proj_idx - 1]) if angles is not None and proj_idx - 1 < len(angles) else 0.0
        samples.append(
            {
                "proj_idx": proj_idx,
                "phase": phase,
                "angle": angle,
                "source_proj": src_file,
                "target_proj": tgt_file,
            }
        )
    return samples


def gray_to_rgb_u8(gray: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    g = np.clip((gray.astype(np.float64) - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    u8 = (g * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray(rgb)
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.array(img)


def draw_quiver(rgb: np.ndarray, u: np.ndarray, v: np.ndarray, stride: int, color=(220, 50, 30)) -> np.ndarray:
    """Overlay in-plane flow arrows (u,v) on HxW RGB image."""
    out = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(out)
    h, w = u.shape
    scale = max(h, w) * 0.11
    for row in range(0, h, stride):
        for col in range(0, w, stride):
            du, dv = float(u[row, col]), float(v[row, col])
            mag = (du * du + dv * dv) ** 0.5
            if mag < 1e-6:
                continue
            dx = (du / mag) * scale
            dy = (dv / mag) * scale
            x0, y0 = col, row
            x1 = int(np.clip(x0 + dx, 0, w - 1))
            y1 = int(np.clip(y0 + dy, 0, h - 1))
            draw.line((x0, y0, x1, y1), fill=color, width=2)
            # arrowhead
            shaft = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if shaft > 3:
                import math

                ang = math.atan2(y1 - y0, x1 - x0)
                hl = min(8, max(4, 0.3 * shaft))
                for da in (2.6, -2.6):
                    hx = x1 - hl * math.cos(ang + da)
                    hy = y1 - hl * math.sin(ang + da)
                    draw.line((x1, y1, int(hx), int(hy)), fill=color, width=2)
    return np.array(out)


def compose_panel(
    tiles: list[tuple[str, np.ndarray]],
    header: str,
    tile_px: int,
    margin: int,
    header_h: int,
    label_h: int,
) -> np.ndarray:
    cols = 2
    rows = 2
    inner_gap = 12
    canvas_w = margin * 2 + cols * tile_px + inner_gap
    canvas_h = margin + header_h + rows * (label_h + tile_px) + inner_gap + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font_h = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_l = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font_h = ImageFont.load_default()
        font_l = ImageFont.load_default()

    draw.text((margin, margin), header, fill=(20, 20, 20), font=font_h)

    y0 = margin + header_h
    for i, (title, rgb) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = margin + c * (tile_px + inner_gap)
        y = y0 + r * (label_h + tile_px + inner_gap)
        draw.text((x, y), title, fill=(60, 60, 60), font=font_l)
        tile = Image.fromarray(resize_square(rgb, tile_px))
        # light border
        draw.rectangle((x - 1, y + label_h - 1, x + tile_px, y + label_h + tile_px), outline=(200, 200, 200))
        canvas.paste(tile, (x, y + label_h))
    return np.array(canvas)


def export_dvf_warp_mp4(
    checkpoint: Path,
    test_dir: Path,
    out_path: Path,
    device: str = "cuda",
    fps: int = 8,
    tile_px: int = 280,
    arrow_stride: int = 10,
    max_frames: int = 0,
    view: VolumeViewConfig | None = None,
) -> int:
    import imageio

    samples = build_sweep_index(test_dir)
    if max_frames > 0:
        samples = samples[:max_frames]
    if not samples:
        raise FileNotFoundError(f"No sweep samples in {test_dir}")

    dev = torch.device(device)
    model = networksFiLM.Model.load(str(checkpoint), str(dev))
    model = model.to(dev).eval()
    transformer = spatialTransform.Network([128, 128, 128]).to(dev).eval()

    src_vol_path = test_dir / "SourceVolumes" / "sub_CT_06_mha.npy"
    if not src_vol_path.is_file():
        src_vol_path = next((test_dir / "SourceVolumes").glob("sub_CT_*.npy"))
    src_vol_np = _normalize(np.load(src_vol_path).squeeze())
    view_cfg = (view or VolumeViewConfig()).resolve()
    if view_cfg.slice_index == 64 and src_vol_np.shape[0] != 128:
        view_cfg.slice_index = src_vol_np.shape[int(view_cfg.slice_axis)] // 2

    # Global display ranges
    all_src_p, all_tgt_p = [], []
    for s in samples[: min(32, len(samples))]:
        all_src_p.append(_normalize(np.load(s["source_proj"])))
        all_tgt_p.append(_normalize(np.load(s["target_proj"])))
    p_lo = float(np.percentile(np.stack(all_src_p + all_tgt_p), 1))
    p_hi = float(np.percentile(np.stack(all_src_p + all_tgt_p), 99))
    v_lo, v_hi = float(np.percentile(src_vol_np, 1)), float(np.percentile(src_vol_np, 99))

    frames = []
    with torch.no_grad():
        src_vol_t = torch.from_numpy(src_vol_np[None, None]).float().to(dev)
        for s in tqdm(samples, desc="DVF warp MP4"):
            src_p = _normalize(np.load(s["source_proj"]))
            tgt_p = _normalize(np.load(s["target_proj"]))
            angle = torch.tensor([s["angle"]], dtype=torch.float32, device=dev)

            src_pt = torch.from_numpy(src_p[None, None]).float().to(dev)
            tgt_pt = torch.from_numpy(tgt_p[None, None]).float().to(dev)

            _, pred_flow = model(src_pt, tgt_pt, src_vol_t, angle=angle)
            warped = transformer(src_vol_t, pred_flow)[0, 0].cpu().numpy()
            flow_np = pred_flow[0].cpu().numpy()  # 3,D,H,W

            src_slice = extract_slice(src_vol_np, view_cfg)
            warped_slice = extract_slice(_normalize(warped), view_cfg)
            u, v = extract_flow_uv(flow_np, view_cfg)

            src_rgb = gray_to_rgb_u8(src_slice, v_lo, v_hi)
            src_arrow_rgb = draw_quiver(src_rgb, u, v, stride=arrow_stride)
            warped_rgb = gray_to_rgb_u8(warped_slice, v_lo, v_hi)
            src_proj_rgb = gray_to_rgb_u8(src_p, p_lo, p_hi)
            tgt_proj_rgb = gray_to_rgb_u8(tgt_p, p_lo, p_hi)

            meta = (
                f"proj {s['proj_idx']:03d}  |  phase {s['phase']:02d}  |  angle {s['angle']:.1f}°"
                f"  |  {view_cfg.plane} slice {view_cfg.clamp_slice(src_vol_np.shape)} ⊥ {view_cfg.slice_normal}"
            )
            frame = compose_panel(
                [
                    ("Source projection (06)", src_proj_rgb),
                    ("Target projection", tgt_proj_rgb),
                    ("Source volume + DVF arrows", src_arrow_rgb),
                    ("Warped volume", warped_rgb),
                ],
                header=meta,
                tile_px=tile_px,
                margin=24,
                header_h=22,
                label_h=18,
            )
            frames.append(frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Pad to multiple of 16 for H.264 compatibility
    h0, w0 = frames[0].shape[:2]
    pad_h = (16 - h0 % 16) % 16
    pad_w = (16 - w0 % 16) % 16
    if pad_h or pad_w:
        frames = [
            np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=255)
            for f in frames
        ]
    imageio.mimsave(str(out_path), frames, fps=fps, codec="libx264")
    return len(frames)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export DVF warp 2x2 panel MP4")
    ap.add_argument("--scan-id", default=SCAN_ID)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--test-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--tile-px", type=int, default=280)
    ap.add_argument("--arrow-stride", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--view-config", type=Path, default=None, help="JSON from volume orientation GUI")
    ap.add_argument("--plane", choices=("axial", "sagittal", "coronal"), default=None)
    ap.add_argument("--slice-index", type=int, default=None)
    ap.add_argument("--flip-h", action="store_true")
    ap.add_argument("--flip-v", action="store_true")
    args = ap.parse_args()

    view = VolumeViewConfig(scan_id=args.scan_id)
    if args.view_config and args.view_config.is_file():
        view = VolumeViewConfig.load_json(args.view_config)
    if args.plane:
        view.plane = args.plane
        view.slice_axis = view.h_axis = view.v_axis = view.flow_u = view.flow_v = None
    if args.slice_index is not None:
        view.slice_index = args.slice_index
    if args.flip_h:
        view.flip_h = True
    if args.flip_v:
        view.flip_v = True
    view.resolve()

    ckpt = args.checkpoint or (REPO / "runs" / args.scan_id / "checkpoints/best.pt")
    test_dir = args.test_dir or (REPO / "runs" / args.scan_id / "ModelTraining/test" / args.scan_id)
    out = args.out or (REPO / "runs" / args.scan_id / "videos" / f"{args.scan_id}_dvf_warp_panels.mp4")

    n = export_dvf_warp_mp4(
        ckpt,
        test_dir,
        out,
        device=args.device,
        fps=args.fps,
        tile_px=args.tile_px,
        arrow_stride=args.arrow_stride,
        max_frames=args.max_frames,
        view=view,
    )
    print(f"Wrote {n} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
