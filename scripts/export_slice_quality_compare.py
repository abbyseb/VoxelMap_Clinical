#!/usr/bin/env python3
"""White-background slice-quality comparison: 128³ (downsampled) vs native CT.

Renders N matched slices (same physical location) side by side at the training
128³ resolution and at the original scan resolution, so the loss of detail from
downsampling can be inspected visually. Output is a single labelled PNG.

Layout (rows = slices, columns = resolution):

    [ header: <scan> — <plane> slice-quality comparison ]

                 128³ (train)        native (original)
    slice A   [ sub_CT tile ]      [ native CT tile ]
    slice B   [ sub_CT tile ]      [ native CT tile ]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.dynamic_dataset import resolve_voxel_map_data_root
from ml.flow_utils import load_mha_array
from ml.volume_view import AXIS_NAMES, VolumeViewConfig, extract_slice
from scripts.run_fullres_dvf_eval import _resolve_native_paths


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _font_bold(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def gray_window_rgb(gray: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    g = np.clip((gray.astype(np.float64) - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    u8 = (g * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def fit_into_box(rgb: np.ndarray, box: int, interp: Image.Resampling) -> np.ndarray:
    """Resize preserving aspect ratio, pad to a square box on white."""
    h, w = rgb.shape[:2]
    scale = box / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    img = Image.fromarray(rgb).resize((nw, nh), interp)
    canvas = Image.new("RGB", (box, box), (255, 255, 255))
    canvas.paste(img, ((box - nw) // 2, (box - nh) // 2))
    return np.array(canvas)


def render_tile(vol: np.ndarray, plane: str, slice_index: int, window, box: int,
                interp: Image.Resampling) -> tuple[np.ndarray, tuple[int, int], int]:
    view = VolumeViewConfig(scan_id="", plane=plane).resolve()
    view.slice_index = int(np.clip(slice_index, 0, vol.shape[int(view.slice_axis)] - 1))
    sl = extract_slice(vol, view)
    rgb = gray_window_rgb(sl, window[0], window[1])
    tile = fit_into_box(rgb, box, interp)
    return tile, sl.shape, view.slice_index


def main() -> int:
    ap = argparse.ArgumentParser(description="128³ vs native slice-quality comparison PNG")
    ap.add_argument("--scan-id", default="CE_P1_V_01_fdk")
    ap.add_argument("--plane", choices=("axial", "sagittal", "coronal"), default="sagittal")
    ap.add_argument(
        "--slice-fracs",
        default="0.4,0.6",
        help="Comma-separated fractional slice positions in [0,1] (matched across both volumes).",
    )
    ap.add_argument("--tile-px", type=int, default=340)
    ap.add_argument(
        "--interp",
        choices=("nearest", "bilinear"),
        default="nearest",
        help="nearest = honest pixel grid (shows 128 blockiness); bilinear = smoothed.",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    scan_id = args.scan_id
    interp = Image.Resampling.NEAREST if args.interp == "nearest" else Image.Resampling.BILINEAR
    fracs = [float(x) for x in args.slice_fracs.split(",") if x.strip()]

    train_dir = resolve_voxel_map_data_root(
        REPO / "runs" / scan_id / "ModelTraining" / "train" / scan_id
    )
    test_dir = REPO / "runs" / scan_id / "ModelTraining" / "test" / scan_id

    native_ct_path, _, _ = _resolve_native_paths(scan_id)
    native_vol = _normalize(load_mha_array(native_ct_path))
    native_size = tuple(int(x) for x in native_vol.shape)

    sub_path = train_dir / "SourceVolumes" / "sub_CT_06_mha.npy"
    if not sub_path.is_file():
        sub_path = test_dir / "SourceVolumes" / "sub_CT_06_mha.npy"
    sub_vol = _normalize(np.load(sub_path).squeeze())
    while sub_vol.ndim > 3:
        sub_vol = sub_vol.squeeze(0)
    sub_size = tuple(int(x) for x in sub_vol.shape)

    sub_win = (float(np.percentile(sub_vol, 1)), float(np.percentile(sub_vol, 99)))
    nat_win = (float(np.percentile(native_vol, 1)), float(np.percentile(native_vol, 99)))

    view = VolumeViewConfig(scan_id="", plane=args.plane).resolve()
    ax = int(view.slice_axis)

    box = args.tile_px
    margin, header_h, colhdr_h, label_w, gap, cap_h = 20, 40, 26, 128, 14, 20
    cols = 2
    rows = len(fracs)
    canvas_w = margin + label_w + cols * box + (cols - 1) * gap + margin
    canvas_h = margin + header_h + colhdr_h + rows * (box + cap_h) + (rows - 1) * gap + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title = f"{scan_id} — {args.plane} slice quality: 128\u00b3 (train) vs native (original)"
    draw.text((margin, margin // 2 + 2), title, fill=(15, 15, 15), font=_font_bold(15))
    sub_str = (
        f"128\u00b3 downsampled {sub_size[0]}\u00d7{sub_size[1]}\u00d7{sub_size[2]}   |   "
        f"native {native_size[0]}\u00d7{native_size[1]}\u00d7{native_size[2]}   |   "
        f"slice axis {AXIS_NAMES[ax]}   |   display: {args.interp}"
    )
    draw.text((margin, margin // 2 + 22), sub_str, fill=(90, 90, 90), font=_font(11))

    x_sub = margin + label_w
    x_nat = margin + label_w + box + gap
    y_colhdr = margin + header_h
    draw.text((x_sub + box // 2 - 40, y_colhdr), "128\u00b3 (train)", fill=(30, 90, 180), font=_font_bold(13))
    draw.text((x_nat + box // 2 - 48, y_colhdr), "native (original)", fill=(30, 130, 60), font=_font_bold(13))

    y0 = y_colhdr + colhdr_h
    for r, frac in enumerate(fracs):
        sub_idx = int(np.clip(frac * sub_size[ax], 0, sub_size[ax] - 1))
        nat_idx = int(np.clip(frac * native_size[ax], 0, native_size[ax] - 1))
        sub_tile, sub_dims, sub_i = render_tile(sub_vol, args.plane, sub_idx, sub_win, box, interp)
        nat_tile, nat_dims, nat_i = render_tile(native_vol, args.plane, nat_idx, nat_win, box, interp)

        y = y0 + r * (box + cap_h + gap)
        draw.text((margin, y + box // 2 - 16), f"slice @ {frac:.2f}", fill=(40, 40, 40), font=_font_bold(12))
        draw.text((margin, y + box // 2 + 2), f"({args.plane})", fill=(110, 110, 110), font=_font(10))

        for x, tile, dims, idx in (
            (x_sub, sub_tile, sub_dims, sub_i),
            (x_nat, nat_tile, nat_dims, nat_i),
        ):
            canvas.paste(Image.fromarray(tile), (x, y))
            draw.rectangle((x - 1, y - 1, x + box, y + box), outline=(200, 200, 200))
            cap = f"idx {idx}  |  {dims[0]}\u00d7{dims[1]} px"
            draw.text((x + 4, y + box + 3), cap, fill=(70, 70, 70), font=_font(10))

    out = args.out or (
        REPO / "results" / scan_id / "plots_nofilm"
        / f"{scan_id}_slice_quality_{args.plane}_{args.interp}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"Saved {out}  ({canvas_w}x{canvas_h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
