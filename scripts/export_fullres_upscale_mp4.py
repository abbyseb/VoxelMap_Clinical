#!/usr/bin/env python3
"""White-background MP4: 128 vs native DVF-upscale visualization.

Layout (5×2):
  [Source proj]           [Target proj]
  [Pred DVF @128]         [GT DVF @128]
  [sub_CT_06 128³]        [CT_06 native + size axes]
  [Upsampled pred DVF]    [Upsampled GT DVF]
  [Warped @128]           [Warped @native]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.dynamic_dataset import resolve_voxel_map_data_root
from ml.evaluator import _load_dvf_tensor, build_train_index
from ml.flow_utils import load_mha_array, upsample_dvf
from ml.mask_utils import apply_body_mask_slice, load_body_mask
from ml.utilities import networksFiLM, spatialTransform
from ml.volume_view import VolumeViewConfig, extract_flow_uv, extract_slice
from scripts.export_dvf_warp_mp4 import (
    draw_quiver,
    gray_to_rgb_u8,
    overlay_ptv_outline_rgb,
    resize_square,
)
from scripts.run_fullres_dvf_eval import _resolve_native_paths, _stratified_sample


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


def flow_mag_rgb(flow: np.ndarray, view: VolumeViewConfig, vmax: float | None = None) -> tuple[np.ndarray, float]:
    """Magnitude of in-plane flow as grayscale RGB; returns (rgb, vmax_used)."""
    u, v = extract_flow_uv(flow, view)
    mag = np.sqrt(u * u + v * v).astype(np.float32)
    if vmax is None:
        vmax = float(np.percentile(mag, 99)) + 1e-6
    rgb = gray_to_rgb_u8(mag, 0.0, vmax)
    rgb = draw_quiver(rgb, u, v, stride=max(6, mag.shape[0] // 16), color=(220, 40, 30))
    return rgb, vmax


def annotate_size_axes(rgb: np.ndarray, shape_dhw: tuple[int, int, int], plane: str) -> np.ndarray:
    """Draw size annotation + H/V axis ticks on a white-framed tile."""
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    font = _font(11)
    font_s = _font(10)
    d, h, w = shape_dhw
    label = f"{d}×{h}×{w}"
    # top-left size badge
    tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else 8 * len(label)
    draw.rectangle((4, 4, 12 + tw, 22), fill=(255, 255, 255), outline=(80, 80, 80))
    draw.text((8, 6), label, fill=(20, 20, 20), font=font)

    # axis cues (image coords: +x right, +y down)
    if plane == "sagittal":
        hx, hy = "← LR →", "↑ SI ↓"
    elif plane == "coronal":
        hx, hy = "← LR →", "↑ AP ↓"
    else:
        hx, hy = "← SI →", "↑ AP ↓"
    iw, ih = img.size
    draw.text((iw // 2 - 20, ih - 16), hx, fill=(30, 90, 180), font=font_s)
    # vertical label along left
    tmp = Image.new("RGBA", (ih, 18), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    td.text((2, 2), hy, fill=(30, 90, 180, 255), font=font_s)
    tmp = tmp.rotate(90, expand=True)
    img.paste(tmp, (2, ih // 2 - tmp.size[1] // 2), tmp)
    return np.array(img.convert("RGB"))


def compose_grid(
    tiles: list[tuple[str, np.ndarray]],
    header: str,
    cols: int,
    tile_px: int,
    margin: int = 18,
    header_h: int = 28,
    label_h: int = 18,
    gap: int = 10,
) -> np.ndarray:
    rows = (len(tiles) + cols - 1) // cols
    canvas_w = margin * 2 + cols * tile_px + (cols - 1) * gap
    canvas_h = margin + header_h + rows * (label_h + tile_px) + (rows - 1) * gap + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_h = _font_bold(13)
    font_l = _font(11)
    draw.text((margin, margin // 2 + 2), header, fill=(15, 15, 15), font=font_h)

    y0 = margin + header_h
    for i, (title, rgb) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = margin + c * (tile_px + gap)
        y = y0 + r * (label_h + tile_px + gap)
        draw.text((x, y), title, fill=(50, 50, 50), font=font_l)
        tile = Image.fromarray(resize_square(rgb, tile_px))
        draw.rectangle(
            (x - 1, y + label_h - 1, x + tile_px, y + label_h + tile_px),
            outline=(200, 200, 200),
        )
        canvas.paste(tile, (x, y + label_h))
    return np.array(canvas)


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-res upscale panel MP4 (white bg)")
    ap.add_argument("--scan-id", default="CE_P1_V_01_fdk")
    ap.add_argument("--no-film", action="store_true", default=True)
    ap.add_argument("--film", action="store_true", help="Use FiLM checkpoint instead")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--plane", choices=("axial", "sagittal", "coronal"), default="sagittal")
    ap.add_argument("--slice-index", type=int, default=None, help="Default: peak-PTV slice")
    ap.add_argument("--tile-px", type=int, default=220)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=90,
        help="Stratified train-pair samples (same protocol as full-res eval). 0 = all.",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    use_nofilm = not args.film

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scan_id = args.scan_id
    train_dir = resolve_voxel_map_data_root(
        REPO / "runs" / scan_id / "ModelTraining" / "train" / scan_id
    )
    if use_nofilm:
        ckpt = REPO / "runs" / scan_id / "checkpoints_nofilm" / "best.pt"
        tag = "nofilm"
    else:
        ckpt = REPO / "runs" / scan_id / "checkpoints" / "best.pt"
        tag = "film"
    if not ckpt.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt}")

    # Same stratified train-pair set as run_fullres_dvf_eval.py
    samples = _stratified_sample(build_train_index(train_dir), args.max_samples)
    samples = sorted(samples, key=lambda s: (int(s["phase"]), float(s["angle"]), int(s["proj_idx"])))
    if not samples:
        raise SystemExit(f"No train-pair samples in {train_dir}")
    print(f"Samples: {len(samples)} stratified train-pairs from {train_dir}")

    native_ct_path, native_ptv_path, native_body_path = _resolve_native_paths(scan_id)
    native_vol = _normalize(load_mha_array(native_ct_path))
    native_ptv = (load_mha_array(native_ptv_path) > 0.5).astype(np.float32)
    native_body = (
        (load_mha_array(native_body_path) > 0.5).astype(np.float32) if native_body_path else None
    )
    native_size = tuple(int(x) for x in native_vol.shape)

    sub_vol = _normalize(np.load(train_dir / "SourceVolumes" / "sub_CT_06_mha.npy").squeeze())
    while sub_vol.ndim > 3:
        sub_vol = sub_vol.squeeze(0)
    sub_size = tuple(int(x) for x in sub_vol.shape)

    ptv_128_path = next((train_dir / "Masks").glob("*PTV*_mha.npy"), None)
    ptv_128 = (
        (np.load(ptv_128_path).squeeze() > 0.5).astype(np.float32) if ptv_128_path is not None else None
    )
    body_128 = load_body_mask(train_dir / "Masks")

    # View configs — same relative slice for both grids
    view_128 = VolumeViewConfig(scan_id=scan_id, plane=args.plane).resolve()
    view_nat = VolumeViewConfig(scan_id=scan_id, plane=args.plane).resolve()
    if args.slice_index is not None:
        view_128.slice_index = args.slice_index
        # map 128-index → native along same axis
        ax = int(view_128.slice_axis)
        frac = (args.slice_index + 0.5) / sub_size[ax]
        view_nat.slice_index = int(np.clip(frac * native_size[ax], 0, native_size[ax] - 1))
    else:
        # peak PTV on 128, map to native
        if ptv_128 is not None:
            ax = int(view_128.slice_axis)
            counts = [(i, int((np.take(ptv_128, i, axis=ax) > 0.5).sum())) for i in range(ptv_128.shape[ax])]
            view_128.slice_index = max(counts, key=lambda x: x[1])[0]
        ax = int(view_128.slice_axis)
        frac = (view_128.slice_index + 0.5) / sub_size[ax]
        view_nat.slice_index = int(np.clip(frac * native_size[ax], 0, native_size[ax] - 1))

    model = networksFiLM.Model.load(str(ckpt), str(device)).to(device).eval()
    use_film = bool(getattr(model, "use_film", False))
    xf_128 = spatialTransform.Network(list(sub_size)).to(device).eval()
    xf_nat = spatialTransform.Network(list(native_size)).to(device).eval()

    sub_t = torch.from_numpy(sub_vol[None, None]).to(device)
    nat_t = torch.from_numpy(native_vol[None, None]).to(device)
    ptv_128_t = torch.from_numpy(ptv_128[None, None]).to(device) if ptv_128 is not None else None
    ptv_nat_t = torch.from_numpy(native_ptv[None, None]).to(device)

    # display ranges
    proj_stack = []
    for s in samples[: min(40, len(samples))]:
        proj_stack.append(_normalize(np.load(s["source_proj"])))
        proj_stack.append(_normalize(np.load(s["target_proj"])))
    p_lo, p_hi = float(np.percentile(np.stack(proj_stack), 1)), float(np.percentile(np.stack(proj_stack), 99))
    v128_lo, v128_hi = float(np.percentile(sub_vol, 1)), float(np.percentile(sub_vol, 99))
    vnat_lo, vnat_hi = float(np.percentile(native_vol, 1)), float(np.percentile(native_vol, 99))

    out = args.out or (
        REPO
        / "runs"
        / scan_id
        / "videos"
        / f"{scan_id}_fullres_upscale_{args.plane}{view_128.slice_index}_strat{len(samples)}_{tag}.mp4"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    mag_vmax_128 = None
    mag_vmax_nat = None

    with torch.no_grad():
        for fi, s in enumerate(tqdm(samples, desc=f"{scan_id} upscale viz")):
            src_p = _normalize(np.load(s["source_proj"]))
            tgt_p = _normalize(np.load(s["target_proj"]))
            src_pt = torch.from_numpy(src_p[None, None]).float().to(device)
            tgt_pt = torch.from_numpy(tgt_p[None, None]).float().to(device)
            angle = torch.tensor([s["angle"]], dtype=torch.float32, device=device)

            if use_film:
                _, pred_flow = model(src_pt, tgt_pt, sub_t, angle=angle)
            else:
                _, pred_flow = model(src_pt, tgt_pt, sub_t)
            gt_flow = _load_dvf_tensor(Path(s["gt_dvf"]), device, im_size=sub_size[0])

            pred_up = upsample_dvf(pred_flow, native_size, align_corners=True)
            gt_up = upsample_dvf(gt_flow, native_size, align_corners=True)

            warped_128 = xf_128(sub_t, pred_flow)[0, 0].cpu().numpy()
            warped_nat = xf_nat(nat_t, pred_up)[0, 0].cpu().numpy()

            pred_np = pred_flow[0].cpu().numpy()
            gt_np = gt_flow[0].cpu().numpy()
            pred_up_np = pred_up[0].cpu().numpy()
            gt_up_np = gt_up[0].cpu().numpy()

            pred_dvf_rgb, mag_vmax_128 = flow_mag_rgb(pred_np, view_128, mag_vmax_128)
            gt_dvf_rgb, _ = flow_mag_rgb(gt_np, view_128, mag_vmax_128)
            pred_up_rgb, mag_vmax_nat = flow_mag_rgb(pred_up_np, view_nat, mag_vmax_nat)
            gt_up_rgb, _ = flow_mag_rgb(gt_up_np, view_nat, mag_vmax_nat)

            sub_slice = extract_slice(sub_vol, view_128)
            nat_slice = extract_slice(native_vol, view_nat)
            w128_slice = extract_slice(_normalize(warped_128), view_128)
            wnat_slice = extract_slice(_normalize(warped_nat), view_nat)

            if body_128 is not None:
                ax, idx = int(view_128.slice_axis), view_128.clamp_slice(sub_vol.shape)
                sub_slice = apply_body_mask_slice(sub_slice, body_128, ax, idx)
                w128_slice = apply_body_mask_slice(w128_slice, body_128, ax, idx)
            if native_body is not None:
                ax, idx = int(view_nat.slice_axis), view_nat.clamp_slice(native_vol.shape)
                nat_slice = apply_body_mask_slice(nat_slice, native_body, ax, idx)
                wnat_slice = apply_body_mask_slice(wnat_slice, native_body, ax, idx)

            sub_rgb = annotate_size_axes(
                gray_to_rgb_u8(sub_slice, v128_lo, v128_hi), sub_size, args.plane
            )
            nat_rgb = annotate_size_axes(
                gray_to_rgb_u8(nat_slice, vnat_lo, vnat_hi), native_size, args.plane
            )
            w128_rgb = gray_to_rgb_u8(w128_slice, v128_lo, v128_hi)
            wnat_rgb = gray_to_rgb_u8(wnat_slice, vnat_lo, vnat_hi)

            if ptv_128_t is not None:
                warped_ptv_128 = xf_128(ptv_128_t, pred_flow)[0, 0].cpu().numpy()
                w128_rgb = overlay_ptv_outline_rgb(w128_rgb, extract_slice(warped_ptv_128, view_128))
            warped_ptv_nat = xf_nat(ptv_nat_t, pred_up)[0, 0].cpu().numpy()
            wnat_rgb = overlay_ptv_outline_rgb(wnat_rgb, extract_slice(warped_ptv_nat, view_nat))
            w128_rgb = annotate_size_axes(w128_rgb, sub_size, args.plane)
            wnat_rgb = annotate_size_axes(wnat_rgb, native_size, args.plane)

            tiles = [
                ("Source projection (phase 06)", gray_to_rgb_u8(src_p, p_lo, p_hi)),
                ("Target projection", gray_to_rgb_u8(tgt_p, p_lo, p_hi)),
                ("Pred DVF @128³ (mag + arrows)", pred_dvf_rgb),
                ("GT DVF @128³ (mag + arrows)", gt_dvf_rgb),
                (f"sub_CT_06  {sub_size[0]}×{sub_size[1]}×{sub_size[2]}", sub_rgb),
                (f"CT_06 native  {native_size[0]}×{native_size[1]}×{native_size[2]}", nat_rgb),
                ("Upsampled pred DVF (native grid)", pred_up_rgb),
                ("Upsampled GT DVF (native grid)", gt_up_rgb),
                ("Warped sub_CT @128³ + PTV", w128_rgb),
                ("Warped CT_06 native + PTV", wnat_rgb),
            ]
            header = (
                f"{scan_id}  |  {tag}  |  stratified {fi + 1}/{len(samples)}  "
                f"proj {int(s['proj_idx']):03d}  phase {int(s['phase']):02d}  "
                f"∠{float(s['angle']):.1f}°  |  {args.plane} 128-slice={view_128.slice_index}  "
                f"native-slice={view_nat.slice_index}  |  upsample = trilinear + scale"
            )
            frames.append(compose_grid(tiles, header, cols=2, tile_px=args.tile_px))

    import imageio

    h0, w0 = frames[0].shape[:2]
    pad_h = (16 - h0 % 16) % 16
    pad_w = (16 - w0 % 16) % 16
    if pad_h or pad_w:
        frames = [
            np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=255)
            for f in frames
        ]
    imageio.mimsave(str(out), frames, fps=args.fps, codec="libx264")
    # publish copy
    pub = REPO / "results" / scan_id / "videos_nofilm" / out.name
    if use_nofilm:
        pub.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(out, pub)
        print(f"Copied -> {pub}")
    print(f"Wrote {len(frames)} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
