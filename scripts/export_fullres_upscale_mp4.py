#!/usr/bin/env python3
"""White-background 5×2 MP4: breathing-sweep No-FiLM DVF + native upscale viz.

Layout:
  [Source proj]              [Target proj]
  [Pred DVF @128 body-mask]  [GT DVF @128 body-mask]
  [sub_CT_06 + size]         [CT_06 native + size]
  [Upsampled pred DVF]       [Upsampled GT DVF]
  [Warped @128 + PTV]        [Warped native + PTV]
"""
from __future__ import annotations

import argparse
import os
import shutil
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
    build_sweep_index,
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


def mask_flow(flow: np.ndarray, body: np.ndarray | None) -> np.ndarray:
    """Zero DVF outside body so outer-boundary / air vectors are hidden."""
    if body is None:
        return flow
    out = flow.copy()
    out[:, body <= 0.5] = 0.0
    return out


def flow_mag_rgb(
    flow: np.ndarray,
    view: VolumeViewConfig,
    vmax: float | None = None,
    body: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    flow = mask_flow(flow, body)
    u, v = extract_flow_uv(flow, view)
    mag = np.sqrt(u * u + v * v).astype(np.float32)
    if body is not None:
        m2 = extract_slice(body.astype(np.float32), view) > 0.5
        mag = np.where(m2, mag, 0.0)
        u = np.where(m2, u, 0.0)
        v = np.where(m2, v, 0.0)
    if vmax is None:
        pos = mag[mag > 0]
        vmax = float(np.percentile(pos, 99)) + 1e-6 if pos.size else 1.0
    rgb = gray_to_rgb_u8(mag, 0.0, vmax)
    rgb = draw_quiver(rgb, u, v, stride=max(6, mag.shape[0] // 16), color=(220, 40, 30))
    return rgb, vmax


def load_sweep_gt_flow(test_dir: Path, phase: int, device: torch.device, im_size: int) -> torch.Tensor:
    if phase == 6 or phase < 0:
        return torch.zeros(1, 3, im_size, im_size, im_size, device=device)
    path = test_dir / "DVFs" / f"DVF_{phase:02d}_mha.npy"
    return _load_dvf_tensor(path, device, im_size=im_size)


def annotate_size_axes(rgb: np.ndarray, shape_dhw: tuple[int, int, int], plane: str) -> np.ndarray:
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    font = _font(11)
    font_s = _font(10)
    d, h, w = shape_dhw
    label = f"{d}×{h}×{w}"
    tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else 8 * len(label)
    draw.rectangle((4, 4, 12 + tw, 22), fill=(255, 255, 255), outline=(80, 80, 80))
    draw.text((8, 6), label, fill=(20, 20, 20), font=font)
    if plane == "sagittal":
        hx, hy = "← LR →", "↑ SI ↓"
    elif plane == "coronal":
        hx, hy = "← LR →", "↑ AP ↓"
    else:
        hx, hy = "← SI →", "↑ AP ↓"
    iw, ih = img.size
    draw.text((iw // 2 - 20, ih - 16), hx, fill=(30, 90, 180), font=font_s)
    tmp = Image.new("RGBA", (ih, 18), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((2, 2), hy, fill=(30, 90, 180, 255), font=font_s)
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
    draw.text((margin, margin // 2 + 2), header, fill=(15, 15, 15), font=_font_bold(13))
    y0 = margin + header_h
    for i, (title, rgb) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = margin + c * (tile_px + gap)
        y = y0 + r * (label_h + tile_px + gap)
        draw.text((x, y), title, fill=(50, 50, 50), font=_font(11))
        tile = Image.fromarray(resize_square(rgb, tile_px))
        draw.rectangle(
            (x - 1, y + label_h - 1, x + tile_px, y + label_h + tile_px),
            outline=(200, 200, 200),
        )
        canvas.paste(tile, (x, y + label_h))
    return np.array(canvas)


def peak_ptv_slice(ptv: np.ndarray, axis: int) -> int:
    counts = [(i, int((np.take(ptv, i, axis=axis) > 0.5).sum())) for i in range(ptv.shape[axis])]
    return max(counts, key=lambda x: x[1])[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-res upscale 5×2 panel MP4 (white bg)")
    ap.add_argument("--scan-id", default="CE_P1_V_01_fdk")
    ap.add_argument("--film", action="store_true", help="Use FiLM (default No-FiLM)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--plane", choices=("axial", "sagittal", "coronal"), default="sagittal")
    ap.add_argument("--slice-index", type=int, default=None)
    ap.add_argument("--tile-px", type=int, default=220)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--source", choices=("sweep", "train"), default="sweep")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all frames")
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
    test_dir = REPO / "runs" / scan_id / "ModelTraining" / "test" / scan_id
    ckpt = (
        REPO / "runs" / scan_id / "checkpoints_nofilm" / "best.pt"
        if use_nofilm
        else REPO / "runs" / scan_id / "checkpoints" / "best.pt"
    )
    tag = "nofilm" if use_nofilm else "film"
    if not ckpt.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt}")

    if args.source == "sweep":
        samples = build_sweep_index(test_dir)
        if args.max_samples > 0:
            samples = samples[: args.max_samples]
        sample_tag = f"sweep{len(samples)}"
        print(f"Samples: {len(samples)} breathing-sweep from {test_dir}")
    else:
        n = args.max_samples if args.max_samples > 0 else 90
        samples = _stratified_sample(build_train_index(train_dir), n)
        samples = sorted(samples, key=lambda s: (int(s["phase"]), float(s["angle"]), int(s["proj_idx"])))
        sample_tag = f"strat{len(samples)}"
        print(f"Samples: {len(samples)} stratified train-pairs from {train_dir}")
    if not samples:
        raise SystemExit("No samples found")

    native_ct_path, native_ptv_path, native_body_path = _resolve_native_paths(scan_id)
    native_vol = _normalize(load_mha_array(native_ct_path))
    native_ptv = (load_mha_array(native_ptv_path) > 0.5).astype(np.float32)
    native_body = (
        (load_mha_array(native_body_path) > 0.5).astype(np.float32) if native_body_path else None
    )
    native_size = tuple(int(x) for x in native_vol.shape)

    sub_path = train_dir / "SourceVolumes" / "sub_CT_06_mha.npy"
    if not sub_path.is_file():
        sub_path = test_dir / "SourceVolumes" / "sub_CT_06_mha.npy"
    sub_vol = _normalize(np.load(sub_path).squeeze())
    while sub_vol.ndim > 3:
        sub_vol = sub_vol.squeeze(0)
    sub_size = tuple(int(x) for x in sub_vol.shape)

    masks_dir = train_dir / "Masks" if (train_dir / "Masks").is_dir() else test_dir / "Masks"
    ptv_128_path = next(masks_dir.glob("*PTV*_mha.npy"), None)
    ptv_128 = (np.load(ptv_128_path).squeeze() > 0.5).astype(np.float32) if ptv_128_path else None
    body_128 = load_body_mask(masks_dir)
    if body_128 is None:
        body_128 = load_body_mask(test_dir / "Masks")

    view_128 = VolumeViewConfig(scan_id=scan_id, plane=args.plane).resolve()
    view_nat = VolumeViewConfig(scan_id=scan_id, plane=args.plane).resolve()
    ax = int(view_128.slice_axis)
    if args.slice_index is not None:
        view_128.slice_index = args.slice_index
    elif ptv_128 is not None:
        view_128.slice_index = peak_ptv_slice(ptv_128, ax)
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

    proj_stack = []
    for s in samples[: min(40, len(samples))]:
        proj_stack += [_normalize(np.load(s["source_proj"])), _normalize(np.load(s["target_proj"]))]
    p_lo, p_hi = float(np.percentile(np.stack(proj_stack), 1)), float(np.percentile(np.stack(proj_stack), 99))
    v128_lo, v128_hi = float(np.percentile(sub_vol, 1)), float(np.percentile(sub_vol, 99))
    vnat_lo, vnat_hi = float(np.percentile(native_vol, 1)), float(np.percentile(native_vol, 99))

    out = args.out or (
        REPO
        / "runs"
        / scan_id
        / "videos"
        / f"{scan_id}_fullres_upscale_{args.plane}{view_128.slice_index}_{sample_tag}_{tag}.mp4"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    mag_vmax_128 = mag_vmax_nat = None
    mode = "sweep" if args.source == "sweep" else "stratified"

    with torch.no_grad():
        for fi, s in enumerate(tqdm(samples, desc=f"{scan_id} {mode} upscale")):
            src_p = _normalize(np.load(s["source_proj"]))
            tgt_p = _normalize(np.load(s["target_proj"]))
            src_pt = torch.from_numpy(src_p[None, None]).float().to(device)
            tgt_pt = torch.from_numpy(tgt_p[None, None]).float().to(device)
            angle = torch.tensor([s["angle"]], dtype=torch.float32, device=device)
            if use_film:
                _, pred_flow = model(src_pt, tgt_pt, sub_t, angle=angle)
            else:
                _, pred_flow = model(src_pt, tgt_pt, sub_t)

            if args.source == "sweep":
                gt_flow = load_sweep_gt_flow(test_dir, int(s["phase"]), device, sub_size[0])
            else:
                gt_flow = _load_dvf_tensor(Path(s["gt_dvf"]), device, im_size=sub_size[0])

            pred_up = upsample_dvf(pred_flow, native_size, align_corners=True)
            gt_up = upsample_dvf(gt_flow, native_size, align_corners=True)
            warped_128 = xf_128(sub_t, pred_flow)[0, 0].cpu().numpy()
            warped_nat = xf_nat(nat_t, pred_up)[0, 0].cpu().numpy()

            pred_np, gt_np = pred_flow[0].cpu().numpy(), gt_flow[0].cpu().numpy()
            pred_up_np, gt_up_np = pred_up[0].cpu().numpy(), gt_up[0].cpu().numpy()

            pred_dvf_rgb, mag_vmax_128 = flow_mag_rgb(pred_np, view_128, mag_vmax_128, body=body_128)
            gt_dvf_rgb, _ = flow_mag_rgb(gt_np, view_128, mag_vmax_128, body=body_128)
            pred_up_rgb, mag_vmax_nat = flow_mag_rgb(pred_up_np, view_nat, mag_vmax_nat, body=native_body)
            gt_up_rgb, _ = flow_mag_rgb(gt_up_np, view_nat, mag_vmax_nat, body=native_body)

            sub_slice = extract_slice(sub_vol, view_128)
            nat_slice = extract_slice(native_vol, view_nat)
            w128_slice = extract_slice(_normalize(warped_128), view_128)
            wnat_slice = extract_slice(_normalize(warped_nat), view_nat)
            if body_128 is not None:
                a, i = int(view_128.slice_axis), view_128.clamp_slice(sub_vol.shape)
                sub_slice = apply_body_mask_slice(sub_slice, body_128, a, i)
                w128_slice = apply_body_mask_slice(w128_slice, body_128, a, i)
            if native_body is not None:
                a, i = int(view_nat.slice_axis), view_nat.clamp_slice(native_vol.shape)
                nat_slice = apply_body_mask_slice(nat_slice, native_body, a, i)
                wnat_slice = apply_body_mask_slice(wnat_slice, native_body, a, i)

            sub_rgb = annotate_size_axes(gray_to_rgb_u8(sub_slice, v128_lo, v128_hi), sub_size, args.plane)
            nat_rgb = annotate_size_axes(gray_to_rgb_u8(nat_slice, vnat_lo, vnat_hi), native_size, args.plane)
            w128_rgb = gray_to_rgb_u8(w128_slice, v128_lo, v128_hi)
            wnat_rgb = gray_to_rgb_u8(wnat_slice, vnat_lo, vnat_hi)
            if ptv_128_t is not None:
                wptv = xf_128(ptv_128_t, pred_flow)[0, 0].cpu().numpy()
                w128_rgb = overlay_ptv_outline_rgb(w128_rgb, extract_slice(wptv, view_128))
            wptv_n = xf_nat(ptv_nat_t, pred_up)[0, 0].cpu().numpy()
            wnat_rgb = overlay_ptv_outline_rgb(wnat_rgb, extract_slice(wptv_n, view_nat))
            w128_rgb = annotate_size_axes(w128_rgb, sub_size, args.plane)
            wnat_rgb = annotate_size_axes(wnat_rgb, native_size, args.plane)

            tiles = [
                ("Source projection (phase 06)", gray_to_rgb_u8(src_p, p_lo, p_hi)),
                ("Target projection", gray_to_rgb_u8(tgt_p, p_lo, p_hi)),
                ("Pred DVF @128³ (body-masked)", pred_dvf_rgb),
                ("GT DVF @128³ (body-masked)", gt_dvf_rgb),
                (f"sub_CT_06  {sub_size[0]}×{sub_size[1]}×{sub_size[2]}", sub_rgb),
                (f"CT_06 native  {native_size[0]}×{native_size[1]}×{native_size[2]}", nat_rgb),
                ("Upsampled pred DVF (body-masked)", pred_up_rgb),
                ("Upsampled GT DVF (body-masked)", gt_up_rgb),
                ("Warped sub_CT @128³ + PTV", w128_rgb),
                ("Warped CT_06 native + PTV", wnat_rgb),
            ]
            header = (
                f"{scan_id} | {tag} | {mode} {fi + 1}/{len(samples)} | "
                f"proj {int(s['proj_idx']):03d} phase {int(s['phase']):02d} ∠{float(s['angle']):.1f}° | "
                f"{args.plane} 128={view_128.slice_index} nat={view_nat.slice_index} | "
                f"upsample=trilinear+scale | DVF body-masked"
            )
            frames.append(compose_grid(tiles, header, cols=2, tile_px=args.tile_px))

    import imageio

    h0, w0 = frames[0].shape[:2]
    pad_h, pad_w = (16 - h0 % 16) % 16, (16 - w0 % 16) % 16
    if pad_h or pad_w:
        frames = [
            np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=255)
            for f in frames
        ]
    imageio.mimsave(str(out), frames, fps=args.fps, codec="libx264")
    if use_nofilm:
        pub = REPO / "results" / scan_id / "videos_nofilm" / out.name
        pub.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, pub)
        print(f"Copied -> {pub}")
    print(f"Wrote {len(frames)} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
