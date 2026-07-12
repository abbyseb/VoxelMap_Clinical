#!/usr/bin/env python3
"""Phase-1 full-res eval: predict DVF @128, upsample+scale, warp native volume/PTV.

Compares metrics at training resolution (128³) vs native grid on the *same* samples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.dynamic_dataset import resolve_voxel_map_data_root
from ml.evaluator import NumpyEncoder, _load_dvf_tensor, _normalize, build_train_index
from ml.flow_utils import load_mha_array, spacing_from_grids, upsample_dvf
from ml.mask_utils import masked_volume_metrics
from ml.training_config import REFERENCE_IM_SIZE
from ml.utilities import losses, networksFiLM, spatialTransform


def _stratified_sample(index: list[dict], max_samples: int) -> list[dict]:
    if max_samples <= 0 or max_samples >= len(index):
        return index
    by_phase: dict[int, list[dict]] = {}
    for s in index:
        by_phase.setdefault(int(s["phase"]), []).append(s)
    phases = sorted(by_phase)
    per = max(1, max_samples // max(len(phases), 1))
    out: list[dict] = []
    for ph in phases:
        rows = by_phase[ph]
        step = max(1, len(rows) // per)
        out.extend(rows[::step][:per])
    return out[:max_samples]


def _resolve_native_paths(scan_id: str) -> tuple[Path, Path, Path | None]:
    """Return (native_ct_06, native_ptv, native_body_or_None).

    For ``*_fdk`` scans, prefer the FDK run/staged native CT and matching masks.
    """
    pnum = scan_id.split("_P")[1].split("_")[0]
    source_id = scan_id[: -len("_fdk")] if scan_id.endswith("_fdk") else scan_id
    run_ct = REPO / "runs" / scan_id / scan_id / "train" / "CT_06.mha"
    staged_fdk = REPO / "data" / "staged_fdk" / f"P{pnum}" / scan_id
    staged = REPO / "data" / "staged" / f"P{pnum}" / source_id

    ct_candidates = [
        run_ct,
        staged_fdk / "GTVol_06.mha",
        staged / "GTVol_06.mha",
    ]
    ct = next((p for p in ct_candidates if p.is_file()), None)
    mask_bases = [staged_fdk, staged]
    ptv = next((b / "Mask_PTV.mha" for b in mask_bases if (b / "Mask_PTV.mha").is_file()), None)
    body = next((b / "Mask_Body.mha" for b in mask_bases if (b / "Mask_Body.mha").is_file()), None)
    if ct is None:
        raise SystemExit(f"Native CT not found for {scan_id}: tried {ct_candidates}")
    if ptv is None:
        raise SystemExit(f"Native PTV mask not found for {scan_id}")
    return ct, ptv, body


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-res DVF upsample eval (Phase 1)")
    ap.add_argument("--scan-id", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=90, help="0 = all samples")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--no-film",
        action="store_true",
        help="Evaluate checkpoints_nofilm/best.pt → eval_fullres_nofilm/",
    )
    args = ap.parse_args()

    scan_id = args.scan_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.no_film:
        ckpt = REPO / "runs" / scan_id / "checkpoints_nofilm" / "best.pt"
        if not ckpt.is_file():
            ckpt = REPO / "runs" / scan_id / "checkpoints_nofilm" / f"{scan_id}_concat_nofilm.pt"
        out_dir = REPO / "runs" / scan_id / "eval_fullres_nofilm"
        tag = "no-FiLM"
    else:
        ckpt = REPO / "runs" / scan_id / "checkpoints" / "best.pt"
        if not ckpt.is_file():
            ckpt = REPO / "runs" / scan_id / "checkpoints" / f"{scan_id}_concat_film.pt"
        out_dir = REPO / "runs" / scan_id / "eval_fullres"
        tag = "FiLM"
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    data = resolve_voxel_map_data_root(REPO / "runs" / scan_id / "ModelTraining" / "train" / scan_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    native_ct_path, native_ptv_path, native_body_path = _resolve_native_paths(scan_id)
    native_vol = _normalize(load_mha_array(native_ct_path))
    native_ptv = (load_mha_array(native_ptv_path) > 0.5).astype(np.float32)
    native_body = None
    if native_body_path is not None:
        native_body = (load_mha_array(native_body_path) > 0.5).astype(np.float32)
    native_size = tuple(int(x) for x in native_vol.shape)
    train_spacing = spacing_from_grids(native_size)
    native_spacing = (1.0, 1.0, 1.0)

    print("--- Full-res DVF upsample eval ---")
    print(f"Scan:       {scan_id}  ({tag})")
    print(f"Checkpoint: {ckpt}")
    print(f"Data:       {data}")
    print(f"Native CT:  {native_ct_path}  shape={native_size}")
    print(f"Device:     {device}")
    print(f"Spacing @128 (mm/vox): {tuple(round(x, 4) for x in train_spacing)}")

    im_size = REFERENCE_IM_SIZE
    model = networksFiLM.Model.load(str(ckpt), str(device)).to(device).eval()
    use_film = bool(getattr(model, "use_film", False))
    xf_128 = spatialTransform.Network([im_size] * 3).to(device).eval()
    xf_nat = spatialTransform.Network(list(native_size)).to(device).eval()

    index = _stratified_sample(build_train_index(data), args.max_samples)
    print(f"Samples:    {len(index)} | model.use_film={use_film}")

    # 128³ source tensors
    src_vol_path = data / "SourceVolumes" / "sub_CT_06_mha.npy"
    src_vol_np = _normalize(np.load(src_vol_path).squeeze())
    while src_vol_np.ndim > 3:
        src_vol_np = src_vol_np.squeeze(0)
    src_vol_t = torch.from_numpy(src_vol_np[None, None]).to(device)

    ptv_path = next((data / "Masks").glob("*PTV*_mha.npy"), None)
    src_ptv_np = _normalize(np.load(ptv_path).squeeze()) if ptv_path else (src_vol_np > 0.08).astype(np.float32)
    while src_ptv_np.ndim > 3:
        src_ptv_np = src_ptv_np.squeeze(0)
    src_ptv_t = torch.from_numpy(src_ptv_np[None, None]).to(device)

    body_128 = None
    body_npy = data / "Masks" / "Mask_Body_mha.npy"
    if body_npy.is_file():
        body_128 = (np.load(body_npy).squeeze() > 0.5).astype(np.float32)

    nat_vol_t = torch.from_numpy(native_vol[None, None]).to(device)
    nat_ptv_t = torch.from_numpy(native_ptv[None, None]).to(device)

    centroid_op = losses.centroid_ptv()
    dice_op = losses.dice()

    buckets = {
        "128": {"dice": [], "err3d": [], "psnr": [], "ssim": [], "mse": [], "gt_shift_3d": []},
        "fullres": {"dice": [], "err3d": [], "psnr": [], "ssim": [], "mse": [], "gt_shift_3d": []},
    }

    with torch.no_grad():
        for sample in tqdm(index, desc=f"{scan_id} fullres"):
            src_p = torch.from_numpy(_normalize(np.load(sample["source_proj"]))[None, None]).to(device)
            tgt_p = torch.from_numpy(_normalize(np.load(sample["target_proj"]))[None, None]).to(device)
            angle = torch.tensor([sample["angle"]], dtype=torch.float32, device=device)

            if use_film:
                _, pred_flow = model(src_p, tgt_p, src_vol_t, angle=angle)
            else:
                _, pred_flow = model(src_p, tgt_p, src_vol_t)
            gt_flow = _load_dvf_tensor(sample["gt_dvf"], device, im_size)

            # --- 128³ metrics ---
            pred_ptv_128 = xf_128(src_ptv_t, pred_flow)
            gt_ptv_128 = xf_128(src_ptv_t, gt_flow)
            slr, ssi, sap = losses.centroid_shift_mm(centroid_op, src_ptv_t, pred_ptv_128, spacing=train_spacing)
            glr, gsi, gap = losses.centroid_shift_mm(centroid_op, src_ptv_t, gt_ptv_128, spacing=train_spacing)
            err_128 = float(np.sqrt((slr - glr) ** 2 + (ssi - gsi) ** 2 + (sap - gap) ** 2))
            dice_128 = float(dice_op.loss(gt_ptv_128, pred_ptv_128).item())
            pred_v_128 = xf_128(src_vol_t, pred_flow)[0, 0].cpu().numpy()
            gt_v_128 = xf_128(src_vol_t, gt_flow)[0, 0].cpu().numpy()
            mse_128, psnr_128, ssim_128 = masked_volume_metrics(gt_v_128, pred_v_128, body_128)
            gt_shift_128 = float(np.sqrt(glr**2 + gsi**2 + gap**2))
            buckets["128"]["dice"].append(dice_128)
            buckets["128"]["err3d"].append(err_128)
            buckets["128"]["psnr"].append(psnr_128)
            buckets["128"]["ssim"].append(ssim_128)
            buckets["128"]["mse"].append(mse_128)
            buckets["128"]["gt_shift_3d"].append(gt_shift_128)

            # --- full-res: upsample DVF then warp native ---
            pred_up = upsample_dvf(pred_flow, native_size, align_corners=True)
            gt_up = upsample_dvf(gt_flow, native_size, align_corners=True)
            pred_ptv_n = xf_nat(nat_ptv_t, pred_up)
            gt_ptv_n = xf_nat(nat_ptv_t, gt_up)
            slr, ssi, sap = losses.centroid_shift_mm(centroid_op, nat_ptv_t, pred_ptv_n, spacing=native_spacing)
            glr, gsi, gap = losses.centroid_shift_mm(centroid_op, nat_ptv_t, gt_ptv_n, spacing=native_spacing)
            err_n = float(np.sqrt((slr - glr) ** 2 + (ssi - gsi) ** 2 + (sap - gap) ** 2))
            dice_n = float(dice_op.loss(gt_ptv_n, pred_ptv_n).item())
            pred_v_n = xf_nat(nat_vol_t, pred_up)[0, 0].cpu().numpy()
            gt_v_n = xf_nat(nat_vol_t, gt_up)[0, 0].cpu().numpy()
            mse_n, psnr_n, ssim_n = masked_volume_metrics(gt_v_n, pred_v_n, native_body)
            gt_shift_n = float(np.sqrt(glr**2 + gsi**2 + gap**2))
            buckets["fullres"]["dice"].append(dice_n)
            buckets["fullres"]["err3d"].append(err_n)
            buckets["fullres"]["psnr"].append(psnr_n)
            buckets["fullres"]["ssim"].append(ssim_n)
            buckets["fullres"]["mse"].append(mse_n)
            buckets["fullres"]["gt_shift_3d"].append(gt_shift_n)

    def _mean(xs):
        return float(np.nanmean(np.asarray(xs, dtype=float)))

    summary = {
        "scan_id": scan_id,
        "variant": tag,
        "use_film": use_film,
        "n_samples": len(index),
        "native_size": list(native_size),
        "train_spacing_mm": list(train_spacing),
        "checkpoint": str(ckpt),
        "native_ct": str(native_ct_path),
        "at_128": {
            "mean_dice": _mean(buckets["128"]["dice"]),
            "mean_3d_error_mm": _mean(buckets["128"]["err3d"]),
            "mean_psnr_db": _mean(buckets["128"]["psnr"]),
            "mean_ssim": _mean(buckets["128"]["ssim"]),
            "mean_mse": _mean(buckets["128"]["mse"]),
        },
        "at_fullres": {
            "mean_dice": _mean(buckets["fullres"]["dice"]),
            "mean_3d_error_mm": _mean(buckets["fullres"]["err3d"]),
            "mean_psnr_db": _mean(buckets["fullres"]["psnr"]),
            "mean_ssim": _mean(buckets["fullres"]["ssim"]),
            "mean_mse": _mean(buckets["fullres"]["mse"]),
        },
        "delta_dice_fullres_minus_128": _mean(buckets["fullres"]["dice"]) - _mean(buckets["128"]["dice"]),
        "delta_3d_error_mm_fullres_minus_128": _mean(buckets["fullres"]["err3d"]) - _mean(buckets["128"]["err3d"]),
        "gt_shift_consistency_mae_mm": float(
            np.mean(
                np.abs(
                    np.asarray(buckets["fullres"]["gt_shift_3d"]) - np.asarray(buckets["128"]["gt_shift_3d"])
                )
            )
        ),
    }

    print("\n--- Summary ---")
    print(f"  Samples              : {summary['n_samples']}")
    print(f"  Dice @128            : {summary['at_128']['mean_dice']:.4f}")
    print(f"  Dice @fullres        : {summary['at_fullres']['mean_dice']:.4f}  (Δ {summary['delta_dice_fullres_minus_128']:+.4f})")
    print(f"  3D err @128 (mm)     : {summary['at_128']['mean_3d_error_mm']:.3f}")
    print(f"  3D err @fullres (mm) : {summary['at_fullres']['mean_3d_error_mm']:.3f}  (Δ {summary['delta_3d_error_mm_fullres_minus_128']:+.3f})")
    print(f"  PSNR @128 / fullres  : {summary['at_128']['mean_psnr_db']:.2f} / {summary['at_fullres']['mean_psnr_db']:.2f} dB")
    print(f"  SSIM @128 / fullres  : {summary['at_128']['mean_ssim']:.4f} / {summary['at_fullres']['mean_ssim']:.4f}")
    print(f"  GT shift MAE 128↔nat : {summary['gt_shift_consistency_mae_mm']:.3f} mm (scale sanity; expect small)")

    out_json = out_dir / "fullres_vs_128_metrics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_sample": buckets}, f, indent=2, cls=NumpyEncoder)
    print(f"Saved {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
