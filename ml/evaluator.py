"""Evaluate VoxelMap on ModelTraining layout (TargetProjections / SourceProjections)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    _psnr = None
    _ssim = None

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.dynamic_dataset import resolve_voxel_map_data_root
from ml.training_config import REFERENCE_IM_SIZE
from ml.utilities import losses, networksFiLM, spatialTransform


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


def get_args():
    p = argparse.ArgumentParser(description="VoxelMap Clinical evaluator")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--architecture", default="concatenated")
    p.add_argument("--use_film", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples", type=int, default=0, help="0 = evaluate all")
    return p.parse_args()


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _load_dvf_tensor(dvf_path: Path, device, im_size=128):
    if not dvf_path.exists():
        return torch.zeros((1, 3, im_size, im_size, im_size), device=device)
    dvf = np.load(dvf_path).astype(np.float32)
    if dvf.shape[:3] != (im_size, im_size, im_size):
        from scipy.ndimage import zoom

        zf = [im_size / s for s in dvf.shape[:3]] + [1.0]
        dvf = zoom(dvf, zf, order=1)
    return torch.from_numpy(dvf).permute(3, 0, 1, 2).unsqueeze(0).to(device)


def build_train_index(patient: Path):
    tgt_dir = patient / "TargetProjections"
    src_dir = patient / "SourceProjections"
    dvf_dir = patient / "DVFs"
    angles_path = patient / "Angles.csv"

    angles = None
    if angles_path.exists():
        angles = pd.read_csv(angles_path, header=None).values.squeeze()
        angles = np.atleast_1d(np.asarray(angles, dtype=np.float64)).ravel()

    samples = []
    for tgt_file in sorted(tgt_dir.glob("*_bin.npy")):
        if tgt_file.name.startswith("._"):
            continue
        parts = tgt_file.name.split("_")
        if len(parts) < 3:
            continue
        phase = int(parts[0])
        proj_num_str = parts[2]
        proj_idx = int(proj_num_str)

        src_file = src_dir / f"06_Proj_{proj_num_str}_bin.npy"
        if not src_file.exists():
            continue

        gt_dvf = dvf_dir / f"DVF_{phase:02d}_mha.npy"
        if phase != 6 and not gt_dvf.exists():
            continue

        angle_val = float(angles[proj_idx - 1]) if angles is not None and proj_idx - 1 < len(angles) else 0.0
        samples.append(
            {
                "source_proj": src_file,
                "target_proj": tgt_file,
                "gt_dvf": gt_dvf,
                "phase": phase,
                "proj_idx": proj_idx,
                "angle": angle_val,
            }
        )
    return samples


def generate_trace_plot(res, save_path: Path) -> None:
    angles = np.array(res["angles"])
    order = np.argsort(angles)
    sa = angles[order]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("VoxelMap Evaluation — CE_P1_V_01", fontsize=13, fontweight="bold")

    labels = ["LR displacement (mm)", "SI displacement (mm)", "AP displacement (mm)"]
    keys = ["lr", "si", "ap"]
    for i, ax in enumerate(axes[:3]):
        gt = np.array(res["gt_shifts_mm"][keys[i]])[order]
        pred = np.array(res["shifts_mm"][keys[i]])[order]
        ax.plot(sa, gt, color="black", lw=1.8, label="Ground-truth (GT DVF)")
        ax.plot(sa, pred, "--", color="red", lw=1.2, label="Prediction", alpha=0.85)
        ax.set_ylabel(labels[i])
        ax.set_xlim(0, 360)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    psnr = np.array(res.get("psnr", []), dtype=float)
    if len(psnr) == len(sa):
        axes[3].plot(sa, psnr[order], color="#e6550d", lw=1.4, label="PSNR (dB)")
        axes[3].set_ylabel("PSNR (dB)")
        axes[3].legend(loc="lower right", fontsize=9)

    ssim = np.array(res.get("ssim", []), dtype=float)
    if len(ssim) == len(sa) and np.any(np.isfinite(ssim)):
        axes[4].plot(sa, ssim[order], color="#2c7fb8", lw=1.4, label="SSIM")
        axes[4].set_ylabel("SSIM")
        axes[4].set_ylim(0.0, 1.05)
        axes[4].legend(loc="lower right", fontsize=9)
    else:
        axes[4].text(0.5, 0.5, "SSIM not available", ha="center", va="center", transform=axes[4].transAxes)

    detj = np.array(res.get("det_j_neg_fraction", []), dtype=float)
    if len(detj) == len(sa) and np.any(np.isfinite(detj)):
        axes[5].plot(sa, detj[order], color="#7b3294", lw=1.4, label="Fraction det(J) ≤ 0")
        axes[5].set_ylabel("Neg. det(J) frac.")
        axes[5].legend(loc="upper right", fontsize=9)
    else:
        axes[5].text(0.5, 0.5, "det(J) not available", ha="center", va="center", transform=axes[5].transAxes)

    axes[5].set_xlabel("Gantry Angle (degrees)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = get_args()
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    patient = resolve_voxel_map_data_root(args.data_dir)
    if not (patient / "TargetProjections").is_dir():
        raise SystemExit(f"Invalid data dir (no TargetProjections): {patient}")

    device = torch.device(args.device)
    im_size = REFERENCE_IM_SIZE

    print("--- VoxelMap Clinical Evaluator ---")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data:       {patient}")
    print(f"Device:     {device}")

    model = networksFiLM.Model.load(str(args.checkpoint), str(device))
    model = model.to(device)
    model.eval()
    transformer = spatialTransform.Network([im_size] * 3).to(device).eval()

    index = build_train_index(patient)
    if args.max_samples > 0:
        index = index[: args.max_samples]
    print(f"Samples to evaluate: {len(index)}")
    if not index:
        raise SystemExit("No samples indexed.")

    centroid_op = losses.centroid_ptv()
    dice_op = losses.dice()
    jacobian_op = losses.jacobian_determinant()

    source_ptv_path = next((patient / "Masks").glob("*PTV*_mha.npy"), None)
    source_vol_path = patient / "SourceVolumes" / "sub_CT_06_mha.npy"
    if not source_vol_path.exists():
        source_vol_path = next((patient / "SourceVolumes").glob("sub_CT_*.npy"), None)

    source_abdomen_path = patient / "Masks" / "sub_Abdomen_mha.npy"
    if not source_abdomen_path.exists():
        source_abdomen_path = next((patient / "Masks").glob("*Body*.npy"), None)

    if source_abdomen_path and source_abdomen_path.exists():
        abdomen_mask_np = _normalize(np.load(source_abdomen_path).squeeze())
        while abdomen_mask_np.ndim > 3:
            abdomen_mask_np = abdomen_mask_np.squeeze(0)
        abdomen_mask_np = (abdomen_mask_np > 0).astype(np.float32)
    else:
        abdomen_mask_np = None

    src_vol_np = _normalize(np.load(source_vol_path).squeeze())
    while src_vol_np.ndim > 3:
        src_vol_np = src_vol_np.squeeze(0)
    src_vol_t = torch.from_numpy(src_vol_np[None, None]).to(device)

    if source_ptv_path is not None and source_ptv_path.exists():
        src_ptv_np = _normalize(np.load(source_ptv_path).squeeze())
    elif abdomen_mask_np is not None:
        src_ptv_np = abdomen_mask_np.astype(np.float32)
    else:
        src_ptv_np = (src_vol_np > 0.08).astype(np.float32)

    while src_ptv_np.ndim > 3:
        src_ptv_np = src_ptv_np.squeeze(0)
    src_ptv_t = torch.from_numpy(src_ptv_np[None, None]).to(device)

    results = {
        "angles": [],
        "phases": [],
        "dice": [],
        "mse": [],
        "psnr": [],
        "ssim": [],
        "det_j_neg_fraction": [],
        "shifts_mm": {"lr": [], "si": [], "ap": [], "3d": []},
        "gt_shifts_mm": {"lr": [], "si": [], "ap": []},
    }

    target_size = [im_size] * 3

    def _ensure_size(t):
        if list(t.shape[-3:]) != target_size:
            return torch.nn.functional.interpolate(
                t, size=target_size, mode="trilinear", align_corners=False
            )
        return t

    with torch.no_grad():
        for sample in tqdm(index, desc="Evaluating"):
            src_p_np = _normalize(np.load(sample["source_proj"]))
            tgt_p_np = _normalize(np.load(sample["target_proj"]))
            angle_val = sample["angle"]

            src_p = torch.from_numpy(src_p_np[None, None]).to(device)
            tgt_p = torch.from_numpy(tgt_p_np[None, None]).to(device)
            angle = torch.tensor([angle_val], dtype=torch.float32, device=device)

            if args.use_film:
                _, pred_flow = model(src_p, tgt_p, src_vol_t, angle=angle)
            else:
                _, pred_flow = model(src_p, tgt_p, src_vol_t)

            gt_flow = _load_dvf_tensor(sample["gt_dvf"], device, im_size)

            pred_ptv = _ensure_size(transformer(src_ptv_t, pred_flow))
            gt_ptv = _ensure_size(transformer(src_ptv_t, gt_flow))
            src_ptv_eval = _ensure_size(src_ptv_t)

            slr, ssi, sap = losses.centroid_shift_mm(centroid_op, src_ptv_eval, pred_ptv)
            glr, gsi, gap = losses.centroid_shift_mm(centroid_op, src_ptv_eval, gt_ptv)
            err_3d = float(np.sqrt((slr - glr) ** 2 + (ssi - gsi) ** 2 + (sap - gap) ** 2))
            dice_val = float(dice_op.loss(gt_ptv, pred_ptv).item())

            pred_vol_3d = transformer(src_vol_t, pred_flow)[0, 0].cpu().numpy()
            gt_vol_3d = transformer(src_vol_t, gt_flow)[0, 0].cpu().numpy()

            if abdomen_mask_np is not None:
                mask_bin = abdomen_mask_np > 0
                mse_pred = pred_vol_3d[mask_bin]
                mse_gt = gt_vol_3d[mask_bin]
            else:
                mse_pred = pred_vol_3d.flatten()
                mse_gt = gt_vol_3d.flatten()

            mse_val = float(np.mean((mse_pred - mse_gt) ** 2))

            if HAS_SKIMAGE and _psnr is not None:
                psnr_val = float(_psnr(mse_gt, mse_pred, data_range=1.0))
            else:
                psnr_val = float(10 * np.log10(1.0 / (mse_val + 1e-8)))

            if HAS_SKIMAGE and _ssim is not None:
                try:
                    ssim_val = float(_ssim(gt_vol_3d, pred_vol_3d, data_range=1.0))
                except Exception:
                    ssim_val = float("nan")
            else:
                ssim_val = float("nan")

            try:
                pf = pred_flow.detach().float().cpu().numpy()
                metric_flows = np.squeeze(pf)
                if metric_flows.ndim == 4 and metric_flows.shape[0] == 3:
                    disp = np.stack(
                        [metric_flows[0], metric_flows[1], metric_flows[2]], axis=-1
                    ).astype(np.float64)
                    det_j = jacobian_op.loss(disp)
                    det_j_ratio = float(np.mean(det_j <= 0))
                else:
                    det_j_ratio = float("nan")
            except Exception:
                det_j_ratio = float("nan")

            results["angles"].append(angle_val)
            results["phases"].append(sample["phase"])
            results["dice"].append(dice_val)
            results["mse"].append(mse_val)
            results["psnr"].append(psnr_val)
            results["ssim"].append(ssim_val)
            results["det_j_neg_fraction"].append(det_j_ratio)
            results["shifts_mm"]["lr"].append(float(slr))
            results["shifts_mm"]["si"].append(float(ssi))
            results["shifts_mm"]["ap"].append(float(sap))
            results["shifts_mm"]["3d"].append(err_3d)
            results["gt_shifts_mm"]["lr"].append(float(glr))
            results["gt_shifts_mm"]["si"].append(float(gsi))
            results["gt_shifts_mm"]["ap"].append(float(gap))

    summary = {
        "n_samples": len(results["angles"]),
        "mean_dice": float(np.mean(results["dice"])),
        "mean_3d_error_mm": float(np.mean(results["shifts_mm"]["3d"])),
        "mean_mse": float(np.mean(results["mse"])),
        "mean_psnr_db": float(np.mean(results["psnr"])),
        "mean_ssim": float(np.nanmean(results["ssim"])) if np.any(np.isfinite(results["ssim"])) else None,
        "mean_neg_det_j_fraction": float(np.nanmean(results["det_j_neg_fraction"]))
        if np.any(np.isfinite(results["det_j_neg_fraction"]))
        else None,
    }

    print("\n--- Evaluation Summary ---")
    print(f"  Samples evaluated : {summary['n_samples']}")
    print(f"  Mean Dice         : {summary['mean_dice']:.4f}")
    print(f"  Mean 3D Error     : {summary['mean_3d_error_mm']:.3f} mm")
    print(f"  Mean MSE          : {summary['mean_mse']:.6f}")
    print(f"  Mean PSNR         : {summary['mean_psnr_db']:.2f} dB")
    if summary["mean_ssim"] is not None:
        print(f"  Mean SSIM         : {summary['mean_ssim']:.4f}")
    else:
        print("  Mean SSIM         : N/A (install scikit-image)")

    payload = {"summary": summary, "per_sample": results}
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)

    generate_trace_plot(results, out_path / "evaluation_trace.png")

    # PSNR / SSIM vs angle scatter summaries
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    angles = np.array(results["angles"])
    axes[0].scatter(angles, results["psnr"], s=4, alpha=0.4, c="#e6550d")
    axes[0].set_xlabel("Gantry angle (deg)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title(f"PSNR (mean {summary['mean_psnr_db']:.2f} dB)")
    axes[0].grid(True, alpha=0.3)

    ssim_arr = np.array(results["ssim"], dtype=float)
    if np.any(np.isfinite(ssim_arr)):
        axes[1].scatter(angles, ssim_arr, s=4, alpha=0.4, c="#2c7fb8")
        axes[1].set_ylabel("SSIM")
        axes[1].set_title(f"SSIM (mean {summary['mean_ssim']:.4f})")
    axes[1].set_xlabel("Gantry angle (deg)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "psnr_ssim_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved metrics + plots to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
