"""Breathing-sweep evaluation with body-masked PSNR/SSIM."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.evaluator import NumpyEncoder, _load_dvf_tensor, _normalize
from ml.mask_utils import load_body_mask, masked_volume_metrics
from ml.training_config import REFERENCE_IM_SIZE
from ml.utilities import losses, networksFiLM, spatialTransform


def build_sweep_index(patient: Path) -> list[dict]:
    test_proj_dir = patient / "TestProjections"
    src_proj_dir = patient / "SourceTestProjections"
    if not src_proj_dir.is_dir():
        src_proj_dir = patient / "SourceProjections"
    dvf_dir = patient / "DVFs"

    angles = None
    ang_path = patient / "Angles.csv"
    if ang_path.is_file():
        angles = np.atleast_1d(pd.read_csv(ang_path, header=None).values.squeeze()).ravel()

    resp_bins = None
    resp_path = patient / "RespBin.csv"
    if resp_path.is_file():
        resp_bins = np.atleast_1d(pd.read_csv(resp_path, header=None).values.squeeze()).ravel()

    samples = []
    for tgt_file in sorted(
        test_proj_dir.glob("Proj_*_bin.npy"),
        key=lambda p: int(re.search(r"Proj_(\d+)", p.name).group(1)),
    ):
        m = re.search(r"Proj_(\d+)", tgt_file.name)
        if not m:
            continue
        proj_idx = int(m.group(1))
        phase = int(resp_bins[proj_idx - 1]) if resp_bins is not None and proj_idx - 1 < len(resp_bins) else -1
        if phase <= 0:
            continue

        src_file = src_proj_dir / f"06_Proj_{proj_idx:03d}_bin.npy"
        if not src_file.is_file():
            cands = list(src_proj_dir.glob(f"06_Proj_{proj_idx:03d}*.npy"))
            src_file = cands[0] if cands else None
        if src_file is None:
            continue

        gt_dvf = dvf_dir / f"DVF_{phase:02d}_mha.npy"
        if phase != 6 and not gt_dvf.is_file():
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


def generate_sweep_trace(res: dict, save_path: Path, title: str) -> None:
    n = len(res["angles"])
    idx = np.arange(1, n + 1)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, (key, label) in enumerate(zip(["lr", "si", "ap"], ["LR (mm)", "SI (mm)", "AP (mm)"])):
        axes[i].plot(idx, res["gt_shifts_mm"][key], color="black", lw=1.5, label="GT")
        axes[i].plot(idx, res["shifts_mm"][key], "--", color="red", lw=1.2, label="Pred")
        axes[i].set_ylabel(label)
        if i == 0:
            axes[i].legend(loc="upper right", fontsize=9)

    axes[3].plot(idx, res["psnr"], color="#e6550d", lw=1.2)
    axes[3].set_ylabel("PSNR (dB, body mask)")

    ssim = np.array(res["ssim"], dtype=float)
    axes[4].plot(idx, ssim, color="#2c7fb8", lw=1.2)
    axes[4].set_ylabel("SSIM (body mask)")
    axes[4].set_xlabel("Projection index (sweep order)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Body-masked breathing sweep evaluation")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--scan-id", default="")
    ap.add_argument("--use_film", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    patient = Path(args.data_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    im_size = REFERENCE_IM_SIZE

    model = networksFiLM.Model.load(str(args.checkpoint), str(device)).to(device).eval()
    transformer = spatialTransform.Network([im_size] * 3).to(device).eval()

    index = build_sweep_index(patient)
    if not index:
        raise SystemExit(f"No sweep samples in {patient}")
    print(f"Sweep samples: {len(index)}")

    body_mask = load_body_mask(patient / "Masks")
    if body_mask is not None:
        print(f"Body mask: {int(body_mask.sum())} voxels (metrics + SSIM masked)")
    else:
        print("[WARN] No body mask — metrics computed globally")

    src_vol_path = patient / "SourceVolumes" / "sub_CT_06_mha.npy"
    if not src_vol_path.is_file():
        src_vol_path = next((patient / "SourceVolumes").glob("sub_CT_*.npy"))
    src_vol_np = _normalize(np.load(src_vol_path).squeeze())
    while src_vol_np.ndim > 3:
        src_vol_np = src_vol_np.squeeze(0)
    src_vol_t = torch.from_numpy(src_vol_np[None, None]).to(device)

    ptv_path = next((patient / "Masks").glob("*PTV*_mha.npy"), None)
    if ptv_path is None:
        ptv_path = next((patient / "Masks").glob("*PTV*.npy"), None)
    src_ptv_np = _normalize(np.load(ptv_path).squeeze()) if ptv_path else (body_mask or (src_vol_np > 0.08).astype(np.float32))
    while src_ptv_np.ndim > 3:
        src_ptv_np = src_ptv_np.squeeze(0)
    src_ptv_t = torch.from_numpy(src_ptv_np[None, None]).to(device)

    centroid_op = losses.centroid_ptv()
    dice_op = losses.dice()
    jacobian_op = losses.jacobian_determinant()
    target_size = [im_size] * 3

    def _ensure_size(t):
        if list(t.shape[-3:]) != target_size:
            return torch.nn.functional.interpolate(t, size=target_size, mode="trilinear", align_corners=False)
        return t

    results = {
        "angles": [],
        "dice": [],
        "mse": [],
        "psnr": [],
        "ssim": [],
        "det_j_neg_fraction": [],
        "shifts_mm": {"lr": [], "si": [], "ap": [], "3d": []},
        "gt_shifts_mm": {"lr": [], "si": [], "ap": []},
    }

    with torch.no_grad():
        for sample in tqdm(index, desc="Sweep eval"):
            src_p = torch.from_numpy(_normalize(np.load(sample["source_proj"]))[None, None]).to(device)
            tgt_p = torch.from_numpy(_normalize(np.load(sample["target_proj"]))[None, None]).to(device)
            angle = torch.tensor([sample["angle"]], dtype=torch.float32, device=device)

            _, pred_flow = model(src_p, tgt_p, src_vol_t, angle=angle)
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
            mse_val, psnr_val, ssim_val = masked_volume_metrics(gt_vol_3d, pred_vol_3d, body_mask)

            try:
                pf = pred_flow.detach().float().cpu().numpy()
                metric_flows = np.squeeze(pf)
                if metric_flows.ndim == 4 and metric_flows.shape[0] == 3:
                    disp = np.stack([metric_flows[0], metric_flows[1], metric_flows[2]], axis=-1).astype(np.float64)
                    det_j_ratio = float(np.mean(jacobian_op.loss(disp) <= 0))
                else:
                    det_j_ratio = float("nan")
            except Exception:
                det_j_ratio = float("nan")

            results["angles"].append(sample["angle"])
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

    title = args.scan_id or patient.name
    print("\n--- Validation Summary (body-masked PSNR/SSIM) ---")
    print(f"  Samples evaluated : {len(results['angles'])}")
    print(f"  Mean Dice         : {np.mean(results['dice']):.4f}")
    print(f"  Mean 3D Error     : {np.mean(results['shifts_mm']['3d']):.3f} mm")
    print(f"  Mean PSNR         : {np.mean(results['psnr']):.2f} dB")
    print(f"  Mean SSIM         : {np.nanmean(results['ssim']):.4f}")
    print(f"  Mean neg det(J)   : {np.nanmean(results['det_j_neg_fraction']):.4f}")

    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    generate_sweep_trace(results, out_path / "Performance_Trace_by_index.png", f"{title} sweep (body mask)")

    idx_path = out_path / "Performance_Trace_by_index.png"
    trace_path = out_path / "Performance_Trace.png"
    if not trace_path.exists():
        trace_path.symlink_to(idx_path.name)

    print(f"Success! Saved metrics + trace plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
