"""Body-mask helpers for metrics and visualization."""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    _psnr = None
    _ssim = None


def load_body_mask(masks_dir: Path) -> np.ndarray | None:
    """Load binary body mask (D,H,W) from ModelTraining Masks/."""
    masks_dir = Path(masks_dir)
    for pat in ("*Body*_mha.npy", "*Body*.npy", "sub_Abdomen_mha.npy", "*Abdomen*.npy"):
        hits = sorted(masks_dir.glob(pat))
        if hits:
            m = np.load(hits[0]).astype(np.float32).squeeze()
            while m.ndim > 3:
                m = m.squeeze(0)
            return (m > 0).astype(np.float32)
    return None


def apply_body_mask_slice(slice_2d: np.ndarray, body_mask_3d: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Zero voxels outside body on a 2D slice (for display)."""
    m = np.take(body_mask_3d, index, axis=axis) > 0
    out = slice_2d.copy()
    out[~m] = 0.0
    return out


def masked_volume_metrics(
    gt_vol: np.ndarray,
    pred_vol: np.ndarray,
    body_mask: np.ndarray | None,
) -> tuple[float, float, float]:
    """Return (mse, psnr, ssim) inside body mask when available."""
    gt_vol = np.asarray(gt_vol, dtype=np.float32)
    pred_vol = np.asarray(pred_vol, dtype=np.float32)

    if body_mask is not None:
        mask_bin = body_mask > 0
        mse_gt = gt_vol[mask_bin]
        mse_pred = pred_vol[mask_bin]
    else:
        mse_gt = gt_vol.ravel()
        mse_pred = pred_vol.ravel()

    mse_val = float(np.mean((mse_pred - mse_gt) ** 2))

    if HAS_SKIMAGE and _psnr is not None and mse_gt.size > 0:
        psnr_val = float(_psnr(mse_gt, mse_pred, data_range=1.0))
    else:
        psnr_val = float(10 * np.log10(1.0 / (mse_val + 1e-8)))

    if HAS_SKIMAGE and _ssim is not None:
        try:
            if body_mask is not None:
                ssim_val = float(
                    _ssim(
                        gt_vol,
                        pred_vol,
                        data_range=1.0,
                        mask=body_mask > 0,
                    )
                )
            else:
                ssim_val = float(_ssim(gt_vol, pred_vol, data_range=1.0))
        except Exception:
            ssim_val = float("nan")
    else:
        ssim_val = float("nan")

    return mse_val, psnr_val, ssim_val
