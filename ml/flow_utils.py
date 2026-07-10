"""DVF / mask resampling helpers for subsample-train → full-res eval."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def upsample_dvf(
    flow: torch.Tensor,
    size_native: tuple[int, int, int],
    *,
    align_corners: bool = True,
) -> torch.Tensor:
    """Upsample a voxel-unit DVF and scale displacements to the target grid.

    Args:
        flow: (B, 3, D, H, W) displacements in *source-grid voxel units*.
        size_native: (D, H, W) target spatial size.
        align_corners: must match SpatialTransformer / model convention (True).

    Returns:
        (B, 3, Dn, Hn, Wn) displacements in *native-grid voxel units*.
    """
    if flow.ndim != 5 or flow.shape[1] != 3:
        raise ValueError(f"Expected flow (B, 3, D, H, W), got {tuple(flow.shape)}")
    d, h, w = flow.shape[2], flow.shape[3], flow.shape[4]
    dn, hn, wn = int(size_native[0]), int(size_native[1]), int(size_native[2])
    if (d, h, w) == (dn, hn, wn):
        return flow
    scale = torch.tensor(
        [dn / d, hn / h, wn / w],
        device=flow.device,
        dtype=flow.dtype,
    ).view(1, 3, 1, 1, 1)
    flow_up = F.interpolate(
        flow, size=(dn, hn, wn), mode="trilinear", align_corners=align_corners
    )
    return flow_up * scale


def upsample_mask(
    mask: torch.Tensor,
    size_native: tuple[int, int, int],
) -> torch.Tensor:
    """Nearest-neighbor upsample for binary / label masks. (B, 1, D, H, W)."""
    if mask.shape[-3:] == tuple(size_native):
        return mask
    try:
        return F.interpolate(mask.float(), size=size_native, mode="nearest-exact")
    except Exception:
        return F.interpolate(mask.float(), size=size_native, mode="nearest")


def load_mha_array(path: Path | str) -> np.ndarray:
    """Load an MHA/MHD volume as float32 (D, H, W) via SimpleITK."""
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    while arr.ndim > 3:
        arr = arr.squeeze(0)
    return arr


def spacing_from_grids(
    native_size: tuple[int, int, int],
    train_size: tuple[int, int, int] = (128, 128, 128),
) -> tuple[float, float, float]:
    """Physical mm/voxel on the training grid, assuming native spacing is 1 mm."""
    return tuple(n / t for n, t in zip(native_size, train_size))
