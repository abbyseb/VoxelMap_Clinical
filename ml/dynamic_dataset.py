import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

from ml.training_config import REFERENCE_IM_SIZE


def resolve_voxel_map_data_root(raw: Path | str) -> Path:
    """Map a folder to the prepared Voxel Map patient root (contains TargetProjections/)."""
    p = Path(raw).expanduser()
    try:
        p = p.resolve(strict=False)
    except OSError:
        return Path(raw).expanduser()

    if not p.exists():
        return p

    if (p / "TargetProjections").is_dir():
        return p

    parts_lower = [x.lower() for x in p.parts]

    def walk_mt_train(patient_id: str, start: Path) -> Path | None:
        cur = start
        while True:
            cand = cur / "ModelTraining" / "train" / patient_id
            if (cand / "TargetProjections").is_dir():
                return cand.resolve()
            if cur.parent == cur:
                break
            cur = cur.parent
        return None

    if p.name.lower() == "train":
        pid = p.parent.name
        found = walk_mt_train(pid, p.parent.parent)
        if found is not None:
            return found

    if "modeltraining" not in parts_lower:
        pid = p.name
        found = walk_mt_train(pid, p.parent)
        if found is not None:
            return found

    return p


class VoxelMapDataset(Dataset):
    """Maps ModelTraining folder layout into PyTorch tensors."""

    def __init__(self, data_roots, use_angles=False, im_size=REFERENCE_IM_SIZE):
        self.data_roots = [Path(p) for p in data_roots]
        self.use_angles = use_angles
        self.im_size = int(im_size)
        self.samples = []
        self._build_index()

    def _build_index(self):
        self.all_patient_roots = []
        for root in self.data_roots:
            if (root / "TargetProjections").exists():
                self.all_patient_roots.append(root)

            found_sub_roots = [p.parent for p in root.rglob("TargetProjections") if p.is_dir()]
            for sr in found_sub_roots:
                if sr not in self.all_patient_roots:
                    self.all_patient_roots.append(sr)

        for root in self.all_patient_roots:
            tgt_dir = root / "TargetProjections"
            src_dir = root / "SourceProjections"
            vol_dir = root / "SourceVolumes"
            dvf_dir = root / "DVFs"
            mask_dir = root / "Masks"
            angle_file = root / "Angles.csv"

            if not tgt_dir.exists():
                continue

            patient_angles = None
            if self.use_angles and angle_file.exists():
                patient_angles = pd.read_csv(angle_file, header=None).values.squeeze()
                patient_angles = np.atleast_1d(np.asarray(patient_angles, dtype=np.float64)).ravel()

            source_vol_path = vol_dir / "sub_CT_06_mha.npy"
            if not source_vol_path.exists():
                source_vol_path = next(vol_dir.glob("sub_CT_*.npy"), None)

            source_mask_path = mask_dir / "sub_Abdomen_mha.npy"
            if not source_mask_path.exists():
                synonyms = ["*Abdomen*", "*Body*", "*Hull*"]
                for syn in synonyms:
                    match = next(mask_dir.glob(f"{syn}.npy"), None)
                    if match:
                        source_mask_path = match
                        break

                if not source_mask_path.exists():
                    source_mask_path = next(mask_dir.glob("Mask_Body*.npy"), None)
                    if not source_mask_path:
                        source_mask_path = next(mask_dir.glob("Mask_Abdomen*.npy"), None)

            for tgt_file in sorted(tgt_dir.glob("*_bin.npy")):
                if tgt_file.name.startswith("._"):
                    continue
                parts = tgt_file.name.split("_")
                if len(parts) < 3:
                    continue

                vol_num = parts[0]
                proj_num_str = parts[2]
                proj_num = int(proj_num_str)

                src_file = src_dir / f"06_Proj_{proj_num_str}_bin.npy"
                if not src_file.exists():
                    continue

                dvf_file = dvf_dir / f"DVF_{vol_num}_mha.npy"
                if not dvf_file.exists():
                    continue

                angle = None
                if patient_angles is not None:
                    pi = proj_num - 1
                    if pi < 0 or pi >= len(patient_angles):
                        continue
                    angle = float(patient_angles[pi])

                self.samples.append(
                    {
                        "target_proj": tgt_file,
                        "source_proj": src_file,
                        "target_dvf": dvf_file,
                        "source_vol": source_vol_path,
                        "source_mask": source_mask_path,
                        "angle": angle,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def _normalize(self, arr):
        min_v = np.min(arr)
        max_v = np.max(arr)
        return (arr - min_v) / (max_v - min_v + 1e-8)

    @staticmethod
    def _resize_flow_field(x, target_spatial):
        if x.shape[-3:] == target_spatial:
            return x
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=target_spatial, mode="trilinear", align_corners=False)
        return x.squeeze(0)

    @staticmethod
    def _resize_mask_field(m, target_spatial):
        if m.shape[-3:] == target_spatial:
            return m
        m = m.unsqueeze(0)
        try:
            out = F.interpolate(m, size=target_spatial, mode="nearest-exact")
        except Exception:
            out = F.interpolate(m, size=target_spatial, mode="nearest")
        return out.squeeze(0)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        target_proj = self._normalize(np.load(sample["target_proj"]))
        source_proj = self._normalize(np.load(sample["source_proj"]))
        source_vol = self._normalize(np.load(sample["source_vol"]))

        mask_path = sample["source_mask"]
        if mask_path is not None and Path(mask_path).exists():
            source_hull = np.load(mask_path)
        else:
            source_hull = np.ones_like(source_vol, dtype=np.float32)

        target_dvf = np.load(sample["target_dvf"])
        target_spatial = tuple(source_vol.shape[-3:])

        source_projections = torch.from_numpy(source_proj[None, :, :]).float()
        target_projections = torch.from_numpy(target_proj[None, :, :]).float()
        source_volumes = torch.from_numpy(source_vol[None, :, :, :]).float()

        source_abdomen = torch.from_numpy(source_hull[None, :, :, :]).float()
        if source_abdomen.shape[-3:] != target_spatial:
            source_abdomen = self._resize_mask_field(source_abdomen, target_spatial)

        target_flow = torch.from_numpy(np.moveaxis(target_dvf, -1, 0)).float()
        if target_flow.shape[-3:] != target_spatial:
            target_flow = self._resize_flow_field(target_flow, target_spatial)

        if self.use_angles:
            val = sample["angle"] if sample["angle"] is not None else 0.0
            angle_tensor = torch.tensor(val).float()
            return (
                source_projections,
                target_projections,
                source_volumes,
                source_abdomen,
                target_flow,
                angle_tensor,
            )

        return source_projections, target_projections, source_volumes, source_abdomen, target_flow
