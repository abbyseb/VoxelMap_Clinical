"""Volume slice / orientation helpers for GUI and DVF warp MP4 export.

SPARE ModelTraining volumes are 128³ in RAI / IEC 61217 order:
  axis 0 = LR (Left–Right)
  axis 1 = SI (Superior–Inferior)
  axis 2 = AP (Anterior–Posterior)

DVF flow channels match: 0=LR, 1=SI, 2=AP.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

AXIS_NAMES = ("LR", "SI", "AP")
PlaneName = Literal["axial", "sagittal", "coronal"]

PLANE_PRESETS: dict[str, dict[str, int | str]] = {
    "axial": {
        "slice_axis": 0,
        "h_axis": 1,
        "v_axis": 2,
        "flow_u": 1,
        "flow_v": 2,
        "slice_normal": "LR",
        "horizontal": "SI",
        "vertical": "AP",
    },
    "sagittal": {
        "slice_axis": 2,
        "h_axis": 0,
        "v_axis": 1,
        "flow_u": 0,
        "flow_v": 1,
        "slice_normal": "AP",
        "horizontal": "LR",
        "vertical": "SI",
    },
    "coronal": {
        "slice_axis": 1,
        "h_axis": 0,
        "v_axis": 2,
        "flow_u": 0,
        "flow_v": 2,
        "slice_normal": "SI",
        "horizontal": "LR",
        "vertical": "AP",
    },
}


@dataclass
class VolumeViewConfig:
    scan_id: str = "CE_P1_V_01"
    volume_relpath: str = "SourceVolumes/sub_CT_06_mha.npy"
    plane: PlaneName = "axial"
    slice_index: int = 64
    flip_h: bool = False
    flip_v: bool = False
    frame: str = "RAI"
    coordinate_system: str = "IEC 61217"
    slice_axis: int | None = None
    h_axis: int | None = None
    v_axis: int | None = None
    flow_u: int | None = None
    flow_v: int | None = None

    def resolve(self) -> VolumeViewConfig:
        preset = PLANE_PRESETS[self.plane]
        if self.slice_axis is None:
            self.slice_axis = int(preset["slice_axis"])
        if self.h_axis is None:
            self.h_axis = int(preset["h_axis"])
        if self.v_axis is None:
            self.v_axis = int(preset["v_axis"])
        if self.flow_u is None:
            self.flow_u = int(preset["flow_u"])
        if self.flow_v is None:
            self.flow_v = int(preset["flow_v"])
        return self

    @property
    def slice_normal(self) -> str:
        return str(PLANE_PRESETS[self.plane]["slice_normal"])

    @property
    def horizontal(self) -> str:
        return str(PLANE_PRESETS[self.plane]["horizontal"])

    @property
    def vertical(self) -> str:
        return str(PLANE_PRESETS[self.plane]["vertical"])

    def clamp_slice(self, shape: tuple[int, ...]) -> int:
        self.resolve()
        n = shape[int(self.slice_axis)]
        return int(np.clip(self.slice_index, 0, n - 1))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> VolumeViewConfig:
        allowed = {f.name for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in allowed}
        return cls(**clean).resolve()

    def save_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> VolumeViewConfig:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def orientation_summary(self) -> str:
        self.resolve()
        return (
            f"Frame: {self.frame} ({self.coordinate_system})\n"
            f"Plane: {self.plane} — slice ⊥ {self.slice_normal}\n"
            f"Display: horizontal → {self.horizontal}, vertical → {self.vertical}\n"
            f"Slice index: {self.slice_index} / 127  (axis {self.slice_axis} = {AXIS_NAMES[self.slice_axis]})\n"
            f"DVF arrows: u={AXIS_NAMES[self.flow_u]}, v={AXIS_NAMES[self.flow_v]}\n"
            f"Flip H: {self.flip_h}  |  Flip V: {self.flip_v}"
        )

    def workstation_command(self, repo_root: Path) -> str:
        cfg_path = repo_root / "results" / self.scan_id / "dvf_view_config.json"
        try:
            rel_cfg = cfg_path.relative_to(repo_root)
        except ValueError:
            rel_cfg = cfg_path
        return "\n".join(
            [
                "# Paste on CUDA workstation",
                "export VOXELMAP_CLINICAL_ROOT=/path/to/VoxelMap_Clinical",
                'cd "$VOXELMAP_CLINICAL_ROOT"',
                f"# Save this JSON to {rel_cfg} (or use Save Config in the GUI)",
                "",
                "python scripts/export_dvf_warp_mp4.py \\",
                f"  --scan-id {self.scan_id} \\",
                f"  --view-config {rel_cfg} \\",
                "  --device cuda",
                "",
                "# --- view config JSON ---",
                json.dumps(self.to_dict(), indent=2),
            ]
        )


def _orient_plane_2d(arr: np.ndarray, slice_axis: int, h_axis: int, v_axis: int) -> np.ndarray:
    remaining = [a for a in range(3) if a != slice_axis]
    if remaining == [h_axis, v_axis]:
        return arr
    if remaining == [v_axis, h_axis]:
        return arr.T
    return arr


def extract_slice(vol: np.ndarray, cfg: VolumeViewConfig) -> np.ndarray:
    cfg.resolve()
    idx = cfg.clamp_slice(vol.shape)
    sl = np.take(vol, idx, axis=int(cfg.slice_axis))
    out = _orient_plane_2d(sl, int(cfg.slice_axis), int(cfg.h_axis), int(cfg.v_axis))
    if cfg.flip_h:
        out = np.flip(out, axis=1)
    if cfg.flip_v:
        out = np.flip(out, axis=0)
    return out


def extract_flow_uv(flow_chw: np.ndarray, cfg: VolumeViewConfig) -> tuple[np.ndarray, np.ndarray]:
    """flow shape (3, D, H, W) with channels LR, SI, AP."""
    cfg.resolve()
    idx = cfg.clamp_slice(flow_chw.shape[1:])
    sa, ha, va = int(cfg.slice_axis), int(cfg.h_axis), int(cfg.v_axis)

    def component(ch: int) -> np.ndarray:
        sl = np.take(flow_chw[ch], idx, axis=sa)
        out = _orient_plane_2d(sl, sa, ha, va)
        if cfg.flip_h:
            out = np.flip(out, axis=1)
        if cfg.flip_v:
            out = np.flip(out, axis=0)
        return out

    return component(int(cfg.flow_u)), component(int(cfg.flow_v))


def default_volume_path(repo: Path, scan_id: str, relpath: str) -> Path:
    candidates = [
        repo / "runs" / scan_id / "ModelTraining" / "test" / scan_id / relpath,
        repo / "runs" / scan_id / "ModelTraining" / "train" / scan_id / relpath,
        repo / "runs" / scan_id / scan_id / "train" / relpath.replace("SourceVolumes/", "sub_").replace("_mha.npy", ".mha"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"No volume found for {scan_id} ({relpath})")
