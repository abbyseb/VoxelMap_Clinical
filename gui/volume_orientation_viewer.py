#!/usr/bin/env python3
"""Interactive source-volume viewer with orientation labels and workstation export."""
from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

REPO = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", Path(__file__).resolve().parents[1]))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.volume_view import (  # noqa: E402
    PLANE_PRESETS,
    VolumeViewConfig,
    default_volume_path,
    extract_slice,
)


def _normalize(vol: np.ndarray) -> np.ndarray:
    lo, hi = float(vol.min()), float(vol.max())
    if hi - lo < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - lo) / (hi - lo)).astype(np.float32)


class VolumeOrientationViewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VoxelMap — Source Volume Orientation Viewer")
        self.geometry("1100x780")

        self.repo = REPO
        self.cfg = VolumeViewConfig().resolve()
        self.volume_path: Path | None = None
        self.volume: np.ndarray | None = None
        self.v_lo = 0.0
        self.v_hi = 1.0

        self._build_ui()
        self._try_load_default()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Scan ID:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        self.scan_var = tk.StringVar(value=self.cfg.scan_id)
        ttk.Entry(top, textvariable=self.scan_var, width=16).grid(row=0, column=1, sticky=tk.W)

        ttk.Button(top, text="Browse volume…", command=self._browse_volume).grid(row=0, column=2, padx=8)
        self.path_label = ttk.Label(top, text="(no file)", foreground="#555")
        self.path_label.grid(row=0, column=3, sticky=tk.W)

        ctrl = ttk.LabelFrame(self, text="View", padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        ttk.Label(ctrl, text="Plane:").grid(row=0, column=0, sticky=tk.W)
        self.plane_var = tk.StringVar(value=self.cfg.plane)
        plane_cb = ttk.Combobox(
            ctrl,
            textvariable=self.plane_var,
            values=list(PLANE_PRESETS.keys()),
            state="readonly",
            width=12,
        )
        plane_cb.grid(row=0, column=1, sticky=tk.W, padx=4)
        plane_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_plane_change())

        ttk.Label(ctrl, text="Slice:").grid(row=0, column=2, padx=(16, 4))
        self.slice_var = tk.IntVar(value=self.cfg.slice_index)
        self.slice_scale = ttk.Scale(
            ctrl,
            from_=0,
            to=127,
            orient=tk.HORIZONTAL,
            variable=self.slice_var,
            command=lambda _v: self._on_slice_change(),
            length=320,
        )
        self.slice_scale.grid(row=0, column=3, sticky=tk.W)
        self.slice_num = ttk.Label(ctrl, text="64")
        self.slice_num.grid(row=0, column=4, padx=4)

        self.flip_h_var = tk.BooleanVar(value=False)
        self.flip_v_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Flip horizontal", variable=self.flip_h_var, command=self._refresh).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=4
        )
        ttk.Checkbutton(ctrl, text="Flip vertical", variable=self.flip_v_var, command=self._refresh).grid(
            row=1, column=2, columnspan=2, sticky=tk.W, pady=4
        )

        mid = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        mid.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        plot_frame = ttk.Frame(mid)
        mid.add(plot_frame, weight=3)

        self.fig, self.ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame)

        info_frame = ttk.Frame(mid, padding=4)
        mid.add(info_frame, weight=1)

        ttk.Label(info_frame, text="Orientation", font=("", 11, "bold")).pack(anchor=tk.W)
        self.info_text = tk.Text(info_frame, width=36, height=14, wrap=tk.WORD, font=("Menlo", 11))
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=4)

        btn_row = ttk.Frame(info_frame)
        btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="Copy workstation cmd", command=self._copy_command).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="Save config JSON", command=self._save_config).pack(side=tk.LEFT)

        ttk.Label(
            info_frame,
            text="Copy the command + JSON and send to the workstation.\n"
            "Run export_dvf_warp_mp4.py with --view-config to match this view.",
            wraplength=280,
            foreground="#444",
        ).pack(anchor=tk.W, pady=8)

    def _sync_cfg_from_ui(self) -> None:
        self.cfg.scan_id = self.scan_var.get().strip() or "CE_P1_V_01"
        self.cfg.plane = self.plane_var.get()  # type: ignore[assignment]
        self.cfg.slice_index = int(self.slice_var.get())
        self.cfg.flip_h = bool(self.flip_h_var.get())
        self.cfg.flip_v = bool(self.flip_v_var.get())
        # reset plane-derived axes when plane changes
        self.cfg.slice_axis = None
        self.cfg.h_axis = None
        self.cfg.v_axis = None
        self.cfg.flow_u = None
        self.cfg.flow_v = None
        self.cfg.resolve()

    def _try_load_default(self) -> None:
        try:
            path = default_volume_path(self.repo, self.cfg.scan_id, self.cfg.volume_relpath)
            self._load_volume(path)
        except FileNotFoundError:
            self.path_label.config(text="No default volume — use Browse")
            self._update_info()

    def _browse_volume(self) -> None:
        path = filedialog.askopenfilename(
            title="Select source volume",
            initialdir=str(self.repo / "runs"),
            filetypes=[("NumPy volume", "*.npy"), ("MetaImage", "*.mha"), ("All", "*.*")],
        )
        if path:
            self._load_volume(Path(path))

    def _load_volume(self, path: Path) -> None:
        if path.suffix == ".npy":
            vol = np.load(path).astype(np.float32).squeeze()
        else:
            try:
                import SimpleITK as sitk

                vol = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
            except ImportError as exc:
                messagebox.showerror("Missing dependency", "Install SimpleITK to load .mha files.\n" + str(exc))
                return
        if vol.ndim != 3:
            messagebox.showerror("Invalid volume", f"Expected 3D array, got shape {vol.shape}")
            return
        self.volume_path = path
        self.volume = _normalize(vol)
        self.v_lo, self.v_hi = np.percentile(self.volume, [1, 99])
        n = self.volume.shape[int(PLANE_PRESETS[self.plane_var.get()]["slice_axis"])]
        self.slice_scale.config(to=max(0, n - 1))
        mid = n // 2
        self.slice_var.set(mid)
        self.cfg.slice_index = mid
        short = str(path)
        if len(short) > 70:
            short = "…" + short[-67:]
        self.path_label.config(text=short)
        self._refresh()

    def _on_plane_change(self) -> None:
        if self.volume is not None:
            self._sync_cfg_from_ui()
            n = self.volume.shape[int(self.cfg.slice_axis)]
            self.slice_scale.config(to=max(0, n - 1))
            self.slice_var.set(n // 2)
        self._refresh()

    def _on_slice_change(self) -> None:
        self.slice_num.config(text=str(int(float(self.slice_var.get()))))
        self._refresh()

    def _refresh(self) -> None:
        self._sync_cfg_from_ui()
        self._update_info()
        if self.volume is None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Load a volume", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        sl = extract_slice(self.volume, self.cfg)
        self.ax.clear()
        self.ax.imshow(
            sl,
            cmap="gray",
            vmin=self.v_lo,
            vmax=self.v_hi,
            origin="upper",
            aspect="equal",
        )
        h, w = sl.shape
        self.ax.set_xlabel(f"← {self.cfg.horizontal} →", fontsize=11)
        self.ax.set_ylabel(f"↑ {self.cfg.vertical}", fontsize=11)
        self.ax.set_title(
            f"{self.cfg.plane.capitalize()}  |  ⊥ {self.cfg.slice_normal}  |  slice {self.cfg.clamp_slice(self.volume.shape)}",
            fontsize=11,
        )
        # corner axis diagram
        self.ax.annotate(
            f"RAI / {self.cfg.coordinate_system}",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=9,
            color="yellow",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.55),
        )
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _update_info(self) -> None:
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, self.cfg.orientation_summary())
        if self.volume_path:
            self.info_text.insert(tk.END, f"\n\nVolume:\n{self.volume_path}")
            if self.volume is not None:
                self.info_text.insert(tk.END, f"\nShape: {tuple(self.volume.shape)} (LR, SI, AP)")

    def _copy_command(self) -> None:
        self._sync_cfg_from_ui()
        text = self.cfg.workstation_command(self.repo)
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copied", "Workstation command + JSON copied to clipboard.")

    def _save_config(self) -> None:
        self._sync_cfg_from_ui()
        default = self.repo / "results" / self.cfg.scan_id / "dvf_view_config.json"
        path = filedialog.asksaveasfilename(
            title="Save view config",
            initialdir=str(default.parent),
            initialfile=default.name,
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        self.cfg.save_json(Path(path))
        messagebox.showinfo("Saved", f"Config saved:\n{path}")


def main() -> int:
    app = VolumeOrientationViewer()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
