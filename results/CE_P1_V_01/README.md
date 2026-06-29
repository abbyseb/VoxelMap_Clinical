# CE_P1_V_01 — Workstation Results

**Scan:** Clinical Elekta `CE_P1_V_01`  
**Model:** concatenated VoxelMap + FiLM  
**Hardware:** NVIDIA RTX 6000 Ada, CUDA GPU 0  
**Date:** 2026-06-29

Heavy artifacts (checkpoints `.pt`, `ModelTraining/`, raw volumes) remain under `runs/` (gitignored).

## Training (50 epochs)

| Item | Value |
|------|-------|
| Train pairs | 2,754 |
| Val pairs | 306 |
| Best epoch | 50 |
| Best val loss | 0.0571 |
| Final train loss | 0.0623 |

- Loss curves: `plots/loss_curves.png`
- **Test sweep MP4:** `videos/CE_P1_V_01_test_sweep.mp4` (340 projections, phase labels)
- **DVF warp panel MP4:** `videos/CE_P1_V_01_dvf_warp_panels.mp4` (2×2: axial default)
- **DVF warp (sagittal slice 78):** `videos/CE_P1_V_01_dvf_warp_sagittal78.mp4` — see `dvf_view_config.json`
- **View config:** `dvf_view_config.json` (plane/slice/orientation for `export_dvf_warp_mp4.py`)
- Sweep trace (projection index): `plots/Performance_Trace_by_index.png`
- Per-epoch history: `loss_history.json`
- Checkpoints (local): `runs/CE_P1_V_01/checkpoints/best.pt`, `epochs/epoch_*.pt`

## Breathing sweep evaluation (340 projections)

Synthetic onboard CBCT sweep (`ModelTraining/test/`, §6 NOTE_TO_AGENT).

| Metric | Mean | Std |
|--------|------|-----|
| Dice (PTV) | 0.893 | 0.030 |
| PSNR (warped CT) | 36.87 dB | 3.30 dB |
| SSIM (warped CT) | 0.889 | 0.050 |
| 3D centroid error | 0.81 mm | — |
| neg det(J) fraction | 0.0000 | — |

- Per-angle metrics: `sweep_metrics.json`
- Trace plot: `plots/Performance_Trace.png` (legacy: vs gantry angle — flat GT segments)
- **Preferred trace:** `plots/Performance_Trace_by_index.png` (vs projection index)

## Reproduce

```bash
export VOXELMAP_CLINICAL_ROOT=/path/to/VoxelMap_Clinical
export LEARN_GUI_ROOT=/path/to/LEARN-GUI/LEARN-GUI-Python
CUDA_VISIBLE_DEVICES=0 python scripts/run_elekta_phase3_train.py
CUDA_VISIBLE_DEVICES=0 python scripts/run_elekta_sweep_eval.py
```
