# No-FiLM FDK — Breathing Sweep Metrics

**Scope:** No-FiLM FDK models (`checkpoints_nofilm/best.pt`).  
**Cohort:** 10 clinical FDK4D scans — Elekta `CE_P1`–`P5` (340 frames), Varian `CV_P1`–`P5` (680 frames).  
**Eval:** full breathing sweep @128³ with body-masked PSNR/SSIM.  
**Artifacts:** `results/<scan>_fdk/sweep_metrics_nofilm.json`  
**Videos:** 5×2 white-background panels (pred/GT DVF body-masked, 128 + native upscale warp) under `results/<scan>_fdk/videos_nofilm/*_sweep*_nofilm.mp4`.

FiLM vs No-FiLM comparison also lives in `FDK_FILM_VS_NOFILM.md` Part A. Stratified train-pair **full-res upscale** quantification (90 samples) is in `NOFILM_FDK_FULLRES_UPSCALE.md`.

---

## Executive summary (No-FiLM sweep @128)

| | Mean (10 patients) |
|--|--------------------|
| Dice | **0.9052** |
| 3D centroid error (mm) | **0.737** |
| SSIM (body mask) | **0.8191** |
| PSNR (dB, body mask) | **41.16** |
| neg det(J) fraction | 0.0000 |

Patients where **No-FiLM beats FiLM** on the same sweep (of 10):

| Metric | No-FiLM wins |
|--------|--------------|
| Dice | 8/10 |
| 3D Error | 8/10 |
| SSIM | 6/10 |
| PSNR | 7/10 |

---

## No-FiLM sweep metrics (@128)

### Elekta (`CE_*_fdk`, 340 projections)

| Scan | Frames | Dice | 3D Err (mm) | SSIM | PSNR (dB) | neg det(J) |
|------|--------|------|-------------|------|-----------|------------|
| `CE_P1_V_01_fdk` | 340 | 0.8883 | 0.675 | 0.8664 | 36.19 | 0.0000 |
| `CE_P2_V_01_fdk` | 340 | 0.9369 | 0.441 | 0.8537 | 36.62 | 0.0000 |
| `CE_P3_V_01_fdk` | 340 | 0.9090 | 0.712 | 0.9204 | 36.80 | 0.0000 |
| `CE_P4_V_01_fdk` | 340 | 0.9456 | 1.195 | 0.7567 | 31.73 | 0.0000 |
| `CE_P5_V_01_fdk` | 340 | 0.9048 | 1.155 | 0.8131 | 39.90 | 0.0000 |
| **Mean** | — | 0.9169 | 0.835 | 0.8421 | 36.25 | 0.0000 |

### Varian (`CV_*_fdk`, 680 projections)

| Scan | Frames | Dice | 3D Err (mm) | SSIM | PSNR (dB) | neg det(J) |
|------|--------|------|-------------|------|-----------|------------|
| `CV_P1_V_01_fdk` | 680 | 0.8861 | 0.857 | 0.7036 | 35.30 | 0.0000 |
| `CV_P2_V_01_fdk` | 680 | 0.9167 | 0.454 | 0.7450 | 38.63 | 0.0000 |
| `CV_P3_V_01_fdk` | 680 | 0.8541 | 0.791 | 0.8826 | 51.92 | 0.0000 |
| `CV_P4_V_01_fdk` | 680 | 0.8787 | 0.545 | 0.8062 | 56.19 | 0.0000 |
| `CV_P5_V_01_fdk` | 680 | 0.9323 | 0.549 | 0.8431 | 48.29 | 0.0000 |
| **Mean** | — | 0.8936 | 0.639 | 0.7961 | 46.06 | 0.0000 |

### All FDK No-FiLM

| Scan | Frames | Dice | 3D Err (mm) | SSIM | PSNR (dB) | neg det(J) |
|------|--------|------|-------------|------|-----------|------------|
| `CE_P1_V_01_fdk` | 340 | 0.8883 | 0.675 | 0.8664 | 36.19 | 0.0000 |
| `CE_P2_V_01_fdk` | 340 | 0.9369 | 0.441 | 0.8537 | 36.62 | 0.0000 |
| `CE_P3_V_01_fdk` | 340 | 0.9090 | 0.712 | 0.9204 | 36.80 | 0.0000 |
| `CE_P4_V_01_fdk` | 340 | 0.9456 | 1.195 | 0.7567 | 31.73 | 0.0000 |
| `CE_P5_V_01_fdk` | 340 | 0.9048 | 1.155 | 0.8131 | 39.90 | 0.0000 |
| `CV_P1_V_01_fdk` | 680 | 0.8861 | 0.857 | 0.7036 | 35.30 | 0.0000 |
| `CV_P2_V_01_fdk` | 680 | 0.9167 | 0.454 | 0.7450 | 38.63 | 0.0000 |
| `CV_P3_V_01_fdk` | 680 | 0.8541 | 0.791 | 0.8826 | 51.92 | 0.0000 |
| `CV_P4_V_01_fdk` | 680 | 0.8787 | 0.545 | 0.8062 | 56.19 | 0.0000 |
| `CV_P5_V_01_fdk` | 680 | 0.9323 | 0.549 | 0.8431 | 48.29 | 0.0000 |
| **Mean** | — | 0.9052 | 0.737 | 0.8191 | 41.16 | 0.0000 |

---

## No-FiLM vs FiLM (same breathing sweep)

### Elekta

| Scan | Dice FiLM | Dice NoFiLM | Δ Dice | 3D FiLM | 3D NoFiLM | Δ 3D | SSIM FiLM | SSIM NoFiLM | Δ SSIM | PSNR FiLM | PSNR NoFiLM | Δ PSNR |
|------|-----------|-------------|--------|---------|-----------|------|-----------|-------------|--------|-----------|-------------|--------|
| `CE_P1_V_01_fdk` | 0.8776 | 0.8883 | +0.0106 | 1.286 | 0.675 | -0.612 | 0.8395 | 0.8664 | +0.0269 | 34.79 | 36.19 | +1.40 |
| `CE_P2_V_01_fdk` | 0.9207 | 0.9369 | +0.0161 | 2.271 | 0.441 | -1.830 | 0.8036 | 0.8537 | +0.0501 | 30.02 | 36.62 | +6.60 |
| `CE_P3_V_01_fdk` | 0.8877 | 0.9090 | +0.0213 | 2.606 | 0.712 | -1.894 | 0.8578 | 0.9204 | +0.0625 | 31.14 | 36.80 | +5.66 |
| `CE_P4_V_01_fdk` | 0.9453 | 0.9456 | +0.0003 | 1.252 | 1.195 | -0.057 | 0.7732 | 0.7567 | -0.0166 | 31.34 | 31.73 | +0.39 |
| `CE_P5_V_01_fdk` | 0.8952 | 0.9048 | +0.0096 | 1.472 | 1.155 | -0.317 | 0.8249 | 0.8131 | -0.0118 | 40.11 | 39.90 | -0.21 |
| **Mean** | 0.9053 | 0.9169 | +0.0116 | 1.777 | 0.835 | -0.942 | 0.8198 | 0.8421 | +0.0222 | 33.48 | 36.25 | +2.77 |

### Varian

| Scan | Dice FiLM | Dice NoFiLM | Δ Dice | 3D FiLM | 3D NoFiLM | Δ 3D | SSIM FiLM | SSIM NoFiLM | Δ SSIM | PSNR FiLM | PSNR NoFiLM | Δ PSNR |
|------|-----------|-------------|--------|---------|-----------|------|-----------|-------------|--------|-----------|-------------|--------|
| `CV_P1_V_01_fdk` | 0.8859 | 0.8861 | +0.0002 | 0.987 | 0.857 | -0.130 | 0.7173 | 0.7036 | -0.0137 | 34.52 | 35.30 | +0.78 |
| `CV_P2_V_01_fdk` | 0.9171 | 0.9167 | -0.0004 | 0.747 | 0.454 | -0.293 | 0.7296 | 0.7450 | +0.0154 | 34.21 | 38.63 | +4.42 |
| `CV_P3_V_01_fdk` | 0.8571 | 0.8541 | -0.0030 | 0.701 | 0.791 | +0.091 | 0.8719 | 0.8826 | +0.0107 | 53.02 | 51.92 | -1.10 |
| `CV_P4_V_01_fdk` | 0.8787 | 0.8787 | +0.0000 | 0.472 | 0.545 | +0.073 | 0.8082 | 0.8062 | -0.0020 | 56.94 | 56.19 | -0.75 |
| `CV_P5_V_01_fdk` | 0.9236 | 0.9323 | +0.0086 | 1.415 | 0.549 | -0.866 | 0.8348 | 0.8431 | +0.0084 | 43.12 | 48.29 | +5.17 |
| **Mean** | 0.8925 | 0.8936 | +0.0011 | 0.864 | 0.639 | -0.225 | 0.7923 | 0.7961 | +0.0037 | 44.36 | 46.06 | +1.70 |

### All FDK

| Scan | Dice FiLM | Dice NoFiLM | Δ Dice | 3D FiLM | 3D NoFiLM | Δ 3D | SSIM FiLM | SSIM NoFiLM | Δ SSIM | PSNR FiLM | PSNR NoFiLM | Δ PSNR |
|------|-----------|-------------|--------|---------|-----------|------|-----------|-------------|--------|-----------|-------------|--------|
| `CE_P1_V_01_fdk` | 0.8776 | 0.8883 | +0.0106 | 1.286 | 0.675 | -0.612 | 0.8395 | 0.8664 | +0.0269 | 34.79 | 36.19 | +1.40 |
| `CE_P2_V_01_fdk` | 0.9207 | 0.9369 | +0.0161 | 2.271 | 0.441 | -1.830 | 0.8036 | 0.8537 | +0.0501 | 30.02 | 36.62 | +6.60 |
| `CE_P3_V_01_fdk` | 0.8877 | 0.9090 | +0.0213 | 2.606 | 0.712 | -1.894 | 0.8578 | 0.9204 | +0.0625 | 31.14 | 36.80 | +5.66 |
| `CE_P4_V_01_fdk` | 0.9453 | 0.9456 | +0.0003 | 1.252 | 1.195 | -0.057 | 0.7732 | 0.7567 | -0.0166 | 31.34 | 31.73 | +0.39 |
| `CE_P5_V_01_fdk` | 0.8952 | 0.9048 | +0.0096 | 1.472 | 1.155 | -0.317 | 0.8249 | 0.8131 | -0.0118 | 40.11 | 39.90 | -0.21 |
| `CV_P1_V_01_fdk` | 0.8859 | 0.8861 | +0.0002 | 0.987 | 0.857 | -0.130 | 0.7173 | 0.7036 | -0.0137 | 34.52 | 35.30 | +0.78 |
| `CV_P2_V_01_fdk` | 0.9171 | 0.9167 | -0.0004 | 0.747 | 0.454 | -0.293 | 0.7296 | 0.7450 | +0.0154 | 34.21 | 38.63 | +4.42 |
| `CV_P3_V_01_fdk` | 0.8571 | 0.8541 | -0.0030 | 0.701 | 0.791 | +0.091 | 0.8719 | 0.8826 | +0.0107 | 53.02 | 51.92 | -1.10 |
| `CV_P4_V_01_fdk` | 0.8787 | 0.8787 | +0.0000 | 0.472 | 0.545 | +0.073 | 0.8082 | 0.8062 | -0.0020 | 56.94 | 56.19 | -0.75 |
| `CV_P5_V_01_fdk` | 0.9236 | 0.9323 | +0.0086 | 1.415 | 0.549 | -0.866 | 0.8348 | 0.8431 | +0.0084 | 43.12 | 48.29 | +5.17 |
| **Mean** | 0.8989 | 0.9052 | +0.0063 | 1.321 | 0.737 | -0.584 | 0.8061 | 0.8191 | +0.0130 | 38.92 | 41.16 | +2.24 |

---

## Full-sweep 5×2 upscale videos (No-FiLM)

Layout: source/target proj → pred/GT DVF @128 (body-masked) → sub_CT / native CT → upsampled pred/GT DVF → warped @128 / warped native (+ PTV).

| Scan | Frames | 5×2 full-res upscale video |
|------|--------|----------------------------|
| `CE_P1_V_01_fdk` | 340 | `results/CE_P1_V_01_fdk/videos_nofilm/CE_P1_V_01_fdk_fullres_upscale_sagittal67_sweep340_nofilm.mp4` |
| `CE_P2_V_01_fdk` | 340 | `results/CE_P2_V_01_fdk/videos_nofilm/CE_P2_V_01_fdk_fullres_upscale_sagittal67_sweep340_nofilm.mp4` |
| `CE_P3_V_01_fdk` | 340 | `results/CE_P3_V_01_fdk/videos_nofilm/CE_P3_V_01_fdk_fullres_upscale_sagittal63_sweep340_nofilm.mp4` |
| `CE_P4_V_01_fdk` | 340 | `results/CE_P4_V_01_fdk/videos_nofilm/CE_P4_V_01_fdk_fullres_upscale_sagittal42_sweep340_nofilm.mp4` |
| `CE_P5_V_01_fdk` | 340 | `results/CE_P5_V_01_fdk/videos_nofilm/CE_P5_V_01_fdk_fullres_upscale_sagittal62_sweep340_nofilm.mp4` |
| `CV_P1_V_01_fdk` | 680 | `results/CV_P1_V_01_fdk/videos_nofilm/CV_P1_V_01_fdk_fullres_upscale_sagittal60_sweep680_nofilm.mp4` |
| `CV_P2_V_01_fdk` | 680 | `results/CV_P2_V_01_fdk/videos_nofilm/CV_P2_V_01_fdk_fullres_upscale_sagittal68_sweep680_nofilm.mp4` |
| `CV_P3_V_01_fdk` | 680 | `results/CV_P3_V_01_fdk/videos_nofilm/CV_P3_V_01_fdk_fullres_upscale_sagittal60_sweep680_nofilm.mp4` |
| `CV_P4_V_01_fdk` | 680 | `results/CV_P4_V_01_fdk/videos_nofilm/CV_P4_V_01_fdk_fullres_upscale_sagittal60_sweep680_nofilm.mp4` |
| `CV_P5_V_01_fdk` | 680 | `results/CV_P5_V_01_fdk/videos_nofilm/CV_P5_V_01_fdk_fullres_upscale_sagittal75_sweep680_nofilm.mp4` |

Export:

```bash
bash scripts/run_fdk_nofilm_sweep_upscale_viz.sh
# or one scan:
$LEARN_GUI_ROOT/.venv/bin/python scripts/export_fullres_upscale_mp4.py \
  --scan-id CE_P1_V_01_fdk --gpu 0 --source sweep --plane sagittal
```

---

## Metric definitions

| Metric | Definition | Better |
|--------|------------|--------|
| **Dice** | Soft PTV overlap after warping with pred vs GT DVF | Higher |
| **3D Error (mm)** | ‖centroid shift_pred − centroid shift_GT‖₂ | Lower |
| **SSIM / PSNR** | Body-masked similarity of pred-warped vs GT-warped CT | Higher |
| **neg det(J)** | Fraction of voxels with negative Jacobian determinant | Lower (~0) |

---

## Related

- FiLM vs No-FiLM (sweep + stratified full-res): `FDK_FILM_VS_NOFILM.md`
- No-FiLM stratified full-res upscale quantification: `NOFILM_FDK_FULLRES_UPSCALE.md`
- Sweep eval runner: `scripts/run_sweep_eval_masked.py` / `ml/sweep_evaluator.py`
