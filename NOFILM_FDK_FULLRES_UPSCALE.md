# No-FiLM FDK — Full-Resolution DVF Upscale Quantification

**Scope:** No-FiLM FDK models only (`checkpoints_nofilm/best.pt`).  
**Cohort:** 10 clinical FDK4D scans — Elekta `CE_P1`–`P5`, Varian `CV_P1`–`P5`.  
**Script:** `scripts/run_fullres_dvf_eval.py --no-film`  
**Artifacts:** `results/<scan>_fdk/fullres_vs_128_metrics_nofilm.json`

This note quantifies the **train/infer @128³ → upsample DVF → warp native CT/PTV** strategy for No-FiLM. FiLM comparisons live in `FDK_FILM_VS_NOFILM.md`.

---

## Why quantify this

Training and inference stay cheap on a **128³** grid. Clinical CTs are larger (Elekta ~270×256×270, Varian ~450×220×450). If an upsampled DVF still warps the native PTV well, we get a practical deployment path without native-resolution training.

---

## How quantification works

Paired **same-sample** A/B test (90 stratified train-pair samples per scan):

1. Predict DVF once at **128³** (No-FiLM checkpoint, DRRs + 128³ source volume).
2. Score metrics at 128³: warp 128³ PTV/CT with **pred vs GT** DVF.
3. Upsample both pred and GT DVFs to native size (`ml/flow_utils.upsample_dvf`):
   - `F.interpolate(..., mode="trilinear", align_corners=True)`
   - then × `(Dn/D, Hn/H, Wn/W)` — displacements are in **voxel units**
4. Score the **same** metrics on **native** `CT_06` / PTV / body with the upsampled DVFs.
5. Report means and **Δ = full-res − 128** (positive Dice/SSIM/PSNR = upscale helped; negative 3D error = upscale helped).

```
DRR @128 ──► VoxelMap (No-FiLM) ──► DVF @128³
                                         │
                          F.interpolate (trilinear)
                                         │
                               × (Dn/D, Hn/H, Wn/W)
                                         │
                                         ▼
                               DVF @ native grid
                                         │
                          warp native CT / PTV / body
                                         │
                                         ▼
                    Dice / 3D err / PSNR / SSIM  vs  same @128
```

### Metrics used for quantification

| Metric | Definition (per sample) | Better |
|--------|-------------------------|--------|
| **Dice** | Soft overlap of PTV warped by pred DVF vs GT DVF | Higher |
| **3D Error (mm)** | ‖centroid shift_pred − centroid shift_GT‖₂ | Lower |
| **SSIM / PSNR** | Body-masked similarity of pred-warped vs GT-warped CT | Higher |
| **GT-shift MAE (mm)** | Sanity: \|GT centroid shift @full − @128\| (mean over samples) | Lower (~1 mm OK) |

**Primary upscale evidence:** Δ Dice and Δ 3D error. Image metrics are secondary on anisotropic native grids.

---

## Executive results (all 10 No-FiLM FDK)

| | @128³ | @full-res | Δ (full−128) |
|--|-------|-----------|--------------|
| Mean Dice | 0.9046 | **0.9549** | **+0.0504** |
| Mean 3D err (mm) | 0.464 | **0.414** | **-0.050** |
| Mean SSIM | 0.8248 | 0.7849 | -0.0399 |
| Mean PSNR (dB) | 28.13 | 33.32 | +5.19 |
| Mean GT-shift MAE (mm) | — | — | 0.747 |

Patients where **full-res beats @128** (of 10):

| Metric | Full-res wins |
|--------|---------------|
| Dice (Δ > 0) | 10/10 |
| 3D Error (Δ < 0) | 5/10 |
| SSIM | 2/10 |
| PSNR | 7/10 |

**Takeaway:** For No-FiLM FDK, DVF upscale **improves PTV Dice on all 10 patients** (mean ≈ +0.05). 3D error improves on average; SSIM/PSNR are mixed. Deployment recipe: No-FiLM @128 → upsample+scale DVF → warp native PTV/CT.

---

## Elekta No-FiLM (`CE_*_fdk`)

| Scan | Dice @128 | Dice @full | Δ Dice | 3D @128 | 3D @full | Δ 3D | SSIM @128 | SSIM @full | Δ SSIM | PSNR @128 | PSNR @full | Δ PSNR | GT-shift MAE |
|------|-----------|------------|--------|---------|----------|------|-----------|------------|--------|-----------|------------|--------|--------------|
| `CE_P1_V_01_fdk` | 0.8864 | 0.9406 | +0.0542 | 0.303 | 0.304 | +0.001 | 0.8721 | 0.8785 | +0.0065 | 28.34 | 29.67 | +1.33 | 1.373 |
| `CE_P2_V_01_fdk` | 0.9376 | 0.9673 | +0.0297 | 0.178 | 0.179 | +0.001 | 0.8692 | 0.8374 | -0.0318 | 29.35 | 28.01 | -1.34 | 0.176 |
| `CE_P3_V_01_fdk` | 0.9092 | 0.9556 | +0.0464 | 0.346 | 0.327 | -0.019 | 0.9307 | 0.9186 | -0.0121 | 28.84 | 32.98 | +4.14 | 0.413 |
| `CE_P4_V_01_fdk` | 0.9435 | 0.9646 | +0.0211 | 0.761 | 0.488 | -0.273 | 0.7629 | 0.6643 | -0.0987 | 28.47 | 19.49 | -8.98 | 0.430 |
| `CE_P5_V_01_fdk` | 0.8982 | 0.9354 | +0.0372 | 0.846 | 0.805 | -0.041 | 0.8130 | 0.8408 | +0.0277 | 33.92 | 29.53 | -4.39 | 0.507 |
| **Mean** | 0.9150 | 0.9527 | +0.0377 | 0.487 | 0.421 | -0.066 | 0.8496 | 0.8279 | -0.0217 | 29.78 | 27.94 | -1.85 | 0.580 |

## Varian No-FiLM (`CV_*_fdk`)

| Scan | Dice @128 | Dice @full | Δ Dice | 3D @128 | 3D @full | Δ 3D | SSIM @128 | SSIM @full | Δ SSIM | PSNR @128 | PSNR @full | Δ PSNR | GT-shift MAE |
|------|-----------|------------|--------|---------|----------|------|-----------|------------|--------|-----------|------------|--------|--------------|
| `CV_P1_V_01_fdk` | 0.8865 | 0.9552 | +0.0687 | 0.728 | 0.384 | -0.344 | 0.7098 | 0.6083 | -0.1015 | 25.50 | 28.38 | +2.88 | 1.080 |
| `CV_P2_V_01_fdk` | 0.9155 | 0.9661 | +0.0506 | 0.222 | 0.388 | +0.166 | 0.7518 | 0.6847 | -0.0671 | 26.79 | 33.12 | +6.33 | 1.203 |
| `CV_P3_V_01_fdk` | 0.8608 | 0.9410 | +0.0802 | 0.530 | 0.559 | +0.029 | 0.8811 | 0.8309 | -0.0502 | 27.03 | 47.04 | +20.01 | 0.496 |
| `CV_P4_V_01_fdk` | 0.8776 | 0.9526 | +0.0749 | 0.414 | 0.255 | -0.159 | 0.8095 | 0.7828 | -0.0267 | 25.91 | 46.00 | +20.09 | 0.629 |
| `CV_P5_V_01_fdk` | 0.9303 | 0.9709 | +0.0405 | 0.308 | 0.449 | +0.140 | 0.8479 | 0.8029 | -0.0449 | 27.13 | 38.98 | +11.85 | 1.159 |
| **Mean** | 0.8942 | 0.9571 | +0.0630 | 0.440 | 0.407 | -0.034 | 0.8000 | 0.7419 | -0.0581 | 26.47 | 38.70 | +12.23 | 0.913 |

## All No-FiLM FDK

| Scan | Dice @128 | Dice @full | Δ Dice | 3D @128 | 3D @full | Δ 3D | SSIM @128 | SSIM @full | Δ SSIM | PSNR @128 | PSNR @full | Δ PSNR | GT-shift MAE |
|------|-----------|------------|--------|---------|----------|------|-----------|------------|--------|-----------|------------|--------|--------------|
| `CE_P1_V_01_fdk` | 0.8864 | 0.9406 | +0.0542 | 0.303 | 0.304 | +0.001 | 0.8721 | 0.8785 | +0.0065 | 28.34 | 29.67 | +1.33 | 1.373 |
| `CE_P2_V_01_fdk` | 0.9376 | 0.9673 | +0.0297 | 0.178 | 0.179 | +0.001 | 0.8692 | 0.8374 | -0.0318 | 29.35 | 28.01 | -1.34 | 0.176 |
| `CE_P3_V_01_fdk` | 0.9092 | 0.9556 | +0.0464 | 0.346 | 0.327 | -0.019 | 0.9307 | 0.9186 | -0.0121 | 28.84 | 32.98 | +4.14 | 0.413 |
| `CE_P4_V_01_fdk` | 0.9435 | 0.9646 | +0.0211 | 0.761 | 0.488 | -0.273 | 0.7629 | 0.6643 | -0.0987 | 28.47 | 19.49 | -8.98 | 0.430 |
| `CE_P5_V_01_fdk` | 0.8982 | 0.9354 | +0.0372 | 0.846 | 0.805 | -0.041 | 0.8130 | 0.8408 | +0.0277 | 33.92 | 29.53 | -4.39 | 0.507 |
| `CV_P1_V_01_fdk` | 0.8865 | 0.9552 | +0.0687 | 0.728 | 0.384 | -0.344 | 0.7098 | 0.6083 | -0.1015 | 25.50 | 28.38 | +2.88 | 1.080 |
| `CV_P2_V_01_fdk` | 0.9155 | 0.9661 | +0.0506 | 0.222 | 0.388 | +0.166 | 0.7518 | 0.6847 | -0.0671 | 26.79 | 33.12 | +6.33 | 1.203 |
| `CV_P3_V_01_fdk` | 0.8608 | 0.9410 | +0.0802 | 0.530 | 0.559 | +0.029 | 0.8811 | 0.8309 | -0.0502 | 27.03 | 47.04 | +20.01 | 0.496 |
| `CV_P4_V_01_fdk` | 0.8776 | 0.9526 | +0.0749 | 0.414 | 0.255 | -0.159 | 0.8095 | 0.7828 | -0.0267 | 25.91 | 46.00 | +20.09 | 0.629 |
| `CV_P5_V_01_fdk` | 0.9303 | 0.9709 | +0.0405 | 0.308 | 0.449 | +0.140 | 0.8479 | 0.8029 | -0.0449 | 27.13 | 38.98 | +11.85 | 1.159 |
| **Mean** | 0.9046 | 0.9549 | +0.0504 | 0.464 | 0.414 | -0.050 | 0.8248 | 0.7849 | -0.0399 | 28.13 | 33.32 | +5.19 | 0.747 |

---

## Native grids

| Scan | Native size (D×H×W) |
|------|---------------------|
| `CE_P1_V_01_fdk` | 270×256×270 |
| `CE_P2_V_01_fdk` | 270×256×270 |
| `CE_P3_V_01_fdk` | 270×256×270 |
| `CE_P4_V_01_fdk` | 270×256×270 |
| `CE_P5_V_01_fdk` | 270×256×270 |
| `CV_P1_V_01_fdk` | 450×220×450 |
| `CV_P2_V_01_fdk` | 450×220×450 |
| `CV_P3_V_01_fdk` | 450×220×450 |
| `CV_P4_V_01_fdk` | 450×220×450 |
| `CV_P5_V_01_fdk` | 450×220×450 |

Approx scale from 128³: Elekta ≈ (2.11, 2.00, 2.11); Varian ≈ (3.52, 1.72, 3.52).

---

## Caveats

- Stratified **train-pair** samples (not a held-out breathing sweep) — answers “does upscale preserve warp quality?” more than “unseen-phase generalization.”
- GT reference at native is the **upsampled GT DVF**, not a separately registered native Elastix field (except that GT-shift MAE checks consistency of that scaling).
- n = 10 patients; no significance tests.

---

## Reproduce

```bash
export VOXELMAP_CLINICAL_ROOT=/home/abhishek/Documents/VoxelMap_Clinical
export LEARN_GUI_ROOT=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python

FDK_FULLRES_MODE=nofilm bash scripts/run_fdk_fullres_eval.sh
# or one scan:
$LEARN_GUI_ROOT/.venv/bin/python scripts/run_fullres_dvf_eval.py \
  --scan-id CE_P1_V_01_fdk --gpu 0 --max-samples 90 --no-film
```

## Related

- FiLM vs No-FiLM (sweep + full-res): `FDK_FILM_VS_NOFILM.md`
- GTVol Phase-1 pilot: `FULLRES_DVF_UPSAMPLE.md`
- Plan: `SUBSAMPLE_TRAIN_FULLRES_EVAL_PLAN.md`
