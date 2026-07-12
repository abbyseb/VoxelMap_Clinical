# FDK FiLM vs No-FiLM — Training, Sweep Eval, and Full-Res DVF Upscale

**Cohort:** 10 clinical FDK4D scans — Elekta `CE_P1`–`CE_P5`, Varian `CV_P1`–`CV_P5` (`*_fdk`).  
**Models:** concatenated VoxelMap, 50 epochs, identical FDK training pairs; only FiLM on/off differs.  
**Date of full-res batch:** 2026-07-12 / 2026-07-13.

This note is the single place that records **what was run**, **why**, **how metrics were computed**, and **all comparison tables** (breathing-sweep @128 and native-resolution DVF upscale for both FiLM and No-FiLM).

---

## Why this matters

1. **FiLM ablation on FDK.** Feature-wise Linear Modulation conditions the network on breathing / acquisition context. On FDK reconstructions (noisier / different contrast than GT volumes), it is not obvious that FiLM helps. A fair paired comparison decides whether to keep it for clinical FDK pipelines.
2. **Train cheap, evaluate dense.** Training and inference stay on a **128³** grid (fast DRRs + VoxelMap). Clinical CTs are much larger (Elekta ~270×256×270, Varian ~450×220×450). Upsampling the **DVF** (not re-training at native res) tests whether a low-res motion field can still warp high-res anatomy usefully.
3. **Target-relevant metrics.** PTV Dice and 3D centroid error speak to localization / tracking; body-masked PSNR/SSIM speak to image fidelity after warp. Sweep eval stresses the full breathing range; full-res eval stresses grid scaling.
4. **Deployment path.** If No-FiLM + DVF upscale wins, the production recipe is: train No-FiLM @128 on FDK → predict DVF @128 → trilinear upsample + displacement scale → warp native CT/PTV.

---

## What was done (checklist)

| Step | Status | Notes |
|------|--------|-------|
| Stage FDK volumes + masks for CE/CV P1–P5 | Done | `scripts/stage_fdk_scan.py`, native `CT_06.mha` under `runs/<scan>_fdk/` |
| Train FiLM-on FDK models (50 ep) | Done | `runs/<scan>_fdk/checkpoints/best.pt` |
| Breathing-sweep eval, FiLM, body-masked | Done | `results/<scan>_fdk/sweep_metrics.json` |
| Train FiLM-off FDK models (50 ep) | Done | `checkpoints_nofilm/best.pt` — does **not** overwrite FiLM |
| Breathing-sweep eval, No-FiLM | Done | `results/<scan>_fdk/sweep_metrics_nofilm.json` |
| Full-res DVF upsample eval, FiLM (90 samples) | Done | `fullres_vs_128_metrics_film.json` |
| Full-res DVF upsample eval, No-FiLM (90 samples) | Done | `fullres_vs_128_metrics_nofilm.json` |
| This comparison document | Done | Sweep + full-res tables + methodology |

**Not** in this experiment: retraining at native resolution; GTVol (non-FDK) FiLM vs No-FiLM; changing loss / architecture beyond `use_film`.

---

## Methodology

### Data and training resolution

- **Source volumes:** FDK4D reconstructions (`FDK4D_*`), staged per patient as `*_fdk`.
- **Train / infer grid:** volumes and DRRs resampled to **128³** / 128×128 (same as the rest of VoxelMap Clinical).
- **Native grid (eval only):** loaded from staged `train/CT_06.mha` (and matching PTV/body masks).
  - Elekta FDK: **270 × 256 × 270**
  - Varian FDK: **450 × 220 × 450**
- **Spacing:** metrics that need mm use the spacing appropriate to the grid (train spacing @128; native treated as 1 mm/voxel in the full-res path where applicable — see `run_fullres_dvf_eval.py`).

### FiLM on vs off

| Variant | Flag / code | Checkpoint | Eval outputs |
|---------|-------------|------------|--------------|
| **FiLM** | default (`use_film=True`) | `runs/<scan>_fdk/checkpoints/best.pt` | `sweep_metrics.json`, `fullres_vs_128_metrics_film.json` |
| **No-FiLM** | `--no-film` | `runs/<scan>_fdk/checkpoints_nofilm/best.pt` | `sweep_metrics_nofilm.json`, `fullres_vs_128_metrics_nofilm.json` |

Both variants:

- Same FDK training pairs and epoch count (50).
- Same concatenated VoxelMap backbone.
- Independent weight files so FiLM runs are never overwritten.

Batch helpers: `scripts/run_fdk_nofilm_batch.sh`, `scripts/run_fdk_nofilm_eval.sh`, `scripts/run_fdk_fullres_eval.sh`.

### Breathing-sweep evaluation (@128)

- Model predicts DVFs over a stratified / sweep set of breathing states on the **128³** training grid.
- Warps PTV and CT; reports Dice, 3D PTV centroid error, body-masked SSIM/PSNR.
- This is the primary “clinical sweep” comparison already published for FDK FiLM; No-FiLM uses the same protocol with `--no-film`.

### Full-resolution evaluation via DVF upsampling

Goal: keep inference @128, apply motion on the **native** CT/PTV.

```
DRR @128 ──► VoxelMap (±FiLM) ──► DVF @128³
                                      │
                       F.interpolate (trilinear, align_corners=True)
                                      │
                            × (Dn/D, Hn/H, Wn/W)   ← required
                                      │
                                      ▼
                            DVF @ native grid
                                      │
                       warp native CT / PTV / body
                                      │
                                      ▼
                      Dice / 3D err / PSNR / SSIM
                   (same samples also scored @128)
```

**Critical detail — interpolate then scale.** DVFs are in **voxel-index units** (added to the sampling grid). Upsampling the field without scaling under-warps on a finer grid:

```python
# ml/flow_utils.py — upsample_dvf
flow_up = F.interpolate(flow_128, size=native_size, mode="trilinear", align_corners=True)
flow_up = flow_up * scale   # per-axis: native_size[i] / 128
```

Other conventions (must match training):

- Integrate (`VecInt`) at 128³ **then** upsample.
- Warp **native** masks loaded from MHAs (do not trilinearly upsample binary masks as soft fields for Dice).
- **90 stratified train-pair samples** per scan; identical sample set for @128 and @full-res within a run.
- Each JSON reports `gt_shift_consistency_mae_mm` (GT DVF 128→native vs native GT) as a scale sanity check (~1 mm MAE typical).

Scale factors from 128³:

| Cohort | Native (D×H×W) | Approx scale |
|--------|----------------|--------------|
| Elekta FDK | 270×256×270 | ≈ (2.11, 2.00, 2.11) |
| Varian FDK | 450×220×450 | ≈ (3.52, 1.72, 3.52) |

### Metrics — definitions

All reported values are **means over samples** (sweep projections or 90 stratified train-pairs). For each sample the model predicts a DVF; we warp the **source** PTV / CT with the **predicted** DVF and with the **GT** DVF, then compare those two warps (not raw CT vs warped CT).

| Metric | Better | What it measures |
|--------|--------|------------------|
| **Dice** | Higher (→ 1) | Soft overlap of predicted-warped PTV vs GT-warped PTV |
| **3D Error (mm)** | Lower (→ 0) | Euclidean mismatch of PTV centroid *shifts* (pred vs GT) |
| **SSIM** | Higher (→ 1) | Structural similarity of warped CTs inside body mask |
| **PSNR (dB)** | Higher | Peak signal-to-noise of warped CTs inside body mask |
| **MSE** | Lower | Mean squared intensity error of warped CTs inside body (logged in JSON; not in summary tables) |
| **GT shift consistency MAE (mm)** | Lower | Full-res sanity: how well upsampled GT DVF matches native GT warp centroid shift |

**Δ convention throughout:** **No-FiLM − FiLM**  
- Positive Dice / SSIM / PSNR → No-FiLM better  
- Negative 3D error → No-FiLM better  

For within-model upscale tables: **Δ = full-res − 128** (same sign convention per metric).

#### Dice (PTV overlap)

Soft Dice between the PTV mask warped by the **predicted** DVF and the same source PTV warped by the **GT** DVF (`losses.dice` in LEARN-GUI):

\[
\mathrm{Dice} = \frac{2\sum_i p_i\, g_i}{\sum_i p_i + \sum_i g_i}
\]

- \(p\) = predicted-warped PTV, \(g\) = GT-warped PTV (continuous after spatial transform).
- Range ideally **0–1**; **1** = identical overlap.
- Answers: “Does the predicted motion put the PTV where the GT motion puts it?”
- Implementation: `ml/utilities/losses.py` → `class dice`.

#### 3D Error (PTV centroid, mm)

1. Compute LR / SI / AP **centroid shift** of the warped PTV relative to the **source** PTV, in mm (`centroid_shift_mm`, using voxel spacing).
2. Do this for both predicted and GT warps → shifts \(\mathbf{s}_\mathrm{pred}\) and \(\mathbf{s}_\mathrm{GT}\).
3. **3D Error** = \(\lVert \mathbf{s}_\mathrm{pred} - \mathbf{s}_\mathrm{GT} \rVert_2\) in mm.

- Lower means the predicted motion moves the PTV center as far (and in the same direction) as the GT motion.
- Sensitive to anisotropic spacing and residual interpolation when upscaling DVFs.
- Axes: LR (left–right), SI (superior–inferior), AP (anterior–posterior).

#### Body mask

Image metrics (MSE / PSNR / SSIM) are restricted to voxels inside the patient **body / abdomen** mask (`ml/mask_utils.py`: `*Body*`, `*Abdomen*`). Air / couch outside the body is ignored so background zeros do not inflate scores.

#### MSE (mean squared error)

Over body voxels only:

\[
\mathrm{MSE} = \frac{1}{N}\sum_{i \in \mathrm{body}} (I^\mathrm{pred}_i - I^\mathrm{GT}_i)^2
\]

where \(I^\mathrm{pred}\) / \(I^\mathrm{GT}\) are CT volumes warped by predicted / GT DVFs (intensities typically normalized to ~[0, 1]). Stored as `mean_mse` in full-res JSONs.

#### PSNR (peak signal-to-noise ratio, dB)

On the same body-masked intensities (`skimage.metrics.peak_signal_noise_ratio`, `data_range=1.0`):

\[
\mathrm{PSNR} = 10\log_{10}\!\left(\frac{\mathrm{data\_range}^2}{\mathrm{MSE}}\right)
\]

with \(\mathrm{data\_range}=1\). Higher dB = closer warped images. Fallback if skimage missing: \(10\log_{10}(1/(\mathrm{MSE}+10^{-8}))\).

#### SSIM (structural similarity index)

Structural similarity between predicted-warped and GT-warped CT volumes (`skimage.metrics.structural_similarity`, `data_range=1.0`, **`mask=` body** when available).

- Combines local luminance, contrast, and structure (windowed comparison).
- Range typically **0–1** (can be slightly outside for edge cases); **1** = identical structure.
- More perceptually aligned than MSE/PSNR alone; still secondary to Dice/3D for “did the target move correctly?”

#### GT shift consistency MAE (full-res only)

Sanity check that DVF upsampling + displacement scaling is in the right ballpark: compare the 3D centroid shift implied by the **upsampled GT DVF** on the native grid vs the same GT warp path @128 (reported as `gt_shift_consistency_mae_mm` in full-res JSONs). Typical values ~0.8–1.4 mm; large values would flag a scaling bug.

#### What is *not* a primary table metric here

| Item | Role |
|------|------|
| Jacobian / folding (`det(J)≤0`) | Used in some GTVol evals; **not** tabulated in this FiLM vs No-FiLM note |
| Training loss | Optimization objective only; see per-run `loss_history.json` |
| Projection-space SSIM/PSNR | Not used; all image metrics are **3D volume** after warp |

### How to reproduce

```bash
export VOXELMAP_CLINICAL_ROOT=/home/abhishek/Documents/VoxelMap_Clinical
export LEARN_GUI_ROOT=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python

# Sweep No-FiLM (example batch)
bash scripts/run_fdk_nofilm_eval.sh

# Full-res FiLM + No-FiLM for all FDK scans
FDK_FULLRES_MODE=both FDK_GPU=0 bash scripts/run_fdk_fullres_eval.sh

# Single scan
$LEARN_GUI_ROOT/.venv/bin/python scripts/run_fullres_dvf_eval.py \
  --scan-id CE_P1_V_01_fdk --gpu 0 --max-samples 90
$LEARN_GUI_ROOT/.venv/bin/python scripts/run_fullres_dvf_eval.py \
  --scan-id CE_P1_V_01_fdk --gpu 0 --max-samples 90 --no-film
```

---

## Executive takeaways

### Sweep @128 (full breathing sweep)

| | Mean Dice | Mean 3D err (mm) | Mean SSIM | Mean PSNR (dB) |
|--|-----------|------------------|-----------|----------------|
| FiLM | 0.8989 | 1.321 | 0.8061 | 38.92 |
| No-FiLM | **0.9052** | **0.737** | **0.8191** | **41.16** |
| Δ (NoFiLM−FiLM) | **+0.0063** | **−0.584** | **+0.0130** | **+2.24** |

No-FiLM wins on **8/10** Dice, **8/10** 3D error, **6/10** SSIM, **7/10** PSNR. Gains are larger on Elekta than Varian.

### Full-res DVF upscale (90 samples / scan)

| | Mean Dice @full-res | Mean 3D @full-res | Mean Dice gain (full−128) |
|--|---------------------|-------------------|---------------------------|
| FiLM | 0.9441 | 0.788 mm | +0.0467 |
| No-FiLM | **0.9549** | **0.414 mm** | **+0.0504** |
| Δ NoFiLM−FiLM @full-res | **+0.0108** | **−0.374 mm** | — |

At native resolution No-FiLM wins **9/10** Dice and **9/10** 3D error. Upsampling helps PTV Dice for **both** models (~+0.05 mean). Image metrics (SSIM/PSNR) are mixed when moving 128→native (anisotropy + interpolation); Dice is the cleanest “does the upscale work?” signal.

**Practical recommendation for FDK clinical runs:** prefer **No-FiLM** checkpoints; use **DVF trilinear upsample + per-axis scale** when warping native CT/PTV.

---

## Part A — Breathing-sweep metrics (@128)

### Clinical Elekta FDK (`CE_*_fdk`)

| Scan | Dice FiLM | Dice NoFiLM | Δ Dice | 3D Err FiLM | 3D Err NoFiLM | Δ 3D | SSIM FiLM | SSIM NoFiLM | Δ SSIM | PSNR FiLM | PSNR NoFiLM | Δ PSNR |
|------|-----------|-------------|--------|-------------|---------------|------|-----------|-------------|--------|-----------|-------------|--------|
| `CE_P1_V_01_fdk` | 0.8776 | 0.8883 | +0.0106 | 1.286 | 0.675 | -0.612 | 0.8395 | 0.8664 | +0.0269 | 34.79 | 36.19 | +1.40 |
| `CE_P2_V_01_fdk` | 0.9207 | 0.9369 | +0.0161 | 2.271 | 0.441 | -1.830 | 0.8036 | 0.8537 | +0.0501 | 30.02 | 36.62 | +6.60 |
| `CE_P3_V_01_fdk` | 0.8877 | 0.9090 | +0.0213 | 2.606 | 0.712 | -1.894 | 0.8578 | 0.9204 | +0.0625 | 31.14 | 36.80 | +5.66 |
| `CE_P4_V_01_fdk` | 0.9453 | 0.9456 | +0.0003 | 1.252 | 1.195 | -0.057 | 0.7732 | 0.7567 | -0.0166 | 31.34 | 31.73 | +0.39 |
| `CE_P5_V_01_fdk` | 0.8952 | 0.9048 | +0.0096 | 1.472 | 1.155 | -0.317 | 0.8249 | 0.8131 | -0.0118 | 40.11 | 39.90 | -0.21 |
| **Mean** | 0.9053 | 0.9169 | +0.0116 | 1.777 | 0.835 | -0.942 | 0.8198 | 0.8421 | +0.0222 | 33.48 | 36.25 | +2.77 |

### Clinical Varian FDK (`CV_*_fdk`)

| Scan | Dice FiLM | Dice NoFiLM | Δ Dice | 3D Err FiLM | 3D Err NoFiLM | Δ 3D | SSIM FiLM | SSIM NoFiLM | Δ SSIM | PSNR FiLM | PSNR NoFiLM | Δ PSNR |
|------|-----------|-------------|--------|-------------|---------------|------|-----------|-------------|--------|-----------|-------------|--------|
| `CV_P1_V_01_fdk` | 0.8859 | 0.8861 | +0.0002 | 0.987 | 0.857 | -0.130 | 0.7173 | 0.7036 | -0.0137 | 34.52 | 35.30 | +0.78 |
| `CV_P2_V_01_fdk` | 0.9171 | 0.9167 | -0.0004 | 0.747 | 0.454 | -0.293 | 0.7296 | 0.7450 | +0.0154 | 34.21 | 38.63 | +4.42 |
| `CV_P3_V_01_fdk` | 0.8571 | 0.8541 | -0.0030 | 0.701 | 0.791 | +0.091 | 0.8719 | 0.8826 | +0.0107 | 53.02 | 51.92 | -1.10 |
| `CV_P4_V_01_fdk` | 0.8787 | 0.8787 | +0.0000 | 0.472 | 0.545 | +0.073 | 0.8082 | 0.8062 | -0.0020 | 56.94 | 56.19 | -0.75 |
| `CV_P5_V_01_fdk` | 0.9236 | 0.9323 | +0.0086 | 1.415 | 0.549 | -0.866 | 0.8348 | 0.8431 | +0.0084 | 43.12 | 48.29 | +5.17 |
| **Mean** | 0.8925 | 0.8936 | +0.0011 | 0.864 | 0.639 | -0.225 | 0.7923 | 0.7961 | +0.0037 | 44.36 | 46.06 | +1.70 |

### All FDK patients (sweep)

| Scan | Dice FiLM | Dice NoFiLM | Δ Dice | 3D Err FiLM | 3D Err NoFiLM | Δ 3D | SSIM FiLM | SSIM NoFiLM | Δ SSIM | PSNR FiLM | PSNR NoFiLM | Δ PSNR |
|------|-----------|-------------|--------|-------------|---------------|------|-----------|-------------|--------|-----------|-------------|--------|
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

### Sweep win counts

Patients where **No-FiLM beats FiLM** (of 10):

| Metric | No-FiLM wins |
|--------|--------------|
| Dice (higher better) | 8/10 |
| 3D Error (lower better) | 8/10 |
| SSIM (higher better) | 6/10 |
| PSNR (higher better) | 7/10 |

---

## Part B — Full-resolution DVF upsample (FiLM and No-FiLM)

Same checkpoints as Part A. For each scan × variant: predict @128 → upsample DVF → score @128 and @native on **90** stratified samples.

Artifacts: `results/<scan>_fdk/fullres_vs_128_metrics_film.json` and `..._nofilm.json`.

### B.1 At 128³ (paired subsample used in the full-res script)

These numbers are **not** identical to Part A (different sample set / protocol), but they are the paired baseline for the upscale Δ.

#### Elekta

| Scan | Dice FiLM | Dice NoFiLM | Δ Dice | 3D Err FiLM | 3D Err NoFiLM | Δ 3D | SSIM FiLM | SSIM NoFiLM | Δ SSIM | PSNR FiLM | PSNR NoFiLM | Δ PSNR |
|------|-----------|-------------|--------|-------------|---------------|------|-----------|-------------|--------|-----------|-------------|--------|
| `CE_P1_V_01_fdk` | 0.8808 | 0.8864 | +0.0055 | 0.642 | 0.303 | -0.338 | 0.8480 | 0.8721 | +0.0241 | 27.69 | 28.34 | +0.64 |
| `CE_P2_V_01_fdk` | 0.9188 | 0.9376 | +0.0188 | 1.470 | 0.178 | -1.292 | 0.8095 | 0.8692 | +0.0597 | 27.04 | 29.35 | +2.31 |
| `CE_P3_V_01_fdk` | 0.8829 | 0.9092 | +0.0263 | 1.620 | 0.346 | -1.274 | 0.8560 | 0.9307 | +0.0747 | 26.60 | 28.84 | +2.24 |
| `CE_P4_V_01_fdk` | 0.9435 | 0.9435 | -0.0000 | 0.812 | 0.761 | -0.051 | 0.7792 | 0.7629 | -0.0162 | 28.72 | 28.47 | -0.25 |
| `CE_P5_V_01_fdk` | 0.8892 | 0.8982 | +0.0091 | 1.008 | 0.846 | -0.162 | 0.8248 | 0.8130 | -0.0117 | 34.29 | 33.92 | -0.37 |
| **Mean** | 0.9031 | 0.9150 | +0.0119 | 1.110 | 0.487 | -0.624 | 0.8235 | 0.8496 | +0.0261 | 28.87 | 29.78 | +0.92 |

#### Varian

| Scan | Dice FiLM | Dice NoFiLM | Δ Dice | 3D Err FiLM | 3D Err NoFiLM | Δ 3D | SSIM FiLM | SSIM NoFiLM | Δ SSIM | PSNR FiLM | PSNR NoFiLM | Δ PSNR |
|------|-----------|-------------|--------|-------------|---------------|------|-----------|-------------|--------|-----------|-------------|--------|
| `CV_P1_V_01_fdk` | 0.8835 | 0.8865 | +0.0031 | 0.914 | 0.728 | -0.186 | 0.7218 | 0.7098 | -0.0121 | 26.13 | 25.50 | -0.63 |
| `CV_P2_V_01_fdk` | 0.9152 | 0.9155 | +0.0003 | 0.529 | 0.222 | -0.307 | 0.7322 | 0.7518 | +0.0196 | 26.55 | 26.79 | +0.24 |
| `CV_P3_V_01_fdk` | 0.8623 | 0.8608 | -0.0015 | 0.454 | 0.530 | +0.076 | 0.8751 | 0.8811 | +0.0061 | 27.00 | 27.03 | +0.03 |
| `CV_P4_V_01_fdk` | 0.8776 | 0.8776 | +0.0000 | 0.336 | 0.414 | +0.078 | 0.8112 | 0.8095 | -0.0017 | 25.81 | 25.91 | +0.11 |
| `CV_P5_V_01_fdk` | 0.9196 | 0.9303 | +0.0107 | 1.347 | 0.308 | -1.039 | 0.8304 | 0.8479 | +0.0175 | 27.05 | 27.13 | +0.08 |
| **Mean** | 0.8917 | 0.8942 | +0.0025 | 0.716 | 0.440 | -0.276 | 0.7941 | 0.8000 | +0.0059 | 26.50 | 26.47 | -0.03 |

#### All FDK

| Scan | Dice FiLM | Dice NoFiLM | Δ Dice | 3D Err FiLM | 3D Err NoFiLM | Δ 3D | SSIM FiLM | SSIM NoFiLM | Δ SSIM | PSNR FiLM | PSNR NoFiLM | Δ PSNR |
|------|-----------|-------------|--------|-------------|---------------|------|-----------|-------------|--------|-----------|-------------|--------|
| `CE_P1_V_01_fdk` | 0.8808 | 0.8864 | +0.0055 | 0.642 | 0.303 | -0.338 | 0.8480 | 0.8721 | +0.0241 | 27.69 | 28.34 | +0.64 |
| `CE_P2_V_01_fdk` | 0.9188 | 0.9376 | +0.0188 | 1.470 | 0.178 | -1.292 | 0.8095 | 0.8692 | +0.0597 | 27.04 | 29.35 | +2.31 |
| `CE_P3_V_01_fdk` | 0.8829 | 0.9092 | +0.0263 | 1.620 | 0.346 | -1.274 | 0.8560 | 0.9307 | +0.0747 | 26.60 | 28.84 | +2.24 |
| `CE_P4_V_01_fdk` | 0.9435 | 0.9435 | -0.0000 | 0.812 | 0.761 | -0.051 | 0.7792 | 0.7629 | -0.0162 | 28.72 | 28.47 | -0.25 |
| `CE_P5_V_01_fdk` | 0.8892 | 0.8982 | +0.0091 | 1.008 | 0.846 | -0.162 | 0.8248 | 0.8130 | -0.0117 | 34.29 | 33.92 | -0.37 |
| `CV_P1_V_01_fdk` | 0.8835 | 0.8865 | +0.0031 | 0.914 | 0.728 | -0.186 | 0.7218 | 0.7098 | -0.0121 | 26.13 | 25.50 | -0.63 |
| `CV_P2_V_01_fdk` | 0.9152 | 0.9155 | +0.0003 | 0.529 | 0.222 | -0.307 | 0.7322 | 0.7518 | +0.0196 | 26.55 | 26.79 | +0.24 |
| `CV_P3_V_01_fdk` | 0.8623 | 0.8608 | -0.0015 | 0.454 | 0.530 | +0.076 | 0.8751 | 0.8811 | +0.0061 | 27.00 | 27.03 | +0.03 |
| `CV_P4_V_01_fdk` | 0.8776 | 0.8776 | +0.0000 | 0.336 | 0.414 | +0.078 | 0.8112 | 0.8095 | -0.0017 | 25.81 | 25.91 | +0.11 |
| `CV_P5_V_01_fdk` | 0.9196 | 0.9303 | +0.0107 | 1.347 | 0.308 | -1.039 | 0.8304 | 0.8479 | +0.0175 | 27.05 | 27.13 | +0.08 |
| **Mean** | 0.8974 | 0.9046 | +0.0072 | 0.913 | 0.464 | -0.450 | 0.8088 | 0.8248 | +0.0160 | 27.69 | 28.13 | +0.44 |

### B.2 At native full resolution

#### Elekta

| Scan | Dice FiLM | Dice NoFiLM | Δ Dice | 3D Err FiLM | 3D Err NoFiLM | Δ 3D | SSIM FiLM | SSIM NoFiLM | Δ SSIM | PSNR FiLM | PSNR NoFiLM | Δ PSNR |
|------|-----------|-------------|--------|-------------|---------------|------|-----------|-------------|--------|-----------|-------------|--------|
| `CE_P1_V_01_fdk` | 0.9360 | 0.9406 | +0.0045 | 0.467 | 0.304 | -0.162 | 0.8595 | 0.8785 | +0.0190 | 27.05 | 29.67 | +2.62 |
| `CE_P2_V_01_fdk` | 0.9405 | 0.9673 | +0.0268 | 1.218 | 0.179 | -1.039 | 0.7630 | 0.8374 | +0.0744 | 26.70 | 28.01 | +1.30 |
| `CE_P3_V_01_fdk` | 0.9084 | 0.9556 | +0.0472 | 1.602 | 0.327 | -1.275 | 0.8628 | 0.9186 | +0.0557 | 32.61 | 32.98 | +0.37 |
| `CE_P4_V_01_fdk` | 0.9641 | 0.9646 | +0.0005 | 0.494 | 0.488 | -0.007 | 0.6790 | 0.6643 | -0.0147 | 22.33 | 19.49 | -2.83 |
| `CE_P5_V_01_fdk` | 0.9303 | 0.9354 | +0.0051 | 0.986 | 0.805 | -0.181 | 0.8515 | 0.8408 | -0.0107 | 31.25 | 29.53 | -1.72 |
| **Mean** | 0.9359 | 0.9527 | +0.0168 | 0.953 | 0.421 | -0.533 | 0.8032 | 0.8279 | +0.0247 | 27.99 | 27.94 | -0.05 |

#### Varian

| Scan | Dice FiLM | Dice NoFiLM | Δ Dice | 3D Err FiLM | 3D Err NoFiLM | Δ 3D | SSIM FiLM | SSIM NoFiLM | Δ SSIM | PSNR FiLM | PSNR NoFiLM | Δ PSNR |
|------|-----------|-------------|--------|-------------|---------------|------|-----------|-------------|--------|-----------|-------------|--------|
| `CV_P1_V_01_fdk` | 0.9535 | 0.9552 | +0.0017 | 0.396 | 0.384 | -0.012 | 0.6060 | 0.6083 | +0.0023 | 28.06 | 28.38 | +0.32 |
| `CV_P2_V_01_fdk` | 0.9613 | 0.9661 | +0.0048 | 0.626 | 0.388 | -0.238 | 0.6613 | 0.6847 | +0.0234 | 32.31 | 33.12 | +0.80 |
| `CV_P3_V_01_fdk` | 0.9435 | 0.9410 | -0.0024 | 0.473 | 0.559 | +0.086 | 0.8281 | 0.8309 | +0.0028 | 46.62 | 47.04 | +0.42 |
| `CV_P4_V_01_fdk` | 0.9517 | 0.9526 | +0.0009 | 0.321 | 0.255 | -0.066 | 0.7848 | 0.7828 | -0.0020 | 46.18 | 46.00 | -0.18 |
| `CV_P5_V_01_fdk` | 0.9516 | 0.9709 | +0.0192 | 1.293 | 0.449 | -0.844 | 0.8053 | 0.8029 | -0.0024 | 39.66 | 38.98 | -0.68 |
| **Mean** | 0.9523 | 0.9571 | +0.0048 | 0.622 | 0.407 | -0.215 | 0.7371 | 0.7419 | +0.0048 | 38.57 | 38.70 | +0.14 |

#### All FDK

| Scan | Dice FiLM | Dice NoFiLM | Δ Dice | 3D Err FiLM | 3D Err NoFiLM | Δ 3D | SSIM FiLM | SSIM NoFiLM | Δ SSIM | PSNR FiLM | PSNR NoFiLM | Δ PSNR |
|------|-----------|-------------|--------|-------------|---------------|------|-----------|-------------|--------|-----------|-------------|--------|
| `CE_P1_V_01_fdk` | 0.9360 | 0.9406 | +0.0045 | 0.467 | 0.304 | -0.162 | 0.8595 | 0.8785 | +0.0190 | 27.05 | 29.67 | +2.62 |
| `CE_P2_V_01_fdk` | 0.9405 | 0.9673 | +0.0268 | 1.218 | 0.179 | -1.039 | 0.7630 | 0.8374 | +0.0744 | 26.70 | 28.01 | +1.30 |
| `CE_P3_V_01_fdk` | 0.9084 | 0.9556 | +0.0472 | 1.602 | 0.327 | -1.275 | 0.8628 | 0.9186 | +0.0557 | 32.61 | 32.98 | +0.37 |
| `CE_P4_V_01_fdk` | 0.9641 | 0.9646 | +0.0005 | 0.494 | 0.488 | -0.007 | 0.6790 | 0.6643 | -0.0147 | 22.33 | 19.49 | -2.83 |
| `CE_P5_V_01_fdk` | 0.9303 | 0.9354 | +0.0051 | 0.986 | 0.805 | -0.181 | 0.8515 | 0.8408 | -0.0107 | 31.25 | 29.53 | -1.72 |
| `CV_P1_V_01_fdk` | 0.9535 | 0.9552 | +0.0017 | 0.396 | 0.384 | -0.012 | 0.6060 | 0.6083 | +0.0023 | 28.06 | 28.38 | +0.32 |
| `CV_P2_V_01_fdk` | 0.9613 | 0.9661 | +0.0048 | 0.626 | 0.388 | -0.238 | 0.6613 | 0.6847 | +0.0234 | 32.31 | 33.12 | +0.80 |
| `CV_P3_V_01_fdk` | 0.9435 | 0.9410 | -0.0024 | 0.473 | 0.559 | +0.086 | 0.8281 | 0.8309 | +0.0028 | 46.62 | 47.04 | +0.42 |
| `CV_P4_V_01_fdk` | 0.9517 | 0.9526 | +0.0009 | 0.321 | 0.255 | -0.066 | 0.7848 | 0.7828 | -0.0020 | 46.18 | 46.00 | -0.18 |
| `CV_P5_V_01_fdk` | 0.9516 | 0.9709 | +0.0192 | 1.293 | 0.449 | -0.844 | 0.8053 | 0.8029 | -0.0024 | 39.66 | 38.98 | -0.68 |
| **Mean** | 0.9441 | 0.9549 | +0.0108 | 0.788 | 0.414 | -0.374 | 0.7701 | 0.7849 | +0.0148 | 33.28 | 33.32 | +0.04 |

### B.3 Δ (full-res − 128) within each model

Positive Dice = full-res better; negative 3D error = full-res better.

| Scan | Δ Dice FiLM | Δ Dice NoFiLM | Δ 3D FiLM | Δ 3D NoFiLM | Δ SSIM FiLM | Δ SSIM NoFiLM | Δ PSNR FiLM | Δ PSNR NoFiLM |
|------|-------------|---------------|-----------|-------------|-------------|---------------|-------------|---------------|
| `CE_P1_V_01_fdk` | +0.0552 | +0.0542 | -0.175 | +0.001 | +0.0115 | +0.0065 | -0.64 | +1.33 |
| `CE_P2_V_01_fdk` | +0.0217 | +0.0297 | -0.252 | +0.001 | -0.0465 | -0.0318 | -0.34 | -1.34 |
| `CE_P3_V_01_fdk` | +0.0255 | +0.0464 | -0.018 | -0.019 | +0.0068 | -0.0121 | +6.01 | +4.14 |
| `CE_P4_V_01_fdk` | +0.0206 | +0.0211 | -0.317 | -0.273 | -0.1002 | -0.0987 | -6.39 | -8.98 |
| `CE_P5_V_01_fdk` | +0.0411 | +0.0372 | -0.022 | -0.041 | +0.0267 | +0.0277 | -3.04 | -4.39 |
| `CV_P1_V_01_fdk` | +0.0700 | +0.0687 | -0.518 | -0.344 | -0.1159 | -0.1015 | +1.93 | +2.88 |
| `CV_P2_V_01_fdk` | +0.0461 | +0.0506 | +0.097 | +0.166 | -0.0708 | -0.0671 | +5.77 | +6.33 |
| `CV_P3_V_01_fdk` | +0.0811 | +0.0802 | +0.019 | +0.029 | -0.0470 | -0.0502 | +19.62 | +20.01 |
| `CV_P4_V_01_fdk` | +0.0741 | +0.0749 | -0.015 | -0.159 | -0.0264 | -0.0267 | +20.38 | +20.09 |
| `CV_P5_V_01_fdk` | +0.0320 | +0.0405 | -0.054 | +0.140 | -0.0251 | -0.0449 | +12.61 | +11.85 |
| **Mean** | +0.0467 | +0.0504 | -0.125 | -0.050 | -0.0387 | -0.0399 | +5.59 | +5.19 |

### B.4 Full-res win counts and native sizes

Patients where **No-FiLM beats FiLM at native resolution** (of 10):

| Metric | No-FiLM wins |
|--------|--------------|
| Dice (higher better) | 9/10 |
| 3D Error (lower better) | 9/10 |
| SSIM (higher better) | 6/10 |
| PSNR (higher better) | 6/10 |

| Scan | Native size (D×H×W) |
|------|---------------------|
| `CE_P1_V_01_fdk` … `CE_P5_V_01_fdk` | 270×256×270 |
| `CV_P1_V_01_fdk` … `CV_P5_V_01_fdk` | 450×220×450 |

### B.5 Interpretation notes

- **Dice always rises** on average when going to native grid with scaled DVF upsampling (~+0.047 FiLM, ~+0.050 No-FiLM). That validates the “train @128, warp @native” path for PTV overlap.
- **3D error** improves on average for FiLM (−0.125 mm) and slightly for No-FiLM (−0.050 mm); a few Varian cases show small increases (centroid metric is sensitive to anisotropic grids).
- **SSIM** tends to drop slightly on upscale (mean ≈ −0.04); **PSNR** mean rises but is patient-dependent (large + on some CV scans, large − on CE_P4). Prefer Dice/3D for deciding whether upscale “works”; treat image metrics as secondary on anisotropic native grids.
- **Largest No-FiLM vs FiLM Dice gaps @full-res:** CE_P3 (+0.047), CE_P2 (+0.027), CV_P5 (+0.019).
- **Only full-res Dice loss for No-FiLM:** CV_P3 (−0.0024) — essentially a tie.

Related earlier GTVol pilot (not FDK): `FULLRES_DVF_UPSAMPLE.md` (CE_P1 / CV_P2) showed the same Dice uplift pattern.

---

## Caveats

- Sweep (Part A) and full-res paired-128 (Part B.1) use **different sample protocols** — do not subtract them row-wise as if identical.
- Full-res scores the train-pair stratified set, not a held-out breathing sweep; it answers “does DVF upscale preserve / improve warp quality?” more than “generalize across unseen phases.”
- FiLM / No-FiLM were trained separately (same recipe); differences can include optimization noise, not only the FiLM module.
- No statistical significance tests; n = 10 patients.

---

## Source files

| Role | Path |
|------|------|
| Sweep FiLM | `results/<scan>_fdk/sweep_metrics.json` |
| Sweep No-FiLM | `results/<scan>_fdk/sweep_metrics_nofilm.json` |
| Full-res FiLM | `results/<scan>_fdk/fullres_vs_128_metrics_film.json` |
| Full-res No-FiLM | `results/<scan>_fdk/fullres_vs_128_metrics_nofilm.json` |
| FiLM checkpoint | `runs/<scan>_fdk/checkpoints/best.pt` |
| No-FiLM checkpoint | `runs/<scan>_fdk/checkpoints_nofilm/best.pt` |
| DVF upsample helper | `ml/flow_utils.py` (`upsample_dvf`) |
| Full-res eval script | `scripts/run_fullres_dvf_eval.py` (`--no-film` supported) |
| Full-res FDK batch | `scripts/run_fdk_fullres_eval.sh` |
| No-FiLM train / sweep batch | `scripts/run_fdk_nofilm_batch.sh`, `scripts/run_fdk_nofilm_eval.sh` |
| GTVol Phase-1 upscale note | `FULLRES_DVF_UPSAMPLE.md` |
| Experiment plan | `SUBSAMPLE_TRAIN_FULLRES_EVAL_PLAN.md` |
