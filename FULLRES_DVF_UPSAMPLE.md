# Full-Resolution Eval via DVF Upsampling

**Status:** Phase 1 complete (CE_P1_V_01, CV_P2_V_01)  
**Date:** 2026-07-10  
**Related plan:** `SUBSAMPLE_TRAIN_FULLRES_EVAL_PLAN.md`

---

## What was done

VoxelMap + FiLM is trained on a **128³** grid (128×128 DRRs). At test time we keep that cheap inference path, then **upsample the predicted DVF** to the native CT grid and warp the full-resolution volume / PTV.

### Pipeline

1. Load existing GTVol-trained checkpoint (`best.pt`).
2. Run inference unchanged: projections + 128³ source volume → DVF @ 128³.
3. Upsample DVF with trilinear interpolation **and scale displacements** to native voxel units.
4. Warp native `CT_06` / `Mask_PTV` / `Mask_Body` with a full-res `SpatialTransformer`.
5. Compare metrics on the **same samples** at 128³ vs native.

```
DRR @128 ──► VoxelMap+FiLM ──► DVF @128
                                   │
                    F.interpolate (trilinear)
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
```

### Critical detail: interpolate **then scale**

DVFs are stored in **voxel-index units** (added directly to the sampling grid). Upsampling the field without scaling under-warps on a finer grid.

```python
# ml/flow_utils.py
flow_up = F.interpolate(flow_128, size=native_size, mode="trilinear", align_corners=True)
flow_up = flow_up * scale   # per-axis: native_size[i] / 128
```

| Patient | Native grid (D×H×W) | Scale from 128³ |
|---------|---------------------|-----------------|
| CE_P1_V_01 | 270 × 256 × 270 | ≈ (2.11, 2.00, 2.11) |
| CV_P2_V_01 | 450 × 220 × 450 | ≈ (3.52, 1.72, 3.52) |

Other conventions used:
- Integrate (`VecInt`) at 128³ **then** upsample (matches training).
- Masks warped at native resolution (loaded from staged MHAs), not trilinearly upsampled.
- Centroid shifts converted to mm with grid-appropriate spacing (native = 1 mm/voxel).

---

## Code added

| Path | Role |
|------|------|
| `ml/flow_utils.py` | `upsample_dvf`, `upsample_mask`, `load_mha_array`, spacing helper |
| `scripts/run_fullres_dvf_eval.py` | Phase-1 eval: 128 vs full-res on stratified train-pair samples |
| `SUBSAMPLE_TRAIN_FULLRES_EVAL_PLAN.md` | Full experiment plan (Phases 0–3) |

### How to re-run

```bash
export VOXELMAP_CLINICAL_ROOT=/home/abhishek/Documents/VoxelMap_Clinical
export LEARN_GUI_ROOT=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python
PY=$LEARN_GUI_ROOT/.venv/bin/python

$PY scripts/run_fullres_dvf_eval.py --scan-id CE_P1_V_01 --gpu 1 --max-samples 90
$PY scripts/run_fullres_dvf_eval.py --scan-id CV_P2_V_01 --gpu 0 --max-samples 90
```

Outputs:
- `runs/{scan}/eval_fullres/fullres_vs_128_metrics.json`
- Copied to `results/{scan}/fullres_vs_128_metrics.json`

---

## Results (Phase 1)

**Setup:** 90 stratified train-pair samples per patient (same samples for both resolutions).  
**Checkpoints:** GTVol-trained `runs/{scan}/checkpoints/best.pt` (not FDK).

### CE_P1_V_01 (Elekta, native 270×256×270)

| Metric | @128³ | @full-res | Δ |
|--------|-------|-----------|---|
| **Dice** | 0.884 | **0.938** | **+0.054** |
| 3D error (mm) | 0.399 | 0.533 | +0.134 |
| PSNR (dB) | 30.99 | 30.53 | −0.46 |
| SSIM | 0.882 | 0.878 | −0.003 |
| GT shift MAE 128↔native | — | 0.85 mm | scale sanity |

### CV_P2_V_01 (Varian, native 450×220×450)

| Metric | @128³ | @full-res | Δ |
|--------|-------|-----------|---|
| **Dice** | 0.932 | **0.971** | **+0.039** |
| **3D error (mm)** | 0.328 | **0.205** | **−0.122** |
| **PSNR (dB)** | 31.89 | **38.57** | **+6.68** |
| **SSIM** | 0.824 | **0.856** | **+0.032** |
| GT shift MAE 128↔native | — | 1.23 mm | scale sanity |

### Interpretation

- **PTV Dice improves on both patients** when warping at native resolution with a scaled upsampled DVF — the main intended win for this technique.
- **CV_P2** also improves 3D centroid error, PSNR, and SSIM.
- **CE_P1** is mixed: Dice up, but 3D error / image metrics slightly worse (centroid metric is sensitive to residual interpolation mismatch on anisotropic 270×256×270).
- GT shift consistency (~0.8–1.2 mm MAE) is acceptable for trilinear upsampling of Elastix DVFs; not perfect, but confirms scaling is in the right ballpark.

---

## What this is / is not

| This **is** | This is **not** |
|-------------|-----------------|
| Cheap full-res **evaluation** on top of 128³ models | Retraining at native resolution |
| Resolution upsample of the motion field | Fixing FDK vs GTVol appearance domain gap |
| Validated on GTVol-trained CE_P1 + CV_P2 | Yet run on all 10 patients or `*_fdk` checkpoints |

---

## Next steps (from the plan)

1. Extend Phase 1 to remaining CE/CV patients.
2. Phase 2: same eval path on `*_fdk` checkpoints → warp full-scan `GTVol`.
3. Ablations: scale vs no-scale, `align_corners` True/False, trilinear vs spline.

---

## Git

Pushed with the Phase-1 implementation and metrics (see commit history on `main`).
