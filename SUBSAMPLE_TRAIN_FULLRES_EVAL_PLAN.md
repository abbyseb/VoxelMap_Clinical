# Subsample-Train / Full-Resolution-Eval Plan (DVF Upsampling)

**Goal:** Train VoxelMap + FiLM cheaply on a subsampled (128³) grid, then evaluate on
the native full-resolution grid by **interpolating the predicted DVF** with
`torch.nn.functional.interpolate(..., mode="trilinear")` and warping the full-res
volume / masks.

---

## 1. Current state

| Stage | Resolution | Data |
|-------|------------|------|
| Train | **128³** volumes, **128×128** DRRs | `sub_CT_*`, Elastix DVF at 128³ |
| Eval | **128³** | Same grid |
| FDK batch (`*_fdk`) | Train on **FDK4D** (1-min scan) | Different voxels from `GTVol_*`, not just lower res |

The model already upsamples its own DVF output to match the transformer grid:

```512:518:ml/utilities/networks.py
        if pos_flow.shape[2:] != tuple(self.transformer.grid.shape[2:]):
            pos_flow = torch.nn.functional.interpolate(
                pos_flow,
                size=self.transformer.grid.shape[2:],
                mode="trilinear",
                align_corners=True,
            )
```

**Key fact:** DVFs are stored in **voxel-index units** (added directly to the sampling
grid in `SpatialTransformer`). Upsampling to a finer grid therefore requires
**scaling the displacement magnitudes**, not just interpolating the values.

Native Elekta grid: **270 × 256 × 270** voxels @ 1 mm (anisotropic — not cubic).

---

## 2. Three distinct "subsample → full" problems

| Variant | Train on | Test on | DVF interpolate helps? |
|---------|----------|---------|------------------------|
| **A — Resolution** | 128³ `GTVol` (current) | Native **270×256×270** `GTVol` | **Yes** — main use case |
| **B — FDK clinical** | FDK4D (340/680 proj) | Full-scan `GTVol` | **Partially** — fixes grid size, not appearance gap |
| **C — Projection count** | 340 proj | ~1000 proj | **No** — need different DRR inputs, not DVF upsample |

This plan targets **Variant A** first, then **Variant B** using the same eval path.

---

## 3. Critical rule: interpolate **then scale**

```python
import torch
import torch.nn.functional as F

def upsample_dvf(flow_128, size_native, align_corners=True):
    """flow_128: (B, 3, D, H, W) displacements in *voxel units* on the 128 grid.
    size_native: (D, H, W) target grid, e.g. (270, 256, 270)."""
    d, h, w = flow_128.shape[2], flow_128.shape[3], flow_128.shape[4]
    scale = torch.tensor(
        [size_native[0] / d, size_native[1] / h, size_native[2] / w],
        device=flow_128.device, dtype=flow_128.dtype,
    ).view(1, 3, 1, 1, 1)
    flow_up = F.interpolate(
        flow_128, size=size_native, mode="trilinear", align_corners=align_corners
    )
    return flow_up * scale
```

**Gotchas:**
- **Per-axis scaling** — native grid is anisotropic (270×256×270), so scale each axis
  independently.
- **`align_corners` consistency** — model uses `True`; `dynamic_dataset._resize_flow_field`
  uses `False`. Pick one convention for the full-res path and use it everywhere.
- **Order of ops** — integrate (`VecInt`) at 128³ **then** upsample (matches training).
  Upsampling before integration can diverge for large motion.
- **Masks** — upsample PTV/body masks with `nearest` / `nearest-exact`, never trilinear.

---

## 4. Experiment plan

### Phase 0 — Baselines (no code change)

| Run | Train data | Eval grid | Notes |
|-----|------------|-----------|-------|
| B0 | GTVol 128³ | 128³ | Current pipeline |
| B1 | FDK4D 128³ | 128³ | `*_fdk` batch (in progress) |

### Phase 1 — Resolution upsample only (~1–2 days)

Same checkpoint as B0; **only the eval changes**.

1. Add `upsample_dvf()` helper → `ml/flow_utils.py`.
2. New eval flag: `--full-res-eval --native-size 270 256 270`.
3. Inference unchanged: **128×128** DRRs → DVF @ 128³.
4. Post-process: upsample + scale DVF → warp **full-res** `GTVol_06` + PTV/body masks.
5. Metrics: PTV Dice, 3D centroid error (mm), body-masked PSNR/SSIM at **native** grid.

**Sanity checks:**
- Upsample **GT** DVF @ 128 → warp full-res PTV → should be near ceiling (validates the
  upsample+scale math).
- Compare: warp @128 then upsample the *volume* vs warp @270 with upsampled *DVF* — the
  latter should be sharper at the PTV boundary.

### Phase 2 — FDK train → GTVol full eval

Same upsample path as Phase 1, but checkpoint from `*_fdk` runs.

**Fair comparison:** GT DVF was registered on **downsampled** volumes. For full-res eval,
upsample the GT DVF with the *same* scaling — do **not** compare against a fresh 270³
Elastix registration unless the supervision is regenerated.

### Phase 3 — Ablations (optional, paper-worthy)

| Ablation | Question |
|----------|------------|
| Trilinear vs bicubic/spline | How much smoothing hurts diaphragm motion? |
| Scale vs no-scale upsample | Quantify the bug if displacement scaling is skipped |
| Per-axis vs isotropic scale | Matters for 270×256×270 anisotropy |
| Body-masked PSNR @128 vs @270 | Does full-res eval change per-patient ranking? |

---

## 5. Expected outcomes

| Likely outcome | Why |
|----------------|-----|
| **Modest Dice gain** at full res | PTV boundary is ~1–2 mm; 128³ ≈ 2.1 mm/voxel |
| **3D error (mm) similar** | Centroid metric already in mm; less resolution-sensitive |
| **FDK→GTVol full eval may disappoint** | Volume appearance mismatch; DVF upsample doesn't fix projection-domain gap |
| **Trilinear oversmooths** | Sharp sliding-boundary motion may blur |

---

## 6. Implementation scope

1. `ml/flow_utils.py` — `upsample_dvf`, mask upsample helper.
2. `ml/evaluator.py` / `ml/sweep_evaluator.py` — `--full-res-eval` branch.
3. Eval-only staging: keep full-res `GTVol_*.mha` + native-grid masks (no retrain).
4. Results doc: section comparing 128 vs 270 eval (e.g. `Results-README-FDK.md`).

**Do not** retrain at 270³ initially — memory and DRR cost explode. The whole point is
**cheap train, expensive eval via DVF upsample**.

---

## 7. Recommended first step

Run **Phase 1 on one patient** (`CE_P1_V_01` or `CV_P1_V_01`) with an existing GTVol
checkpoint:

1. Predict DVF @ 128³ (unchanged).
2. Upsample + scale → warp full-res PTV.
3. Compare Dice @128 vs @270.

If Phase 1 shows a gain, apply the same eval path to **FDK-trained** checkpoints (Phase 2)
to assess whether clinical training + full-res warping is a viable reporting strategy.
