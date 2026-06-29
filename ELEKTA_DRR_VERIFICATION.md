# Elekta DRR Generation — Verification Plan

**Scan:** `CE_P1_V_01`  
**Phase:** 1 (staged) → DRR prep for your review before full pipeline  
**Date:** 2026-06-29

---

## Phase 1 status

| Step | Status | Location |
|------|--------|----------|
| Selective extract (participant `Proj/` + `FDKRecon`) | Done | `data/staged/P1/CE_P1_V_01/` |
| Selective extract (Evaluation `GTVol_*.mha`, masks) | Done | same folder |
| Canonical staging layout | Done | `Proj/`, `GTVol_01–10.mha`, `Mask_*.mha` |
| DRR config module | Done | `config/elekta_drr.py` |
| Geometry/coordinate comparison script | Done | `scripts/compare_spare_geometry.py` |

**Staged scan path:**

```
VoxelMap-SPARE-Clinical/data/staged/P1/CE_P1_V_01/
├── Proj/
│   ├── Geometry.xml      ← use this for DRR (340 projections)
│   ├── Proj_00001.bin …  ← float32 512×512
│   ├── RespBin.csv
│   └── RespPhase.csv
├── GTVol_01.mha … GTVol_10.mha
├── Mask_Body.mha, Mask_Lung.mha, Mask_PTV.mha, Mask_Rib.mha
└── FDKRecon/FDK4D_06.mha   ← sanity reference (phase 06)
```

Nested extract cache (safe to delete after review):  
`data/staged/ClinicalElektaDatasets/…`

---

## Coordinate system: MC vs Elekta

SPARE states **all volumes use IEC 61217** (see `SPARE_PublicArchive/README.md` §12).

### Volume frame (verified on `GTVol_06.mha`)

| Field | MC `MC_V_P1_NS_01` | Elekta `CE_P1_V_01` | Match? |
|-------|--------------------|---------------------|--------|
| `AnatomicalOrientation` | **RAI** | **RAI** | Yes |
| `TransformMatrix` | `1 0 0 0 1 0 0 0 1` | `1 0 0 0 1 0 0 0 1` | Yes |
| `ElementSpacing` | `1 1 1` mm | `1 1 1` mm | Yes |
| `CenterOfRotation` | `0 0 0` | `0 0 0` | Yes |
| `DimSize` | `450 220 450` | `270 256 270` | No (FOV) |
| `Offset` | `-224.5 -109.5 -224.5` | `-134.5 -127.5 -134.5` | No (FOV) |

**Verdict:** The **patient coordinate system is the same** (RAI, identity direction, 1 mm voxels). Only **field-of-view and origin offset** differ because Elekta reconstructions are smaller (270×256×270 vs 450×220×450).

Axis convention (IEC 61217):

| Axis | Direction | +ve |
|------|-----------|-----|
| x | Left–right | Patient's left |
| y | Superior–inferior | Inferior (feet) |
| z | Anterior–posterior | Anterior |

Respiratory bins: **01 = end inhale**, **06 = end exhale** (same as MC).

**No extra transpose/flip is required** for Elekta relative to the MC pipeline — use the same `dicom2mha` spare normalize, downsampling, and prep_train flow.

---

## Geometry XML: MC vs Elekta (must differ)

| Parameter | MC `MC_V_P1_NS_01` | Elekta `CE_P1_V_01` |
|-----------|-------------------|---------------------|
| Fan mode | Half-fan | **Full-fan** |
| Projections | 680 | **340** |
| SID / SDD (mm) | 1000 / **1500** | 1000 / **1536** |
| Global `ProjectionOffsetX` | **148** (half-fan shift) | *per-projection* (~0.36 mm) |
| Gantry start / step | 0° / ~0.53° | **270.36°** / ~0.59° |
| Acquired detector | 512×384 @ 0.776 mm (sim) | **512×512 @ 0.8 mm** |
| DRR detector (LEARN-GUI) | 1024×768 @ 0.388 mm | **512×512 @ 0.8 mm** |

**Verdict:** Do **not** use `Geometry_SPARE.xml` or MC gold-batch `DRR_OPTS` for Elekta. Always pass the scan's own `Proj/Geometry.xml` plus Elekta detector settings from `config/elekta_drr.py`.

---

## DRR configuration (`config/elekta_drr.py`)

```python
ELEKTA_DRR_OPTS = {
    "geometry_source": "xml",
    "geometry_path": "<scan>/Proj/Geometry.xml",   # per scan
    "detector_origin": (-204.8, -204.8, 0.0),      # 512 × 0.8 mm / 2
    "detector_spacing": (0.8, 0.8, 1.0),
    "detector_size_xy": (512, 512),
}
```

Helper:

```python
from config.elekta_drr import elekta_drr_opts_for_scan
opts = elekta_drr_opts_for_scan("data/staged/P1/CE_P1_V_01")
```

### MC gold-batch settings (for comparison — wrong for Elekta)

```python
MC_VARIAN_DRR_OPTS = {
    "detector_origin": (-200.0, -150.0, 0.0),
    "detector_spacing": (0.388, 0.388, 1.0),
    "detector_size_xy": (1024, 768),
}
```

---

## How to verify locally

### 1. Run geometry / coordinate comparison

```bash
cd "/Volumes/T7 Shield/DENNIS_BACKUP/VoxelMap-SPARE-Clinical"
python3 scripts/compare_spare_geometry.py
```

### 2. Normalize volumes (spare)

```bash
cd "/Volumes/T7 Shield/DENNIS_BACKUP/LEARN-GUI/LEARN-GUI-Python"
python3 -c "
from pathlib import Path
from modules.dicom2mha.implementations.spare import run
run(Path('.../data/staged/P1/CE_P1_V_01'), Path('.../runs/CE_P1_V_01/train'))
"
```

Creates `CT_01.mha` … `CT_10.mha` (hardlinks from `GTVol_*`).

### 3. DRR smoke test (one phase, ITK-RTK env required)

```bash
cd "/Volumes/T7 Shield/DENNIS_BACKUP/LEARN-GUI/LEARN-GUI-Python"
python3 -c "
from pathlib import Path
from modules.drr_generation.run import run as run_drr
import sys; sys.path.insert(0, '.../VoxelMap-SPARE-Clinical/config')
from elekta_drr import elekta_drr_opts_for_scan
scan = Path('.../data/staged/P1/CE_P1_V_01')
opts = elekta_drr_opts_for_scan(scan)
vol = Path('.../runs/CE_P1_V_01/train/CT_06.mha')
ok, err = run_drr(str(vol), str(Path('.../runs/CE_P1_V_01/drr_smoke')), ct_num=6, dataset_type='clinical', **opts)
print(ok, err)
"
```

### 4. Visual DRR check (recommended)

Compare for **same gantry index** (e.g. projection 1):

| Source | File |
|--------|------|
| Acquired | `Proj/Proj_00001.bin` (float32 512×512) |
| Simulated DRR | `runs/CE_P1_V_01/drr_smoke/06_Proj_001.png` |

Expect similar anatomy and gantry angle (~270.36° for proj 1). Intensity scale may differ (clinical scatter/beam hardening); geometry alignment is the critical check.

### 5. FDK reference

Open `FDKRecon/FDK4D_06.mha` alongside `GTVol_06.mha` in ITK-SNAP or LEARN-GUI — same phase, same RAI frame.

---

## Checklist before Phase 2 (full pipeline)

- [x] `compare_spare_geometry.py` shows RAI match + geometry differences as above
- [x] `CT_06.mha` exists after spare normalize
- [x] DRR all phases with Elekta detector settings (~3 min on GPU)
- [ ] Optional: overlay `Proj_00001.bin` vs `06_Proj_001.bin` for visual geometry check
- [x] `Angles.csv` from `prep_train` has 340 angles matching `Geometry.xml`

## Phase 2 complete (2026-06-29)

`scripts/run_elekta_phase2.py` finished in ~5 min for `CE_P1_V_01`.

Output: `runs/CE_P1_V_01/ModelTraining/train/CE_P1_V_01/`

---

## Known Elekta caveats (SPARE README §17)

1. Some Elekta geometries were **re-estimated** (even angular spacing) after original files were lost — treat `Geometry.xml` as authoritative but validate visually.
2. Clinical intensities are **not HU-calibrated** (attenuation scale).
3. Full GT is FDK from full scan — registration DVFs will be noisier than MC.

---

## Next steps (Phase 2)

1. Downsample `CT_*` → `sub_CT_*` (128³)
2. DRR all phases with `ELEKTA_DRR_OPTS`
3. Compress → `.bin`
4. Elastix DVF (phase 06 reference)
5. `prep_train` → `ModelTraining/`
6. `trainer.py --architecture concatenated --use_film`

See `ELEKTA_CONCAT_TRAINING_PLAN.md` for the full training path.
