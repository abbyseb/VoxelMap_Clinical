# Elekta SPARE — Concatenated VoxelMap Training Plan

Train **concatenated VoxelMap (+ FiLM)** on one **Clinical Elekta** SPARE case (`CE_P*`).

**Workspace:** `VoxelMap-SPARE-Clinical`  
**Pipeline stack:** `LEARN-GUI/LEARN-GUI-Python` on T7  
**Related:** DREME paper (Shao et al. 2025) used 5 Elekta SPARE patients (their IDs P4–P8); this plan targets the public `CE_P1`–`CE_P5` folders.

---

## Scope

| Item | Choice |
|------|--------|
| Patient | One of `CE_P1`–`CE_P5` (different people; **`CE_P1` recommended** for first run) |
| Scan | One validation scan to start, e.g. `CE_P1_V_01` |
| Architecture | `concatenated` + `--use_film` |
| Not usable as-is | MC gold batch (`Geometry_SPARE.xml`, 1024×768 Varian half-fan) |

### Naming reminder

- `CE_P1` is **not** the same person as `MC_V_P1` or DREME paper patient P1.
- SPARE patient numbers are **per dataset type only** (MC / Varian / Elekta).

### Elekta vs MC geometry

| Parameter | Monte Carlo / Varian (`MC_V_*`) | Clinical Elekta (`CE_*`) |
|-----------|--------------------------------|---------------------------|
| Fan mode | Half-fan | Full-fan |
| Projections (1-min) | 680 | 340 |
| Detector | 1024×768, 0.388 mm | 512×512, 0.8 mm |
| SID / SDD | 1000 / 1500 mm | 1000 / 1536 mm |
| Recon size | 450×220×450 | 270×256×270 |
| Ground truth | Simulated 4D-CT | FDK full-scan reference (registration-based DVFs) |

---

## Phase 0 — Prerequisites

**Disk:** Plan **~150–200 GB free** on T7 for Elekta participant extract + GroundTruth for the chosen patient.

**Software:**

- `LEARN-GUI/LEARN-GUI-Python`
- ITK/RTK (DRR generation)
- Elastix (3D DVF)
- CUDA GPU for training

**Archives on T7 (not yet extracted):**

```
SPARE_PublicArchive/ParticipantDatasets/ClinicalElektaDatasets.7z.001–006
SPARE_PublicArchive/Evaluation/SPARE_GroundTruth.7z.001–024
```

**Scan folder pattern** (after extract):

```
ClinicalElektaDatasets/Validation/P{n}/CE_P{n}_V_{01,02,...}/
```

---

## Phase 1 — Extract and stage data

### 1a. Extract archives

Password: `#MakeIt1Minute!` (see `SPARE_PublicArchive/README.md`).

```bash
cd "/Volumes/T7 Shield/DENNIS_BACKUP/SPARE_PublicArchive/ParticipantDatasets"
7z x ClinicalElektaDatasets.7z.001 -p'#MakeIt1Minute!'

cd "/Volumes/T7 Shield/DENNIS_BACKUP/SPARE_PublicArchive/Evaluation"
7z x SPARE_GroundTruth.7z.001 -p'#MakeIt1Minute!'
```

### 1b. Merge participant + evaluation ground truth

LEARN-GUI `dicom2mha` (spare) expects `GTVol_*.mha` alongside projections. Participant packs have `Proj/`; true anatomy volumes live in **Evaluation GroundTruth**. Stage under this workspace:

```
VoxelMap-SPARE-Clinical/data/staged/
└── P1/
    └── CE_P1_V_01/
        ├── Proj/              ← ParticipantDatasets
        ├── Geometry.xml       ← per-scan (do not use Geometry_SPARE.xml)
        ├── RespBin.csv
        ├── GTVol_01.mha …     ← Evaluation GroundTruth (same scan name)
        └── (masks if present)
```

Copy or hardlink GT volumes from:

```
Evaluation/ClinicalElektaDatasets/.../CE_P1_V_01/
```

into the staged tree above.

---

## Phase 2 — Preprocessing pipeline

Run LEARN-GUI modules headless (GUI equivalent is fine). Use **`dataset_type="spare"`** for volume rename; use the **scan’s own `Geometry.xml`** for DRR.

| Step | Module | Notes |
|------|--------|-------|
| 1. Normalize volumes | `modules/dicom2mha` → `spare` | `GTVol_XX.mha` → `CT_XX.mha` |
| 2. Downsample | `modules/downsampling` | 270×256×270 → **128³** (`sub_CT_*.mha`) |
| 3. DRR | `modules/drr_generation` | Elekta: **512×512**, **0.8 mm** spacing, SID/SDD **1000/1536** |
| 4. Compress | `modules/drr_compression` | 128×128 `.bin` |
| 5. 3D DVF | `modules/dvf_generation` | Elastix; reference phase **06** → other phases |
| 6. Pack tensors | `modules/prep_train` | `ModelTraining/` layout |

### Elekta DRR parameters (not in MC gold batch)

`VoxelMapTestBed` gold batch hardcodes Varian settings (`1024×768`, `0.388 mm`). For Elekta, override when calling DRR:

```python
DRR_OPTS_ELEKTA = {
    "geometry_source": "xml",
    "geometry_path": "<scan>/Geometry.xml",
    "detector_origin": (...),      # confirm from Geometry.xml / SPARE README
    "detector_spacing": (0.8, 0.8, 1.0),
    "detector_size_xy": (512, 512),
}
```

### Expected run output

```
VoxelMap-SPARE-Clinical/runs/CE_P1_V_01/
├── train/
│   ├── CT_*.mha, sub_CT_*.mha, DVF_sub_*.mha
│   └── *_Proj_*.bin
└── ModelTraining/train/CE_P1_V_01/
    ├── SourceProjections/    # phase 06 @ each angle
    ├── TargetProjections/    # other phases
    ├── DVFs/
    ├── Masks/                # optional
    └── Angles.csv            # required for FiLM
```

---

## Phase 3 — Train concatenated VoxelMap

From `LEARN-GUI-Python`:

```bash
cd "/Volumes/T7 Shield/DENNIS_BACKUP/LEARN-GUI/LEARN-GUI-Python"

python ml/trainer.py \
  --data_dirs "/Volumes/T7 Shield/DENNIS_BACKUP/VoxelMap-SPARE-Clinical/runs/CE_P1_V_01/ModelTraining/train/CE_P1_V_01" \
  --architecture concatenated \
  --use_film \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-5 \
  --val_split 0.1
```

**Defaults** (from `ml/training_config.py`): 128³ grid, FP32, 10% validation split.

### Single-scan vs multi-scan

One `CE_P*_V_*` scan still yields many pairs (~9 target phases × ~340 angles). For more variety on the same patient, pass multiple `--data_dirs`:

```bash
--data_dirs .../CE_P1_V_01 .../CE_P1_V_02
```

---

## Phase 4 — Validate

```bash
python ml/tester.py \
  --checkpoint <saved_model.pt> \
  --data_dirs <ModelTraining/test or held-out paths>
```

Metrics: Dice, centroid shift, DVF traces (same as MC workflow).

---

## Workflow overview

```
Extract CE + GT  →  Stage one CE scan  →  Pipeline (Elekta geometry)
       →  prep_train  →  trainer (concatenated + FiLM)  →  tester.py
```

| Phase | Effort | Status on T7 |
|-------|--------|--------------|
| Extract + stage | ~0.5–1 day | Archives not extracted |
| Elekta DRR config | ~0.5 day | Not in gold batch |
| Pipeline on 1 scan | Hours (RTK + Elastix) | Not started |
| Training | ~2–4 h GPU | Blocked on prep_train |

---

## Patient selection

| SPARE ID | Notes |
|----------|--------|
| **CE_P1** | Recommended first run — thoracic (stage I NSCLC / HCC cohort) |
| CE_P2–CE_P4 | Other thoracic Elekta patients |
| CE_P5 | Fifth Elekta patient in archive |

DREME paper Elekta cohort (their P4–P8) maps to **five `CE_*` patients** in SPARE, but exact `CE_P#` folder mapping is in supplementary material, not the main PDF.

---

## Known risks (clinical Elekta)

1. **No simulation GT** — DVFs from registration to FDK reference; noisier than MC.
2. **Estimated geometry** — SPARE README notes some Elekta `Geometry.xml` files use even angular spacing after originals were lost.
3. **Intensity scale** — not HU-calibrated; same convention as other SPARE clinical data.
4. **Cross-scanner training** — do not pool `CE_*` with `MC_V_*` in one model without explicit harmonization.

---

## Phase 1 progress (2026-06-29)

- [x] Selective extract + stage `CE_P1_V_01` → `data/staged/P1/CE_P1_V_01/`
- [x] DRR config → `config/elekta_drr.py`
- [x] Coordinate / geometry verification → `ELEKTA_DRR_VERIFICATION.md` + `scripts/compare_spare_geometry.py`
- [x] DRR smoke test (anaconda + ITK-RTK)
- [ ] Full extract of remaining Elekta patients (optional)

## Phase 2 progress (2026-06-29)

- [x] Downsample 10 phases → `sub_CT_*.mha` (128³)
- [x] DRR all phases with Elekta `Geometry.xml` + `config/elekta_drr.py` (~3 min)
- [x] 3400 projection bins (`{phase}_Proj_{###}.bin`, 128×128)
- [x] Elastix DVF: 9 fields (ref phase 06)
- [x] `prep_train` → `runs/CE_P1_V_01/ModelTraining/train/CE_P1_V_01/`
- [ ] Phase 3: `trainer.py --architecture concatenated --use_film` (running on CPU — see `runs/CE_P1_V_01/logs/phase3_train.log`)

**ModelTraining layout (ready for training):**

```
runs/CE_P1_V_01/ModelTraining/train/CE_P1_V_01/
├── SourceProjections/   340  (phase 06)
├── TargetProjections/  3060  (9 other phases × 340 angles)
├── DVFs/                  9
├── Masks/                 4
├── Angles.csv           340  (from scan Geometry.xml)
└── SourceVolumes/        10
```

**Runner:** `scripts/run_elekta_phase2.py` (log: `runs/CE_P1_V_01/logs/phase2.log`)

**Disk:** ~445 GB free on T7 after removing duplicate gold mirror.
4. [ ] Smoke-train (few epochs) to confirm non-empty `VoxelMapDataset`
5. [ ] Full 50-epoch run + `tester.py`
6. [ ] (Optional) Add Elekta profile to `VoxelMapTestBed` for repeatable gold prep

---

## References on T7

| Resource | Path |
|----------|------|
| SPARE dataset guide | `../SPARE_PublicArchive/README.md` |
| Headless trainer | `../LEARN-GUI/LEARN-GUI-Python/ml/trainer.py` |
| MC gold batch (Varian geometry — do not reuse for Elekta) | `../VoxelMapTestBed/scripts/batch_gold_spare.py` |
| DREME paper (Elekta clinical cohort) | `Shao_2025_Phys._Med._Biol._70_025026.pdf` |
