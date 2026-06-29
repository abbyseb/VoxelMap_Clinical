# Note to Agent — VoxelMap Clinical Elekta (SPARE)

**Repo:** https://github.com/abbyseb/VoxelMap_Clinical  
**Goal:** Train **concatenated VoxelMap + FiLM** on **Clinical Elekta** SPARE data (`CE_P*`), with **CUDA GPU** on a Linux workstation.

This document is for an agent (or human) cloning the repo on a **new machine** and continuing work. Large data and run artifacts stay **outside git** — you copy them locally or extract from SPARE archives.

---

## 1. What is already done (Mac / T7 session)

| Phase | Status | Details |
|-------|--------|---------|
| **Planning & docs** | Done | `ELEKTA_CONCAT_TRAINING_PLAN.md`, `ELEKTA_DRR_VERIFICATION.md` |
| **Elekta DRR config** | Done | `config/elekta_drr.py` — 512×512 @ 0.8 mm, SID/SDD 1000/1536, per-scan `Geometry.xml` |
| **Coordinate check** | Done | Elekta & MC both **RAI / IEC 61217**; geometry differs (full-fan vs half-fan) — see verification doc |
| **Phase 1 staging** | Done locally | `CE_P1_V_01` selectively extracted (participant `Proj/` + evaluation `GTVol_*.mha`) |
| **Phase 2 preprocessing** | Done locally | Downsample → DRR → DVF → `prep_train` for `CE_P1_V_01` |
| **DRR MP4 verification** | Done locally | `scripts/export_drr_mp4.py` → gantry sweep / acquired vs sim |
| **Phase 3 training** | **Started, not finished** | Mac CPU only; ~0 epochs completed before abort; MPS unsupported |

### Reference patient / scan (pilot)

- **Patient:** `CE_P1` (Elekta clinical — not `MC_P1`)
- **Scan:** `CE_P1_V_01`
- **Geometry:** `Proj/Geometry.xml` from SPARE (340 projections, full-fan) — **never** `Geometry_SPARE.xml`

### ModelTraining layout (after Phase 2)

```
runs/CE_P1_V_01/ModelTraining/train/CE_P1_V_01/
├── SourceProjections/   340   (phase 06)
├── TargetProjections/  3060   (9 phases × 340)
├── DVFs/                 9
├── Masks/                4
├── Angles.csv          340
└── SourceVolumes/       10
```

**3,060** training pairs for `VoxelMapDataset` (concatenated + FiLM).

---

## 2. What is NOT in this Git repo

`.gitignore` excludes all heavy artifacts. Copy or re-extract on the workstation:

| Asset | Typical location on T7 | In git? |
|-------|----------------------|--------|
| SPARE archives | `…/SPARE_PublicArchive/` | No |
| Staged scan | `data/staged/P1/CE_P1_V_01/` | No |
| Pipeline `train/` tensors | `runs/CE_P1_V_01/CE_P1_V_01/train/` | No |
| `ModelTraining/` | `runs/CE_P1_V_01/ModelTraining/` | No |
| Checkpoints / videos | `runs/CE_P1_V_01/checkpoints/`, `videos/` | No |
| **LEARN-GUI** (training stack) | `…/LEARN-GUI/LEARN-GUI-Python/` | No — separate clone/copy |

**Fastest handoff:** `rsync` the whole `runs/CE_P1_V_01/` tree from T7 if you want to skip re-preprocessing and go straight to GPU training.

---

## 3. Environment variables (set on workstation)

Configure once per machine — paths will differ from T7.

```bash
# This repo (clone)
export VOXELMAP_CLINICAL_ROOT="$HOME/VoxelMap_Clinical"

# LEARN-GUI Python package root (required for preprocessing + training)
export LEARN_GUI_ROOT="$HOME/LEARN-GUI/LEARN-GUI-Python"

# SPARE public archive root (participant + evaluation .7z or extracted trees)
export SPARE_PUBLIC_ARCHIVE="/path/to/SPARE_PublicArchive"

# Optional: CUDA device for training
export CUDA_VISIBLE_DEVICES=0
```

**T7 reference path (do not assume on workstation):**

```
/Volumes/T7 Shield/DENNIS_BACKUP/SPARE_PublicArchive
/Volumes/T7 Shield/DENNIS_BACKUP/LEARN-GUI/LEARN-GUI-Python
/Volumes/T7 Shield/DENNIS_BACKUP/VoxelMap-SPARE-Clinical
```

---

## 4. Workstation setup checklist

### 4.1 Clone repos

```bash
git clone https://github.com/abbyseb/VoxelMap_Clinical.git
# LEARN-GUI must be available separately (same revision as T7 if possible)
```

### 4.2 Python environment (CUDA)

Use a conda/venv with:

- `torch` + **CUDA** (verify `torch.cuda.is_available()`)
- `itk`, `itk-rtk` (GPU RTK optional but helps DRR)
- `itk-elastix` or elastix via ITK
- `numpy`, `pandas`, `scipy`, `pillow`

From LEARN-GUI: see `LEARN-GUI-Python/requirements.txt`.

**Do not use Apple MPS** for training — `grid_sampler_3d_backward` is not implemented on MPS. Workstation should use **`--device cuda`**.

### 4.3 SPARE data

Either:

**A. Copy preprocessed run from T7 (fastest to train):**

```bash
rsync -av "/source/T7/.../VoxelMap-SPARE-Clinical/runs/CE_P1_V_01/" \
  "$VOXELMAP_CLINICAL_ROOT/runs/CE_P1_V_01/"
```

**B. Copy SPARE archive only and re-run pipeline:**

Copy or mount `SPARE_PublicArchive` to `$SPARE_PUBLIC_ARCHIVE`, then follow §5.

---

## 5. Full pipeline (agent workflow)

### Step 0 — Locate Elekta in SPARE

Under `$SPARE_PUBLIC_ARCHIVE`:

```
ParticipantDatasets/ClinicalElektaDatasets.7z.001–006   # not extracted on T7 (full set)
Evaluation/SPARE_GroundTruth.7z.001–024
```

After extract:

```
ClinicalElektaDatasets/Validation/P1/CE_P1_V_01/
├── Proj/Geometry.xml, Proj_*.bin, RespBin.csv
└── (no GTVol — those are in GroundTruth)

Evaluation/ClinicalElektaDatasets/.../CE_P1_V_01/
└── GTVol_01.mha … GTVol_10.mha, Mask_*.mha
```

**Password:** `#MakeIt1Minute!` (see `SPARE_PublicArchive/README.md` on T7 or SPARE site).

**Selective extract** (one scan, saves disk):

```bash
cd "$VOXELMAP_CLINICAL_ROOT/data/staged"
7z x "$SPARE_PUBLIC_ARCHIVE/ParticipantDatasets/ClinicalElektaDatasets.7z.001" \
  -p'#MakeIt1Minute!' -o. \
  "ClinicalElektaDatasets/P1/CE_P1_V_01/Proj/*" \
  "ClinicalElektaDatasets/P1/CE_P1_V_01/FDKRecon/FDK4D_06.mha"

7z x "$SPARE_PUBLIC_ARCHIVE/Evaluation/SPARE_GroundTruth.7z.001" \
  -p'#MakeIt1Minute!' -o. \
  "ClinicalElektaDatasets/P1/CE_P1_V_01/GTVol_*.mha" \
  "ClinicalElektaDatasets/P1/CE_P1_V_01/Mask_*.mha"
```

Merge into: `data/staged/P1/CE_P1_V_01/` (see `ELEKTA_CONCAT_TRAINING_PLAN.md`).

### Step 1 — Verify geometry / coordinates

```bash
cd "$VOXELMAP_CLINICAL_ROOT"
python scripts/compare_spare_geometry.py
```

Read `ELEKTA_DRR_VERIFICATION.md`.

### Step 2 — Preprocess (Phase 2)

```bash
export LEARN_GUI_ROOT=...   # see §3
python scripts/run_elekta_phase2.py \
  --run-root "$VOXELMAP_CLINICAL_ROOT/runs/CE_P1_V_01" \
  --staged "$VOXELMAP_CLINICAL_ROOT/data/staged/P1/CE_P1_V_01"
```

Produces: `runs/CE_P1_V_01/ModelTraining/train/CE_P1_V_01/`

**Critical:** DRR uses `config/elekta_drr.py` + scan `Proj/Geometry.xml` — not MC `Geometry_SPARE.xml`.

**Optional:** DRR verification MP4:

```bash
python scripts/export_drr_mp4.py
```

### Step 3 — Train (Phase 3, GPU)

```bash
cd "$LEARN_GUI_ROOT"
python ml/trainer.py \
  --data_dirs "$VOXELMAP_CLINICAL_ROOT/runs/CE_P1_V_01/ModelTraining/train/CE_P1_V_01" \
  --architecture concatenated \
  --use_film \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-5 \
  --val_split 0.1 \
  --device cuda \
  --num_workers 4 \
  --save_path "$VOXELMAP_CLINICAL_ROOT/runs/CE_P1_V_01/checkpoints/CE_P1_V_01_concat_film.pt"
```

Or wrapper:

```bash
LEARN_GUI_ROOT=... VOXELMAP_CLINICAL_ROOT=... python scripts/run_elekta_phase3_train.py
```

(Update `run_elekta_phase3_train.py` to pass `--device cuda` on workstation — see §7.)

### Step 4 — Evaluate (after training)

```bash
cd "$LEARN_GUI_ROOT"
python ml/tester.py \
  --checkpoint "$VOXELMAP_CLINICAL_ROOT/runs/CE_P1_V_01/checkpoints/CE_P1_V_01_concat_film.pt" \
  --data_dirs "$VOXELMAP_CLINICAL_ROOT/runs/CE_P1_V_01/ModelTraining/train/CE_P1_V_01"
```

(Confirm `tester.py` CLI flags match your LEARN-GUI revision.)

---

## 6. Known issues & fixes

| Issue | Fix |
|-------|-----|
| `ValueError: invalid literal for int() with base 10: 'Proj'` | macOS `._*` sidecar files in `ModelTraining/` — delete `find … -name '._*' -delete`; `dynamic_dataset.py` skips `._*` prefix |
| MPS / Mac GPU training fails | Use **CUDA** on workstation; MPS lacks `grid_sampler_3d_backward` |
| Slow CPU training | Expected on Mac; workstation GPU is the reason to migrate |
| `pydicom` import if loading `dicom2mha` package | Phase 2 uses direct `spare.py` import to avoid this |
| Elekta geometry estimated on some scans | SPARE README §17 — validate DRR vs acquired `Proj/*.bin` (MP4 script) |
| Hardcoded T7 paths in old scripts | Use `LEARN_GUI_ROOT` / `VOXELMAP_CLINICAL_ROOT` env vars (§3) |

---

## 7. Suggested agent improvements (not yet implemented)

Priority for workstation agent:

1. **`run_elekta_phase3_train.py`** — add `--device cuda` (auto-detect CUDA, fail clearly if missing).
2. **`run_elekta_phase2.py`** — read `LEARN_GUI_ROOT` from env (partially done).
3. **`scripts/stage_elekta_scan.py`** — parameterized selective 7z extract + merge using `$SPARE_PUBLIC_ARCHIVE`.
4. **Per-epoch checkpoints** in `trainer.py` (currently only `--save_path` at end).
5. **Multi-scan training** — `--data_dirs` for `CE_P1_V_01` + `CE_P1_V_02` etc.
6. **Other Elekta patients** — `CE_P2`…`CE_P5` (same pipeline, new `--scan-id`).

---

## 8. File map (this repo)

```
VoxelMap_Clinical/
├── NOTE_TO_AGENT.md              ← this file
├── ELEKTA_CONCAT_TRAINING_PLAN.md
├── ELEKTA_DRR_VERIFICATION.md
├── config/
│   └── elekta_drr.py             ← Elekta DRR detector + geometry path helper
├── ml/utilities/                 ← vendored VoxelMap networks (from LEARN-GUI)
│   ├── networksFiLM.py           ← concatenated + FiLM Model (primary)
│   ├── networks.py
│   ├── layers.py
│   ├── modelio.py
│   ├── losses.py
│   └── README.md
└── scripts/
    ├── compare_spare_geometry.py ← MC vs Elekta coordinate/geometry diff
    ├── export_drr_mp4.py         ← verification videos
    ├── run_elekta_phase2.py      ← preprocess → ModelTraining
    └── run_elekta_phase3_train.py← training wrapper (point at CUDA on workstation)
```

**External dependency:** `LEARN-GUI/LEARN-GUI-Python` — `ml/trainer.py`, `ml/dynamic_dataset.py`, preprocessing modules.

**Patches already applied on T7 LEARN-GUI (may need re-applying on workstation clone):**

- `ml/trainer.py` — `--save_path`, `--num_workers` (default 0)
- `ml/dynamic_dataset.py` — skip `._*` files

---

## 9. Success criteria

Agent is done when:

1. [ ] `ModelTraining/train/CE_P1_V_01/` exists with 3060+ samples, 340 angles
2. [ ] DRR MP4 shows reasonable acquired vs simulated alignment
3. [ ] Training completes 50 epochs on **CUDA** without error
4. [ ] Checkpoint saved under `runs/CE_P1_V_01/checkpoints/`
5. [ ] `tester.py` metrics logged (Dice / centroid shift)

---

## 10. Quick decision tree

```
Have rsync'd ModelTraining from T7?
  YES → Skip to Step 3 (GPU train)
  NO  → Have SPARE_PUBLIC_ARCHIVE?
          YES → Step 0–2 (extract, stage, preprocess)
          NO  → Copy archive or full runs/ from T7 first
```

---

*Last updated: 2026-06-29 — pilot scan `CE_P1_V_01`, Mac preprocessing complete, training pending on CUDA workstation.*
