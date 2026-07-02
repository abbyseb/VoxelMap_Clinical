# Methods — Clinical Elekta SPARE Dataset

This document describes how the **Clinical Elekta** subset of the SPARE challenge dataset was obtained, staged, preprocessed, used for training, and evaluated in this repository. It reflects the workflow implemented for patients **CE_P1_V_01** through **CE_P5_V_01** on a Linux workstation (NVIDIA RTX 6000 Ada, CUDA).

**Related files:** `ELEKTA_CONCAT_TRAINING_PLAN.md`, `ELEKTA_DRR_VERIFICATION.md`, `NOTE_TO_AGENT.md`, `config/elekta_drr.py`

---

## 1. Dataset source and extraction

### 1.1 SPARE public archive

Data come from the **SPARE** (Synthetic Projection Archive for Registration Evaluation) public release (Shieh et al.). The Clinical Elekta cohort is distributed as multi-part 7z archives:

| Archive | Contents |
|---------|----------|
| `ParticipantDatasets/ClinicalElektaDatasets.7z.001`–`.006` | Onboard projections, geometry, FDK reconstructions (per scan) |
| `Evaluation/SPARE_GroundTruth.7z.001`–`.024` | Ground-truth 4D-CT volumes and structure masks (per scan) |

**Password:** `#MakeIt1Minute!` (documented in `SPARE_PublicArchive/README.md`).

On the workstation, archives were either fully extracted or selectively extracted under:

```
$SPARE_ROOT/Participant_Datasets/ClinicalElektaDatasets/
$SPARE_ROOT/Evaluation/ClinicalElektaDatasets/
```

Local path used: `/home/abhishek/research-data/2RESEARCH/1_ClinicalData/SPAREChallenge/`

### 1.2 Selective extraction (single scan)

To save disk, one validation scan can be extracted without unpacking the full archive:

```bash
cd "$VOXELMAP_CLINICAL_ROOT/data/staged"
7z x "$SPARE_ROOT/Participant_Datasets/ClinicalElektaDatasets.7z.001" \
  -p'#MakeIt1Minute!' -o. \
  "ClinicalElektaDatasets/P1/CE_P1_V_01/Proj/*"

7z x "$SPARE_ROOT/Evaluation/SPARE_GroundTruth.7z.001" \
  -p'#MakeIt1Minute!' -o. \
  "ClinicalElektaDatasets/P1/CE_P1_V_01/GTVol_*.mha" \
  "ClinicalElektaDatasets/P1/CE_P1_V_01/Mask_*.mha"
```

### 1.3 Staging (merge participant + evaluation)

Participant packs contain **projections only**; anatomical ground truth lives in the **Evaluation** tree. `scripts/stage_elekta_scan.py` merges them via symlinks:

```bash
python scripts/stage_elekta_scan.py CE_P1_V_01 CE_P2_V_01 ...
```

**Source paths:**

```
Participant_Datasets/ClinicalElektaDatasets/P{n}/{scan_id}/Proj/
Evaluation/ClinicalElektaDatasets/P{n}/{scan_id}/GTVol_*.mha, Mask_*.mha
```

**Staged layout** (`data/staged/P{n}/{scan_id}/`):

```
CE_P{n}_V_01/
├── Proj/
│   ├── Geometry.xml
│   ├── Proj_00001.bin … Proj_00340.bin   # float32, 512×512
│   ├── RespBin.csv
│   └── RespPhase.csv
├── GTVol_01.mha … GTVol_10.mha
└── Mask_Body.mha, Mask_Lung.mha, Mask_PTV.mha, Mask_Rib.mha, …
```

### 1.4 File provenance — which files come from where

SPARE splits **imaging data** and **anatomical ground truth** across two archive families. Both must be extracted (or symlinked) for preprocessing.

| File / folder | In `ClinicalElektaDatasets.7z` (Participant)? | In `SPARE_GroundTruth.7z` (Evaluation)? | Extracted path on disk |
|---------------|-----------------------------------------------|----------------------------------------|-------------------------|
| `Proj/Geometry.xml` | **Yes** | No | `Participant_Datasets/ClinicalElektaDatasets/P{n}/{scan}/Proj/` |
| `Proj/Proj_*.bin` (acquired onboard CBCT) | **Yes** | No | same `Proj/` folder |
| `Proj/RespBin.csv`, `RespPhase.csv` | **Yes** | No | same `Proj/` folder |
| `FDKRecon/FDK4D_*.mha` (FDK recon volumes) | **Yes** | No | `Participant_Datasets/…/{scan}/FDKRecon/` |
| `GTVol_01.mha` … `GTVol_10.mha` (4D-CT GT) | **No** | **Yes** | `Evaluation/ClinicalElektaDatasets/P{n}/{scan}/` |
| `Mask_Body.mha` | **No** | **Yes** | `Evaluation/…/{scan}/` |
| `Mask_Lung.mha` | **No** | **Yes** | `Evaluation/…/{scan}/` |
| `Mask_PTV.mha` | **No** | **Yes** | `Evaluation/…/{scan}/` |
| `Mask_Rib.mha` | **No** | **Yes** | `Evaluation/…/{scan}/` |
| `Mask_CNR.mha` | **No** | **Yes** | `Evaluation/…/{scan}/` |
| `CNR_GroundTruth.mat`, `LinePoints_ERW.mat` | **No** | **Yes** | `Evaluation/…/{scan}/` (analysis metadata; not used in pipeline) |

**What `stage_elekta_scan.py` symlinks:**

| Staged file | Symlink target |
|-------------|----------------|
| `Proj/` | `Participant_Datasets/ClinicalElektaDatasets/P{n}/{scan_id}/Proj/` |
| `GTVol_*.mha` | `Evaluation/ClinicalElektaDatasets/P{n}/{scan_id}/GTVol_*.mha` |
| `Mask_*.mha` | `Evaluation/ClinicalElektaDatasets/P{n}/{scan_id}/Mask_*.mha` |

**Not copied by staging** (remain only under Participant if needed for QC): `FDKRecon/`. Optional reference for DRR verification; not required for VoxelMap training.

**After preprocessing**, masks appear again as downsampled NumPy arrays:

| Output | Produced from |
|--------|----------------|
| `train/Mask_*.mha` | Copied from staged `Mask_*.mha` during phase 2 |
| `ModelTraining/.../Masks/*_mha.npy` | `prep_train` resamples masks to 128³ |

---

## 2. What we found inside the Elekta data

### 2.1 Scan naming and patients

| Prefix | Meaning |
|--------|---------|
| `CE_P{n}` | Clinical **E**lekta patient *n* (independent of MC or Varian patient numbers) |
| `CE_P{n}_V_{01…}` | Validation onboard CBCT scans |
| `CE_P{n}_T_01` | Test scan (full projection set) |
| `CE_P{n}_Prior` | Planning CT |

This project trained one validation scan per patient: **CE_P1_V_01 … CE_P5_V_01**.

### 2.2 Imaging geometry (Elekta full-fan)

Verified on `CE_P1_V_01` (`ELEKTA_DRR_VERIFICATION.md`, `scripts/compare_spare_geometry.py`):

| Parameter | Clinical Elekta (`CE_*`) |
|-----------|--------------------------|
| Fan mode | **Full-fan** |
| Projections per 1-min sweep | **340** |
| Detector | **512 × 512**, **0.8 mm** pixel spacing |
| SID / SDD | **1000 / 1536 mm** |
| Native CT grid | **270 × 256 × 270** voxels @ **1 mm** (RAI / IEC 61217) |
| Respiratory phases | **10** bins (reference phase **06** = max exhale) |
| Projection files | `Proj_*.bin` — float32, 512×512 per frame |

**Important:** Use each scan’s own `Proj/Geometry.xml`. Do **not** use `Geometry_SPARE.xml` from the Monte Carlo / Varian gold batch (half-fan, 1024×768).

### 2.3 Ground truth

Unlike Monte Carlo SPARE data, Clinical Elekta has **no simulated 4D-CT**. Ground-truth volumes (`GTVol_*.mha`) are derived from a **full-scan FDK reference**; inter-phase **DVFs** are obtained by **non-rigid registration** (Elastix) of each phase to the reference phase 06. Intensities are **not HU-calibrated** (SPARE clinical convention).

### 2.4 Coordinate system

All volumes share **RAI** orientation and **IEC 61217** metadata (`AnatomicalOrientation`, identity `TransformMatrix`, 1 mm spacing). Elekta and Monte Carlo volumes use the same frame; only field-of-view and scanner geometry differ.

---

## 3. Preprocessing pipeline

Preprocessing is run by `scripts/run_elekta_phase2.py`, which calls LEARN-GUI modules headlessly (`LEARN_GUI_ROOT` → `LEARN-GUI-Python`).

```bash
export VOXELMAP_CLINICAL_ROOT=/path/to/VoxelMap_Clinical
export LEARN_GUI_ROOT=/path/to/LEARN-GUI-Python
python scripts/run_elekta_phase2.py --scan-id CE_P1_V_01 --with-test
```

### 3.1 Pipeline steps

| Step | Module (LEARN-GUI) | Method | Input → output |
|------|-------------------|--------|----------------|
| **1. Volume normalize** | `modules/dicom2mha/implementations/spare.py` | Rename/copy MHA headers | `GTVol_XX.mha` → `CT_XX.mha` |
| **2. Downsample** | `modules/downsampling/downsample.py` | Trilinear resample + metadata update | `CT_*.mha` (270³) → `sub_CT_*.mha` (**128³**) |
| **3. DRR** | `modules/drr_generation` + `config/elekta_drr.py` | ITK-RTK ray casting | `sub_CT_*.mha` → `{phase}_Proj_{###}.bin` (128×128) |
| **4. Compress** | `modules/drr_compression/compress.py` | Float32 → packed `.bin` | Same naming, training resolution |
| **5. 3D DVF** | `modules/dvf_generation` | ITK-Elastix B-spline (`Elastix_BSpline_Sliding_LowRes.txt`) | `sub_CT_06` fixed → `DVF_sub_{XX}.mha` (9 fields) |
| **6. Pack tensors** | `modules/prep_train/run.py` | NumPy export + pair indexing | `ModelTraining/train/` and `test/` |

**DRR settings** (`config/elekta_drr.py`):

```python
detector_size_xy = (512, 512)
detector_spacing = (0.8, 0.8, 1.0)   # mm
detector_origin  = (-204.8, -204.8, 0.0)  # mm, centred panel
geometry_path    = <scan>/Proj/Geometry.xml
```

DRR output is **128×128** to match the downsampled volume grid used by VoxelMap.

### 3.2 Intermediate files (`runs/{scan_id}/{scan_id}/train/`)

After steps 1–5, the run folder contains:

```
runs/CE_P1_V_01/CE_P1_V_01/train/
├── CT_01.mha … CT_10.mha
├── sub_CT_01.mha … sub_CT_10.mha          # 128³
├── 06_Proj_001.bin … 10_Proj_340.bin      # 10 phases × 340 angles (128×128 DRR)
├── DVF_sub_01.mha … DVF_sub_10.mha        # (no DVF for reference phase 06)
├── Mask_*.mha
└── Proj/Geometry.xml, RespBin.csv
```

### 3.3 ModelTraining outputs (step 6)

**Train split** — real phase-labelled pairs for supervised learning:

```
runs/{scan_id}/ModelTraining/train/{scan_id}/
├── SourceProjections/     # 340 files: phase 06 at each gantry angle
├── TargetProjections/     # 3,060 files: 9 other phases × 340 angles
├── DVFs/                  # 9 numpy DVFs (DVF_01_mha.npy … except 06)
├── SourceVolumes/         # sub_CT_*.npy (10 phases)
├── Masks/                 # Body, PTV, etc. (128³)
└── Angles.csv             # 340 gantry angles (degrees)
```

**Training pairs:** 9 target phases × 340 projections = **3,060** `(source, target, volume, DVF, angle)` samples. With 10% validation split → ~2,754 train / ~306 val per scan.

**Test split** (`--with-test`) — synthetic **breathing sweep** simulating continuous onboard CBCT:

```
runs/{scan_id}/ModelTraining/test/{scan_id}/
├── TestProjections/         # Proj_00001_bin.npy … Proj_00340_bin.npy
├── SourceTestProjections/   # phase 06 at every angle
├── RespBin.csv              # breathing phase per projection index (1…10 cycle)
├── Angles.csv
├── DVFs/, Masks/, SourceVolumes/
└── Test_Breathing_Sweep.mp4 # preview video
```

For projection index `i`, target phase = `((i - 1) % 10) + 1`; source is always phase **06**. This mimics a patient breathing while the gantry rotates (see `NOTE_TO_AGENT.md` §6).

---

## 4. Model and training

### 4.1 Architecture

- **Model:** Concatenated **VoxelMap + FiLM** (`ml/utilities/networksFiLM.py`)
- **Inputs:** source projection (phase 06), target projection, source volume (`sub_CT_06`), gantry **angle** (FiLM conditioning)
- **Output:** 3D displacement vector field (DVF), 128³, 3 channels (LR, SI, AP)
- **Spatial transform:** `ml/utilities/spatialTransform.py` (differentiable volume warping)

### 4.2 Training procedure

Script: `scripts/run_elekta_phase3_train.py` → `ml/trainer.py`

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 50 |
| Batch size | 8 |
| Learning rate | 1e-5 |
| Validation split | 10% (random) |
| Volume size | 128³ |
| Mixed precision | Off |
| Device | CUDA (`CUDA_VISIBLE_DEVICES=0`) |
| Optimizer | Adam (default in trainer) |

**Loss:** Masked flow MSE (`losses.flow_mask`) when a body/lung mask is available; otherwise MSE between predicted and ground-truth DVF. FiLM uses `Angles.csv` per sample.

**Checkpoints** (`runs/{scan_id}/checkpoints/`):

- `epochs/epoch_{N}.pt` — per-epoch weights
- `best.pt` — lowest validation loss
- `{scan_id}_concat_film.pt` — final best model symlink/copy

**Plots:** `runs/{scan_id}/plots/loss_curves.png`, `loss_history.json`

```bash
python scripts/run_elekta_phase3_train.py --scan-id CE_P1_V_01 --epochs 50 --gpu 0
```

### 4.3 Patients trained

| Scan | Train pairs | Best val loss | Sweep Dice (mean ± std) |
|------|-------------|---------------|-------------------------|
| CE_P1_V_01 | 3,060 | 0.0571 | 0.893 ± 0.030 |
| CE_P2_V_01 | 3,060 | 0.1012 | 0.942 ± 0.014 |
| CE_P3_V_01 | 3,060 | 0.0984 | 0.911 ± 0.012 |
| CE_P4_V_01 | 3,060 | 0.0314 | 0.953 ± 0.010 |
| CE_P5_V_01 | 3,060 | 0.1805 | 0.921 ± 0.023 |

Summaries published under `results/CE_P*_V_01/` (metrics JSON, plots, MP4s).

---

## 5. Evaluation (testing)

Two evaluation modes were used.

### 5.1 Train-pair evaluation (held-out layout)

**Script:** `scripts/run_elekta_phase4_eval.py` → `ml/evaluator.py`

**Data:** `ModelTraining/train/{scan_id}/` — all indexed source/target pairs (including those seen during training; not a separate held-out test set).

**Per sample:**

1. Load source proj (phase 06), target proj, `sub_CT_06`, gantry angle.
2. Forward pass → predicted DVF.
3. Warp **PTV mask** (or body mask fallback) with predicted and GT DVFs (`spatialTransform`).
4. Compute metrics vs ground-truth DVF.

**Metrics:**

| Metric | Definition |
|--------|------------|
| **Dice** | Overlap of warped PTV (pred vs GT DVF) |
| **PSNR** | On warped CT volume inside body mask (`skimage.metrics.peak_signal_noise_ratio`, data_range=1) |
| **SSIM** | Structural similarity of warped CT vs GT-warped CT (`skimage.metrics.structural_similarity`) |
| **Centroid shift** | LR/SI/AP PTV centroid error (mm) vs GT warp |
| **det(J) ≤ 0** | Fraction of voxels with non-positive Jacobian determinant (folding) |

**Outputs:** `runs/{scan_id}/eval/metrics.json`, `evaluation_trace.png`, `psnr_ssim_scatter.png`

### 5.2 Breathing-sweep evaluation (synthetic onboard test)

**Script:** `scripts/run_elekta_sweep_eval.py` → LEARN-GUI `ml/tester.py`

**Data:** `ModelTraining/test/{scan_id}/` — 340 projection **sweep** with cycling respiratory phases (`TestProjections/`, `RespBin.csv`).

**Per projection index:**

1. Source = phase 06 projection at that angle.
2. Target = sweep projection (phase from `RespBin.csv`).
3. GT DVF = `DVFs/DVF_{phase:02d}_mha.npy`.
4. Model predicts DVF → warp PTV / volume → same metrics as above.

**Outputs:**

- `runs/{scan_id}/eval_sweep/metrics.json` → copied to `results/{scan_id}/sweep_metrics.json`
- `Performance_Trace.png`, `Performance_Trace_by_index.png`
- `videos/{scan_id}_test_sweep.mp4` — phase-labelled sweep preview
- `videos/{scan_id}_dvf_warp_panels.mp4` — 2×2 DVF warp panel (`scripts/export_dvf_warp_mp4.py`)
- `videos/{scan_id}_dvf_warp_sagittal{N}_ptv.mp4` — sagittal slice with green PTV outline (`--ptv-mask`)

Sweep metrics in the table above (§4.3) are **mean ± std over 340 projection indices** from this test.

### 5.3 Visualization

- **Volume orientation GUI:** `scripts/launch_volume_viewer.py` — pick sagittal/coronal/axial slice; saved in `results/{scan_id}/dvf_view_config.json`
- **DRR verification (optional):** `scripts/export_drr_mp4.py` — compare RTK DRR vs acquired `Proj_*.bin`

---

## 6. Software and reproducibility

| Component | Role |
|-----------|------|
| **VoxelMap_Clinical** | Training, evaluation, staging scripts, Elekta DRR config |
| **LEARN-GUI-Python** | Preprocessing (DRR, DVF, prep_train), sweep `tester.py` |
| **PyTorch** | Training and inference (CUDA) |
| **ITK / ITK-RTK** | DRR generation |
| **ITK-Elastix** | Inter-phase 3D registration (DVFs) |

**Environment variables:**

```bash
export VOXELMAP_CLINICAL_ROOT=/path/to/VoxelMap_Clinical
export LEARN_GUI_ROOT=/path/to/LEARN-GUI-Python
export CUDA_VISIBLE_DEVICES=0
```

**Full per-patient workflow:**

```bash
python scripts/stage_elekta_scan.py CE_P1_V_01
python scripts/run_elekta_phase2.py --scan-id CE_P1_V_01 --with-test
python scripts/run_elekta_phase3_train.py --scan-id CE_P1_V_01 --epochs 50 --gpu 0
python scripts/run_elekta_phase4_eval.py --scan-id CE_P1_V_01 --gpu 0
python scripts/run_elekta_sweep_eval.py --scan-id CE_P1_V_01 --gpu 0
```

Heavy artifacts (`runs/`, `data/staged/`) are gitignored; summary results live in `results/CE_P*_V_01/`.

---

## 7. References

- SPARE challenge dataset and README (`SPARE_PublicArchive/README.md`)
- Shieh et al. — SPARE dataset description
- Shao et al. 2025 (DREME) — Elekta clinical cohort context
- Repository: https://github.com/abbyseb/VoxelMap_Clinical

*Last updated: 2026-07-02 — CE_P1–P5 validation scans, concatenated VoxelMap + FiLM, 50 epochs.*
