# Varian SPARE — Concatenated VoxelMap Training Plan

Train **concatenated VoxelMap (+ FiLM)** on **Clinical Varian** SPARE cases (`CV_P*`).

This document mirrors the Elekta workflow documented in `ELEKTA_CONCAT_TRAINING_PLAN.md` and lists what must change in this repo to run on Varian patients instead of Elekta (`CE_*`).

**Workspace:** `VoxelMap_Clinical`  
**Pipeline stack:** `LEARN-GUI/LEARN-GUI-Python`  
**SPARE root (local):** `/home/abhishek/research-data/2RESEARCH/1_ClinicalData/SPAREChallenge`

---

## Scope

| Item | Choice |
|------|--------|
| Dataset | `ClinicalVarianDatasets` (`CV_P*`) |
| Patients | P1–P5 (five independent people; **not** the same people as `CE_P*` or `MC_V_*`) |
| Scans | Start with one validation scan per patient, e.g. `CV_P1_V_01` |
| Architecture | `concatenated` + `--use_film` (same as Elekta) |
| Do **not** reuse | Elekta DRR settings (`512×512`, full-fan, 340 projections) |

### Naming reminder

SPARE patient numbers are **per dataset type only**:

| Prefix | Meaning |
|--------|---------|
| `MC_V_*` | Monte Carlo simulation (Varian half-fan geometry, **simulated** 4D-CT GT) |
| `CV_*` | **Clinical Varian** (half-fan geometry, **FDK reference** GT via registration) |
| `CE_*` | Clinical Elekta (full-fan geometry, FDK reference GT) |

`CV_P1` is not the same person as `CE_P1` or `MC_V_P1`.

---

## Geometry & data differences (Varian vs Elekta)

Verified on local SPARE copy (2026-07-01). See also `Participant_Datasets/ClinicalVarianDatasets/README_DataInfo.txt`.

| Parameter | Clinical Varian (`CV_*`) | Clinical Elekta (`CE_*`) | Monte Carlo (`MC_V_*`) |
|-----------|--------------------------|--------------------------|-------------------------|
| Fan mode | **Half-fan** | Full-fan | Half-fan (same as CV) |
| Projections (1-min sweep) | **680** | 340 | 680 |
| Detector (P1, P2) | **1024×768**, 0.388 mm | 512×512, 0.8 mm | 1024×768, 0.388 mm |
| Detector (P3–P5) | **1008×752**, 0.388 mm | 512×512, 0.8 mm | 1008×752, 0.388 mm |
| Detector lateral offset | **148 mm** (global in XML) | Per-projection offsets | 148 mm |
| SID / SDD | **1000 / 1500 mm** | 1000 / 1536 mm | 1000 / 1500 mm |
| Native recon size | **450×220×450** mm³ @ 1 mm | 270×256×270 @ 1 mm | 450×220×450 @ 1 mm |
| Downsampled grid | **128³** (same pipeline) | 128³ | 128³ |
| Ground truth | FDK full-scan reference (registration DVFs) | FDK full-scan reference | Simulated 4D-CT |
| Reference phase | **06** (max exhale, bin 6) | 06 | 06 |

**Implication:** Varian clinical is **geometry-compatible with MC gold-batch defaults**, but **GT quality matches Elekta clinical** (noisy registration DVFs, not simulation). Do not mix `CV_*` and `CE_*` in one model without harmonization.

---

## SPARE folder layout

### Participant data (projections)

```
Participant_Datasets/ClinicalVarianDatasets/
└── P{n}/
    ├── CV_P{n}_V_01/ … CV_P{n}_V_05/   # validation scans (5 per patient)
    ├── CV_P{n}_T_01/                   # test scan (+ Proj_Full, FDKGroundTruth)
    └── CV_P{n}_Prior/
        └── Proj/
            ├── Geometry.xml
            ├── Proj_*.bin              # 680 projections per scan
            └── RespBin.csv
```

Each validation patient has **5 scans** (`V_01`–`V_05`) → 25 Varian validation scans total.

### Evaluation ground truth (volumes + masks)

```
Evaluation/ClinicalVarianDatasets/
└── P{n}/CV_P{n}_V_01/
    ├── GTVol_01.mha … GTVol_10.mha
    └── Mask_Body.mha, Mask_PTV.mha, Mask_Rib.mha, Mask_CNR.mha, …
```

Participant packs do **not** include `GTVol_*.mha`; merge Evaluation GT into staged tree (same pattern as Elekta).

---

## Required code changes (this repo)

The current Elekta pipeline is hard-coded for `CE_*`. For Varian, add parallel entry points or generalize the existing scripts.

### 1. DRR configuration — `config/`

**Existing:** `config/elekta_drr.py` already defines `MC_VARIAN_DRR_OPTS` (half-fan 1024×768) used by `VoxelMapTestBed` gold batch.

**Add / refactor:**

| File | Action |
|------|--------|
| `config/varian_drr.py` (or extend `elekta_drr.py` → `spare_drr.py`) | Export `varian_drr_opts_for_scan(scan_dir, patient_num)` |
| | P1–P2: `detector_size_xy=(1024, 768)`, origin `(-200, -150, 0)` |
| | P3–P5: `detector_size_xy=(1008, 752)`, same spacing `(0.388, 0.388, 1.0)` |
| | Always set `geometry_path` to `<scan>/Proj/Geometry.xml` (never `Geometry_SPARE.xml`) |

**Verify** detector origin against `CV_P1_V_01/Proj/Geometry.xml` before batch DRR (add `scripts/compare_varian_geometry.py`, analogous to `compare_spare_geometry.py`).

### 2. Staging — `scripts/stage_varian_scan.py`

Mirror `scripts/stage_elekta_scan.py`:

```python
# Paths
participant = spare_root / "Participant_Datasets/ClinicalVarianDatasets" / f"P{pnum}" / scan_id
evaluation  = spare_root / "Evaluation/ClinicalVarianDatasets" / f"P{pnum}" / scan_id
out         = staged_root / f"P{pnum}" / scan_id

# Symlink Proj/ + GTVol_*.mha + Mask_*.mha
```

**Scan ID regex:** `CV_P(\d+)_` instead of `CE_P(\d+)_`.

Example:

```bash
python scripts/stage_varian_scan.py CV_P1_V_01 CV_P2_V_01 \
  --spare-root "$SPARE_PUBLIC_ARCHIVE"
```

### 3. Preprocessing — `scripts/run_varian_phase2.py`

Copy `scripts/run_elekta_phase2.py` with these substitutions:

| Elekta (current) | Varian (needed) |
|------------------|-----------------|
| `elekta_drr_opts_for_scan()` | `varian_drr_opts_for_scan()` |
| `_patient_num`: `CE_P(\d+)_` | `CV_P(\d+)_` |
| `dataset_type="clinical"` in DRR | Same (keep `clinical`) |
| Downsample, DVF, prep_train | **Unchanged** — same LEARN-GUI modules, `dataset_type="spare"` |

**Runtime note:** Varian DRR is ~**4–6× slower** per scan than Elekta (680 vs 340 projections, larger detector). Budget **~15–30 min DRR + Elastix** per scan on RTX 6000 Ada (estimate; measure on `CV_P1_V_01` smoke test).

**Expected `ModelTraining` scale (one scan):**

```
runs/CV_P1_V_01/ModelTraining/train/CV_P1_V_01/
├── SourceProjections/    680   (phase 06)
├── TargetProjections/   6120   (9 phases × 680 angles)
├── DVFs/                   9
├── Masks/                  4+
└── Angles.csv            680
```

Use `--with-test` to also build the 680-projection breathing sweep under `ModelTraining/test/`.

### 4. Training / eval / export — mostly scan-id agnostic

These scripts already take `--scan-id` and resolve paths under `runs/{scan_id}/`:

| Script | Varian change |
|--------|---------------|
| `run_elekta_phase3_train.py` | None (or rename → `run_phase3_train.py`) |
| `run_elekta_phase4_eval.py` | None |
| `run_elekta_sweep_eval.py` | None |
| `export_dvf_warp_mp4.py` | None; PTV slice finder works on `Mask_PTV_mha.npy` |
| `gui/volume_orientation_viewer.py` | None |

### 5. Batch orchestration

Add `scripts/run_varian_batch_train_eval.sh` — same as `run_elekta_batch_train_eval.sh` but:

```bash
SCANS=(CV_P1_V_01 CV_P2_V_01 CV_P3_V_01 CV_P4_V_01 CV_P5_V_01)
# call run_varian_phase2.py (once) then phase3/4/sweep per scan
```

### 6. Documentation / verification

| Artifact | Purpose |
|----------|---------|
| `VARIAN_DRR_VERIFICATION.md` | DRR smoke test vs native `Proj_*.bin` |
| `scripts/compare_varian_geometry.py` | CV vs MC vs CE geometry table |
| `results/CV_P*_V_01/` | Metrics, plots, DVF MP4s (git-tracked summaries) |

---

## Training plan (recommended order)

### Phase 0 — Prerequisites

- [ ] SPARE Varian archives extracted (or selective 7z extract like Elekta)
- [ ] LEARN-GUI venv with CUDA PyTorch, ITK-RTK, Elastix
- [ ] ~**50–80 GB** free per staged patient (5 scans × ~10 GB each, rough estimate)
- [ ] Env vars: `VOXELMAP_CLINICAL_ROOT`, `LEARN_GUI_ROOT`, `CUDA_VISIBLE_DEVICES=0`

Archives (if not already extracted locally):

```
Participant_Datasets/ClinicalVarianDatasets.7z.*
Evaluation/SPARE_GroundTruth.7z.*   # contains ClinicalVarianDatasets GT
```

Password: `#MakeIt1Minute!` (see SPARE README).

### Phase 1 — Implement Varian pipeline (1–2 days dev)

1. [ ] Add `config/varian_drr.py` with P1/P2 vs P3–P5 detector sizes
2. [ ] Add `scripts/stage_varian_scan.py`
3. [ ] Add `scripts/run_varian_phase2.py` (fork from Elekta phase2)
4. [ ] DRR smoke test on `CV_P1_V_01` — compare one RTK DRR to `Proj_00001.bin`
5. [ ] Document verification in `VARIAN_DRR_VERIFICATION.md`

### Phase 2 — Single-patient pilot (`CV_P1_V_01`)

```bash
export VOXELMAP_CLINICAL_ROOT=/home/abhishek/Documents/VoxelMap_Clinical
export LEARN_GUI_ROOT=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python
export CUDA_VISIBLE_DEVICES=0

# Stage
python scripts/stage_varian_scan.py CV_P1_V_01

# Preprocess (DRR → compress → DVF → prep_train)
python scripts/run_varian_phase2.py --scan-id CV_P1_V_01 --with-test

# Train 50 epochs
python scripts/run_elekta_phase3_train.py --scan-id CV_P1_V_01 --epochs 50 --gpu 0

# Eval + sweep + MP4
python scripts/run_elekta_phase4_eval.py --scan-id CV_P1_V_01 --gpu 0
python scripts/run_elekta_sweep_eval.py --scan-id CV_P1_V_01 --gpu 0

# Sagittal DVF + PTV outline (find peak slice first)
python scripts/export_dvf_warp_mp4.py \
  --scan-id CV_P1_V_01 --plane sagittal --slice-index <ptv_peak> \
  --ptv-mask --out runs/CV_P1_V_01/videos/CV_P1_V_01_dvf_warp_sagittal<N>_ptv.mp4
```

**Success criteria (compare to Elekta P1 baseline):**

| Metric | Elekta `CE_P1_V_01` (reference) | Varian target (pilot) |
|--------|-----------------------------------|------------------------|
| Best val loss (50 ep) | 0.057 | ≤ 0.12 (clinical GT is noisier) |
| Sweep mean Dice | see `results/CE_P1_V_01/sweep_metrics.json` | within ~10% of Elekta or document gap |
| DVF warp MP4 | PTV visible on sagittal peak slice | Same |

### Phase 3 — All five patients (`CV_P1`–`CV_P5`, one scan each)

```bash
for scan in CV_P1_V_01 CV_P2_V_01 CV_P3_V_01 CV_P4_V_01 CV_P5_V_01; do
  python scripts/stage_varian_scan.py "$scan"
  python scripts/run_varian_phase2.py --scan-id "$scan" --with-test
done

bash scripts/run_varian_batch_train_eval.sh
```

Export results to `results/CV_P*_V_01/` and push (metrics, loss curves, DVF MP4s with PTV outline).

### Phase 4 — Multi-scan per patient (optional, more data)

Each Varian patient has **5 validation scans**. After single-scan baselines:

```bash
python scripts/run_elekta_phase3_train.py \
  --scan-id CV_P1_V_01 \
  --data-dirs \
    runs/CV_P1_V_01/ModelTraining/train/CV_P1_V_01 \
    runs/CV_P1_V_02/ModelTraining/train/CV_P1_V_02 \
  ...
```

(Requires extending phase3 script to accept multiple `--data-dirs`, or merge `ModelTraining` trees.)

### Phase 5 — Cross-scanner analysis (research, not production)

| Experiment | Purpose |
|------------|---------|
| Train on `CV_*` only | Varian-specific model |
| Train on `CE_*` only | Elekta-specific model (done) |
| Fine-tune CE → CV | Scanner transfer |
| **Do not** naively pool CE + CV | Different geometry, intensity, noise |

---

## Workflow overview

```
Extract CV + GT  →  stage_varian_scan  →  run_varian_phase2 (Varian DRR)
       →  prep_train  →  phase3_train (concat + FiLM)  →  sweep eval + DVF MP4
       →  results/CV_P*_V_01/  →  git push
```

```
Elekta pipeline (existing)          Varian pipeline (to build)
─────────────────────────          ───────────────────────────
stage_elekta_scan.py        →      stage_varian_scan.py
run_elekta_phase2.py        →      run_varian_phase2.py
config/elekta_drr.py        →      config/varian_drr.py (+ MC_VARIAN reuse)
run_elekta_phase3_train.py  →      (reuse as-is)
run_elekta_sweep_eval.py    →      (reuse as-is)
export_dvf_warp_mp4.py      →      (reuse as-is)
```

---

## Known risks (clinical Varian)

1. **Registration GT noise** — same class of issue as Elekta; expect higher val loss than MC simulation.
2. **680-angle sweep** — longer DRR/prep; 2× training pairs vs Elekta per scan.
3. **Detector size split** — P1/P2 vs P3–P5 need separate DRR panel configs; wrong size silently misaligns DRR.
4. **Half-fan vs full-fan** — never pass Elekta `elekta_drr_opts_for_scan()` to a `CV_*` scan.
5. **Patient ID confusion** — always use full scan id (`CV_P3_V_01`), not bare `P3`.

---

## Quick reference — file paths

| Resource | Path |
|----------|------|
| Varian README (detector sizes) | `SPAREChallenge/Participant_Datasets/ClinicalVarianDatasets/README_DataInfo.txt` |
| Staged scan (target) | `data/staged/P{n}/CV_P{n}_V_01/` |
| Run output | `runs/CV_P{n}_V_01/` |
| Git-tracked results | `results/CV_P{n}_V_01/` |
| MC/Varian DRR defaults (existing) | `config/elekta_drr.py` → `MC_VARIAN_DRR_OPTS` |
| Elekta plan (template) | `ELEKTA_CONCAT_TRAINING_PLAN.md` |
| Agent runbook | `NOTE_TO_AGENT.md` |

---

## Implementation checklist (copy to issue tracker)

- [ ] `config/varian_drr.py`
- [ ] `scripts/stage_varian_scan.py`
- [ ] `scripts/run_varian_phase2.py`
- [ ] `scripts/compare_varian_geometry.py`
- [ ] `VARIAN_DRR_VERIFICATION.md`
- [ ] `scripts/run_varian_batch_train_eval.sh`
- [ ] Pilot: `CV_P1_V_01` end-to-end
- [ ] Batch: `CV_P1_V_01` … `CV_P5_V_01`
- [ ] DVF sagittal + PTV MP4s → `results/`
- [ ] Git push results summaries
