# VoxelMap + FiLM — Patient Evaluation Metrics

Mean metrics from **breathing-sweep evaluation** (680 test projections per scan) for all SPARE Clinical patients trained with the VoxelMap + FiLM concatenated model.

| Metric | Description |
|--------|-------------|
| **3D Error** | Mean PTV centroid displacement error (mm), computed as the Euclidean distance between predicted and ground-truth shifts |
| **SSIM** | Structural Similarity Index on warped CT volumes |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) on warped CT volumes |

Source files: `results/<scan_id>/sweep_metrics.json`

---

## Clinical Elekta (`CE_P*`)

| Patient | Scan ID | 3D Error (mm) | SSIM | PSNR (dB) |
|---------|---------|---------------|------|-----------|
| P1 | `CE_P1_V_01` | 0.805 | 0.8887 | 36.87 |
| P2 | `CE_P2_V_01` | 2.268 | 0.9241 | 36.38 |
| P3 | `CE_P3_V_01` | 0.872 | 0.9310 | 37.71 |
| P4 | `CE_P4_V_01` | 0.704 | 0.8710 | 36.87 |
| P5 | `CE_P5_V_01` | 1.727 | 0.9141 | 43.08 |

---

## Clinical Varian (`CV_P*`)

| Patient | Scan ID | 3D Error (mm) | SSIM | PSNR (dB) |
|---------|---------|---------------|------|-----------|
| P1 | `CV_P1_V_01` | 3.033 | 0.8308 | 31.83 |
| P2 | `CV_P2_V_01` | 0.549 | 0.8281 | 44.98 |
| P3 | `CV_P3_V_01` | 0.462 | 0.8850 | 51.56 |
| P4 | `CV_P4_V_01` | 0.656 | 0.8395 | 53.16 |
| P5 | `CV_P5_V_01` | 0.733 | 0.8615 | 45.06 |

---

## All patients (combined)

| Vendor | Patient | Scan ID | 3D Error (mm) | SSIM | PSNR (dB) |
|--------|---------|---------|---------------|------|-----------|
| Elekta | P1 | `CE_P1_V_01` | 0.805 | 0.8887 | 36.87 |
| Elekta | P2 | `CE_P2_V_01` | 2.268 | 0.9241 | 36.38 |
| Elekta | P3 | `CE_P3_V_01` | 0.872 | 0.9310 | 37.71 |
| Elekta | P4 | `CE_P4_V_01` | 0.704 | 0.8710 | 36.87 |
| Elekta | P5 | `CE_P5_V_01` | 1.727 | 0.9141 | 43.08 |
| Varian | P1 | `CV_P1_V_01` | 3.033 | 0.8308 | 31.83 |
| Varian | P2 | `CV_P2_V_01` | 0.549 | 0.8281 | 44.98 |
| Varian | P3 | `CV_P3_V_01` | 0.462 | 0.8850 | 51.56 |
| Varian | P4 | `CV_P4_V_01` | 0.656 | 0.8395 | 53.16 |
| Varian | P5 | `CV_P5_V_01` | 0.733 | 0.8615 | 45.06 |
