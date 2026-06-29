# VoxelMap network code (vendored from LEARN-GUI)

Copied from `LEARN-GUI/LEARN-GUI-Python/ml/utilities/` on T7 for reference and workstation handoff.

| File | Purpose |
|------|---------|
| `networksFiLM.py` | **Primary** — concatenated / dual / separate / broadcast + FiLM `Model` |
| `networks.py` | Same architectures without FiLM-specific paths |
| `layers.py` | `SpatialTransformer`, `VecInt`, building blocks |
| `modelio.py` | `LoadableModel`, checkpoint helpers |
| `losses.py` | Masked flow loss (used by `trainer.py`) |
| `spatialTransform.py` | Spatial transform module (legacy / related) |

Training on the workstation still uses `LEARN_GUI_ROOT/ml/trainer.py`, which imports these modules as `ml.utilities.*`. To use **this** copy instead, add the repo root to `PYTHONPATH` or install as a package.

**Upstream:** `/Volumes/T7 Shield/DENNIS_BACKUP/LEARN-GUI/LEARN-GUI-Python`
