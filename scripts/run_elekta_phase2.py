#!/usr/bin/env python3
"""Phase 2 Elekta pipeline: downsample → DRR → compress → DVF → prep_train."""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/Volumes/T7 Shield/DENNIS_BACKUP/LEARN-GUI/LEARN-GUI-Python"))
sys.path.insert(0, str(LEARN))
sys.path.insert(0, str(REPO / "config"))

from elekta_drr import elekta_drr_opts_for_scan  # noqa: E402

SCAN_ID = "CE_P1_V_01"
DEFAULT_RUN = REPO / "runs" / SCAN_ID


def _load_spare_d2m():
    spec = importlib.util.spec_from_file_location(
        "spare_d2m", LEARN / "modules/dicom2mha/implementations/spare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


def ensure_layout(run_root: Path, staged: Path) -> Path:
    """prep_train expects run_root/<scan_id>/train/."""
    train = run_root / SCAN_ID / "train"
    flat_train = run_root / "train"
    if flat_train.is_dir() and not train.is_dir():
        train.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(flat_train), str(train))
        logging.info("Moved train/ → %s/train/", SCAN_ID)
    train.mkdir(parents=True, exist_ok=True)

    # Masks + Proj geometry from staged if missing
    for pat in ("Mask_*.mha",):
        for src in staged.glob(pat):
            dst = train / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
    proj = train / "Proj"
    if not (proj / "Geometry.xml").is_file() and (staged / "Proj" / "Geometry.xml").is_file():
        if proj.exists():
            shutil.rmtree(proj)
        shutil.copytree(staged / "Proj", proj)
    return train


def run_downsample(train: Path) -> int:
    from modules.downsampling.downsample import process_directory

    process_directory(str(train))
    return len(list(train.glob("sub_CT_*.mha")))


def run_drr(train: Path, geom_xml: Path) -> int:
    from modules.drr_generation.run import run as run_drr_fn

    opts = elekta_drr_opts_for_scan(train)
    opts["geometry_path"] = str(geom_xml)
    mhas = sorted(train.glob("CT_*.mha"))
    for i, mha in enumerate(mhas, start=1):
        m = re.search(r"(\d+)", mha.stem)
        ct_num = int(m.group(1)) if m else i
        logging.info("DRR %s (%d/%d)", mha.name, i, len(mhas))
        ok, err = run_drr_fn(
            str(mha),
            str(train),
            geometry_file=str(geom_xml),
            ct_num=ct_num,
            dataset_type="clinical",
            **opts,
        )
        if not ok:
            raise RuntimeError(f"DRR failed {mha.name}: {err}")
    return len(mhas)


def run_compress(train: Path) -> int:
    from modules.drr_compression.compress import process_directory

    process_directory(train)
    return len(list(train.glob("*_Proj_*.bin")))


def run_dvf(train: Path) -> int:
    from modules.dvf_generation.run import run as run_dvf_fn

    param = LEARN / "modules/dvf_generation/Elastix_BSpline_Sliding_LowRes.txt"
    fixed = train / "sub_CT_06.mha"
    if not fixed.is_file():
        raise FileNotFoundError(fixed)
    jobs = []
    for m in sorted(train.glob("sub_CT_*.mha")):
        num_m = re.search(r"(\d+)", m.stem)
        num = num_m.group(1) if num_m else ""
        if num == "06":
            continue
        jobs.append((fixed, m, param, train / f"DVF_sub_{num.zfill(2)}.mha"))
    os.environ.setdefault("ITK_NUMBER_OF_THREADS", "1")
    workers = min(2, max(1, len(jobs)))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(run_dvf_fn, f, m, p, o, dataset_type="spare"): m.name
            for f, m, p, o in jobs
        }
        for fut in as_completed(futs):
            fut.result()
            done += 1
            logging.info("DVF %d/%d %s", done, len(jobs), futs[fut])
    return done


def run_prep(run_root: Path, train: Path, geom_xml: Path) -> None:
    from modules.prep_train.run import run_prep_train

    def log(msg):
        logging.info(msg)

    run_prep_train(
        run_root,
        {SCAN_ID: "train"},
        dataset_type="spare",
        on_log=log,
        angles_xml_path=geom_xml,
        prefer_patient_xml=False,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Elekta SPARE phase 2 pipeline")
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    p.add_argument(
        "--staged",
        type=Path,
        default=REPO / "data/staged/P1/CE_P1_V_01",
    )
    p.add_argument("--skip-drr", action="store_true", help="Resume after DRR")
    p.add_argument("--skip-dvf", action="store_true", help="Resume after DVF")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_root = args.run_root.resolve()
    staged = args.staged.resolve()
    train = ensure_layout(run_root, staged)
    geom_xml = train / "Proj" / "Geometry.xml"
    if not geom_xml.is_file():
        raise FileNotFoundError(geom_xml)

    t0 = time.perf_counter()
    spare_run = _load_spare_d2m()
    spare_run(staged, train)

    logging.info("=== DOWNSAMPLE ===")
    n_sub = run_downsample(train)
    logging.info("sub_CT count: %d", n_sub)

    if not args.skip_drr:
        logging.info("=== DRR (Elekta 512×512, %s) ===", geom_xml.name)
        run_drr(train, geom_xml)

    logging.info("=== COMPRESS ===")
    n_bin = run_compress(train)
    logging.info("projection bins: %d", n_bin)

    if not args.skip_dvf:
        logging.info("=== DVF3D_LOW ===")
        run_dvf(train)

    logging.info("=== PREP_TRAIN ===")
    run_prep(run_root, train, geom_xml)

    mt = run_root / "ModelTraining" / "train" / SCAN_ID
    logging.info("Done in %.1f min", (time.perf_counter() - t0) / 60)
    logging.info("ModelTraining: %s", mt)
    if mt.is_dir():
        for sub in ("SourceProjections", "TargetProjections", "DVFs", "Angles.csv"):
            pth = mt / sub if sub != "Angles.csv" else mt / sub
            if pth.exists():
                n = len(list(pth.glob("*"))) if pth.is_dir() else 1
                logging.info("  %s: %s", sub, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
