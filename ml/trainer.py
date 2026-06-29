"""VoxelMap trainer with per-epoch checkpoints, best-model save, and loss-curve PNGs."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.dynamic_dataset import VoxelMapDataset, resolve_voxel_map_data_root
from ml.training_config import (
    REFERENCE_BATCH_SIZE,
    REFERENCE_EPOCHS,
    REFERENCE_IM_SIZE,
    REFERENCE_LR,
    REFERENCE_USE_MIXED_PRECISION,
    REFERENCE_VAL_SPLIT,
)
from ml.utilities import losses, networksFiLM


def get_args():
    p = argparse.ArgumentParser(description="VoxelMap Clinical trainer")
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--batch_size", type=int, default=REFERENCE_BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=REFERENCE_EPOCHS)
    p.add_argument(
        "--architecture",
        type=str,
        default="concatenated",
        choices=["concatenated", "dual", "separate", "broadcast"],
    )
    p.add_argument("--lr", type=float, default=REFERENCE_LR)
    p.add_argument("--val_split", type=float, default=REFERENCE_VAL_SPLIT)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_film", action="store_true")
    p.add_argument("--use_amp", action="store_true", default=False)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=Path, required=True)
    p.add_argument("--plots_dir", type=Path, required=True)
    p.add_argument("--save_path", type=Path, default=None, help="Final symlink target (best model)")
    return p.parse_args()


def _run_epoch(model, loader, device, use_film, use_amp, masked_flow_loss, mse_loss, optimizer=None, scaler=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_sum = 0.0
    n_batches = 0

    ctx = torch.no_grad() if not train_mode else torch.enable_grad()
    with ctx:
        for batch_data in loader:
            if use_film:
                src_proj, tgt_proj, src_vol, src_mask, tgt_flow, angles = batch_data
                angles = angles.to(device)
            else:
                src_proj, tgt_proj, src_vol, src_mask, tgt_flow = batch_data
                angles = None

            src_proj = src_proj.to(device)
            tgt_proj = tgt_proj.to(device)
            src_vol = src_vol.to(device)
            tgt_flow = tgt_flow.to(device)
            if src_mask is not None:
                src_mask = src_mask.to(device)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            def _forward_loss():
                if use_film:
                    _, pred_flow = model(src_proj, tgt_proj, src_vol, angle=angles)
                else:
                    _, pred_flow = model(src_proj, tgt_proj, src_vol)
                if src_mask is not None and src_mask.shape[-3:] == pred_flow.shape[-3:]:
                    return masked_flow_loss.loss(tgt_flow, pred_flow, src_mask)
                return mse_loss(pred_flow, tgt_flow)

            amp_device = "cuda" if "cuda" in device else "cpu"
            if use_amp and train_mode:
                with torch.autocast(device_type=amp_device, dtype=torch.float16):
                    loss = _forward_loss()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif use_amp:
                with torch.autocast(device_type=amp_device, dtype=torch.float16):
                    loss = _forward_loss()
            elif train_mode:
                loss = _forward_loss()
                loss.backward()
                optimizer.step()
            else:
                loss = _forward_loss()

            loss_sum += loss.item()
            n_batches += 1

    return loss_sum / n_batches if n_batches else 0.0


def save_loss_plot(train_losses, val_losses, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train", color="#2563eb", linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, label="val", color="#dc2626", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = get_args()
    use_amp = args.use_amp or REFERENCE_USE_MIXED_PRECISION

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    ckpt_dir = args.checkpoint_dir.resolve()
    epoch_dir = ckpt_dir / "epochs"
    plots_dir = args.plots_dir.resolve()
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    loss_png = plots_dir / "loss_curves.png"
    history_json = plots_dir / "loss_history.json"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("--- VoxelMap Clinical Trainer ---")
    print(f"Device: {args.device} | AMP: {use_amp} | FiLM: {args.use_film}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Loss plots: {plots_dir}")

    resolved = [resolve_voxel_map_data_root(Path(d)) for d in args.data_dirs]
    dataset = VoxelMapDataset(resolved, use_angles=args.use_film, im_size=REFERENCE_IM_SIZE)
    print(f"Discovered {len(dataset)} projection pairs.")
    if len(dataset) == 0:
        raise SystemExit("ERROR: Dataset is empty.")

    n = len(dataset)
    val_n = int(n * args.val_split) if args.val_split > 0 else 0
    train_n = n - val_n
    if val_n > 0:
        train_ds, val_ds = random_split(dataset, [train_n, val_n])
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        )
        print(f"Split: train={train_n} | val={val_n}")
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        )
        val_loader = None
        print("Split: full dataset used for training (no val).")

    model = networksFiLM.Model(
        architecture=args.architecture,
        im_size=REFERENCE_IM_SIZE,
        in_channels=1,
        use_film=args.use_film,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()
    masked_flow_loss = losses.flow_mask()
    scaler = torch.amp.GradScaler("cuda" if "cuda" in args.device else "cpu") if use_amp else None

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_metric = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        start_time = time.time()
        avg_train = _run_epoch(
            model,
            train_loader,
            args.device,
            args.use_film,
            use_amp,
            masked_flow_loss,
            mse_loss,
            optimizer=optimizer,
            scaler=scaler,
        )

        avg_val = None
        if val_loader is not None:
            avg_val = _run_epoch(
                model,
                val_loader,
                args.device,
                args.use_film,
                use_amp,
                masked_flow_loss,
                mse_loss,
            )

        train_losses.append(avg_train)
        if avg_val is not None:
            val_losses.append(avg_val)

        epoch_num = epoch + 1
        epoch_path = epoch_dir / f"epoch_{epoch_num:03d}.pt"
        model.save(str(epoch_path))
        model.save(str(last_path))

        metric = avg_val if avg_val is not None else avg_train
        is_best = metric < best_metric
        if is_best:
            best_metric = metric
            best_epoch = epoch_num
            model.save(str(best_path))

        save_loss_plot(train_losses, val_losses, loss_png)
        history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        }
        history_json.write_text(json.dumps(history, indent=2), encoding="utf-8")

        elapsed = time.time() - start_time
        best_tag = " *best*" if is_best else ""
        if avg_val is not None:
            print(
                f"Epoch {epoch_num}/{args.epochs} | Train: {avg_train:.4f} | "
                f"Val: {avg_val:.4f} | {elapsed:.1f}s{best_tag}"
            )
        else:
            print(f"Epoch {epoch_num}/{args.epochs} | Train: {avg_train:.4f} | {elapsed:.1f}s{best_tag}")
        print(f"  saved {epoch_path.name} | plot -> {loss_png}")

    if args.save_path:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        if best_path.exists():
            import shutil

            shutil.copy2(best_path, args.save_path)
            print(f"Best model copied to {args.save_path}")

    print(f"--- Done | best val/train loss: {best_metric:.4f} ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
