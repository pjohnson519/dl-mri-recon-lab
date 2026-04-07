"""
Training script for SimpleVarNet on fastMRI knee data.

Usage:
    python scripts/train.py --config configs/random_4x.yaml \
                            --data_path /path/to/knee \
                            --split_csv /path/to/fastMRI_paired_knee.csv \
                            --output_dir runs/random_4x

Multi-GPU (DDP):
    torchrun --nproc_per_node=4 scripts/train.py --config configs/random_4x.yaml ...
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

# Allow running from repo root: `python scripts/train.py ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.varnet import SimpleVarNet
from utils.data import FastMRIKneeDataset, collate_fn
from utils.metrics import SSIMLoss, ssim_metric


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> SimpleVarNet:
    return SimpleVarNet(
        num_cascades=cfg.get("num_cascades", 8),
        chans=cfg.get("chans", 18),
        pools=cfg.get("pools", 4),
        sens_chans=cfg.get("sens_chans", 8),
        sens_pools=cfg.get("sens_pools", 4),
        use_dc=cfg.get("use_dc", True),
    )


def build_dataset(cfg: dict, data_path: str, split_csv: str, split: str) -> FastMRIKneeDataset:
    return FastMRIKneeDataset(
        data_path=data_path,
        split_csv=split_csv,
        split=split,
        mask_type=cfg.get("mask_type", "random"),
        center_fractions=cfg.get("center_fractions", [0.08]),
        accelerations=cfg.get("accelerations", [4]),
        use_seed=(split != "train"),
    )


def train_epoch(model, loader, optimizer, loss_fn, device, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        masked_kspace, mask, target, max_values, _, _, num_lf = batch
        masked_kspace = masked_kspace.to(device)
        mask = mask.to(device)
        target = target.to(device)

        output = model(masked_kspace, mask, num_low_frequencies=int(num_lf[0]))
        loss = loss_fn(
            output, target,
            max_value=torch.tensor(max_values, dtype=output.dtype, device=device),
        )
        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * masked_kspace.shape[0]

    return total_loss / max(len(loader.dataset), 1)


SKIP_SLICES = 4  # Skip first N slices per volume (noisy edge slices)


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    ssim_total = 0.0
    n = 0
    for batch in loader:
        masked_kspace, mask, target, max_values, _, slice_nums, num_lf = batch
        masked_kspace = masked_kspace.to(device)
        mask = mask.to(device)
        target = target.to(device)

        output = model(masked_kspace, mask, num_low_frequencies=int(num_lf[0]))

        for i in range(output.shape[0]):
            if slice_nums[i] < SKIP_SLICES:
                continue
            ssim_total += ssim_metric(output[i], target[i], max_value=max_values[i])
            n += 1

    return ssim_total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train SimpleVarNet on fastMRI knee data.")
    parser.add_argument("--config",     required=True, help="Path to YAML config file.")
    parser.add_argument("--data_path",  required=True, help="Directory with *.h5 files.")
    parser.add_argument("--split_csv",  required=True, help="Path to fastMRI_paired_knee.csv.")
    parser.add_argument("--output_dir", required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--resume",     default=None,  help="Path to checkpoint to resume from.")
    parser.add_argument("--epochs",     type=int, default=None, help="Override epochs from config.")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps.")
    args = parser.parse_args()

    # DDP setup
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(cfg.get("seed", 42))

    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    train_ds = build_dataset(cfg, args.data_path, args.split_csv, "train")
    val_ds   = build_dataset(cfg, args.data_path, args.split_csv, "val")

    if rank == 0:
        print(f"Train: {len(train_ds)} slices | Val: {len(val_ds)} slices")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 1),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 3e-4))
    epochs = cfg.get("epochs", 50)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.get("lr_step_size", 40),
        gamma=cfg.get("lr_gamma", 0.1),
    )
    loss_fn = SSIMLoss()
    grad_accum = args.grad_accum

    start_epoch = 0
    best_ssim = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        raw_model = model.module if distributed else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_ssim = ckpt.get("best_ssim", 0.0)
        if rank == 0:
            print(f"Resumed from epoch {ckpt['epoch']} (best SSIM {best_ssim:.4f})")

    log_file = None
    writer = None
    if rank == 0:
        log_path = Path(args.output_dir) / "log.csv"
        write_header = not log_path.exists()
        log_file = open(log_path, "a", newline="", buffering=1)
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "val_ssim", "lr"])
        if write_header:
            writer.writeheader()

    if rank == 0:
        print(f"Training on {device} | {world_size} GPU(s) | {epochs} epochs | config: {args.config}")

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, grad_accum)
        val_ssim   = val_epoch(model, val_loader, device)

        if distributed:
            val_ssim_t = torch.tensor(val_ssim, device=device)
            dist.all_reduce(val_ssim_t, op=dist.ReduceOp.AVG)
            val_ssim = float(val_ssim_t)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        if rank == 0:
            print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | val SSIM {val_ssim:.4f} | lr {lr:.2e}")
            writer.writerow({"epoch": epoch, "train_loss": train_loss,
                             "val_ssim": val_ssim, "lr": lr})

            raw_model = model.module if distributed else model
            ckpt = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_ssim": val_ssim,
                "best_ssim": best_ssim,
                "config": cfg,
            }
            torch.save(ckpt, Path(args.output_dir) / "last.pt")
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                torch.save(ckpt, Path(args.output_dir) / "best.pt")
                print(f"  *** New best: {best_ssim:.4f} ***")

    if rank == 0:
        log_file.close()
        print(f"Done. Best val SSIM: {best_ssim:.4f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
