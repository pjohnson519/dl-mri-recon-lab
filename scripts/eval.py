"""
Evaluate a trained SimpleVarNet checkpoint on the test (or val) split.

Usage:
    python scripts/eval.py --checkpoint runs/equi4x/best.pt \
                           --data_path /path/to/knee \
                           --split_csv /path/to/fastMRI_paired_knee.csv

    # Use val split instead of test:
    python scripts/eval.py --checkpoint runs/equi4x/best.pt \
                           --data_path /path/to/knee \
                           --split_csv /path/to/fastMRI_paired_knee.csv \
                           --split val

    # Override mask type (e.g. evaluate a random-trained model on equispaced masks):
    python scripts/eval.py --checkpoint runs/random4x/best.pt \
                           --data_path /path/to/knee \
                           --split_csv /path/to/fastMRI_paired_knee.csv \
                           --mask_type equispaced

    # Save example figures:
    python scripts/eval.py --checkpoint runs/equi4x/best.pt \
                           --data_path /path/to/knee \
                           --split_csv /path/to/fastMRI_paired_knee.csv \
                           --save_figures figures/equi4x
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.varnet import SimpleVarNet
from utils.data import FastMRIKneeDataset, collate_fn
from utils.metrics import ssim_metric


def load_model(checkpoint_path, device, use_dc=None):
    """Load a SimpleVarNet from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]

    # Allow overriding use_dc at eval time
    dc = cfg.get("use_dc", True) if use_dc is None else use_dc

    model = SimpleVarNet(
        num_cascades=cfg.get("num_cascades", 8),
        chans=cfg.get("chans", 18),
        pools=cfg.get("pools", 4),
        sens_chans=cfg.get("sens_chans", 8),
        sens_pools=cfg.get("sens_pools", 4),
        use_dc=dc,
        num_input_slices=cfg.get("num_input_slices", 1),
        num_coils=cfg.get("num_coils", 15),
        num_contrasts=cfg.get("num_contrasts", 1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device), cfg


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Run evaluation and collect per-slice SSIM grouped by volume.

    Returns:
        volume_ssims: dict mapping filename -> list of per-slice SSIM values
    """
    volume_ssims = defaultdict(list)

    for batch in loader:
        masked_kspace, mask, target, max_values, fnames, slice_nums, num_lf = batch
        masked_kspace = masked_kspace.to(device)
        mask = mask.to(device)
        target = target.to(device)

        output = model(masked_kspace, mask, num_low_frequencies=int(num_lf[0]))

        for i in range(output.shape[0]):
            ssim_val = ssim_metric(output[i], target[i], max_value=max_values[i])
            volume_ssims[fnames[i]].append((slice_nums[i], ssim_val))

    return volume_ssims


@torch.no_grad()
def save_example_figures(model, loader, device, output_dir, num_examples=4):
    """Save side-by-side comparison figures for a few example slices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for batch in loader:
        masked_kspace, mask, target, max_values, fnames, slice_nums, num_lf = batch
        masked_kspace = masked_kspace.to(device)
        mask = mask.to(device)

        output = model(masked_kspace, mask, num_low_frequencies=int(num_lf[0]))
        output = output.cpu()

        for i in range(output.shape[0]):
            if saved >= num_examples:
                return

            recon = output[i].numpy()
            gt = target[i].numpy()
            error = np.abs(recon - gt)
            ssim_val = ssim_metric(output[i], target[i], max_value=max_values[i])

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(gt, cmap="gray")
            axes[0].set_title("Ground truth")
            axes[0].axis("off")

            axes[1].imshow(recon, cmap="gray")
            axes[1].set_title(f"Reconstruction (SSIM={ssim_val:.4f})")
            axes[1].axis("off")

            axes[2].imshow(error * 2, cmap="hot")
            axes[2].set_title("Error (2x)")
            axes[2].axis("off")

            vol_name = Path(fnames[i]).stem
            fig.suptitle(f"{vol_name}  slice {slice_nums[i]}", fontsize=11)
            plt.tight_layout()

            fig_path = Path(output_dir) / f"{vol_name}_slice{slice_nums[i]:03d}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved += 1


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SimpleVarNet checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    parser.add_argument("--data_path",  required=True, help="Directory with *.h5 files.")
    parser.add_argument("--split_csv",  required=True, help="Path to fastMRI_paired_knee.csv.")
    parser.add_argument("--split",      default="test", choices=["val", "test"],
                        help="Which split to evaluate on (default: test).")
    parser.add_argument("--mask_type",  default=None,
                        help="Override mask type from config (random or equispaced).")
    parser.add_argument("--no_dc",      action="store_true",
                        help="Disable data consistency (override checkpoint config).")
    parser.add_argument("--save_figures", default=None, metavar="DIR",
                        help="Save example comparison figures to this directory.")
    parser.add_argument("--num_figures", type=int, default=4,
                        help="Number of example figures to save (default: 4).")
    parser.add_argument("--skip_slices", type=int, default=4,
                        help="Skip the first N slices per volume for SSIM "
                             "stats (noisy edge slices). Default: 4. Set to 0 "
                             "to include all slices.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    use_dc = False if args.no_dc else None
    model, cfg = load_model(args.checkpoint, device, use_dc=use_dc)

    # Build dataset — use config mask settings unless overridden
    mask_type = args.mask_type if args.mask_type else cfg.get("mask_type", "random")
    dataset = FastMRIKneeDataset(
        data_path=args.data_path,
        split_csv=args.split_csv,
        split=args.split,
        mask_type=mask_type,
        center_fractions=cfg.get("center_fractions", [0.08]),
        accelerations=cfg.get("accelerations", [4]),
        use_seed=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    skip = args.skip_slices

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split} ({len(dataset)} slices)")
    print(f"Mask       : {mask_type} {cfg.get('accelerations', [4])}x")
    print(f"Skip slices: first {skip} per volume")
    print(f"Device     : {device}")
    print()

    # Run evaluation
    volume_ssims = evaluate(model, loader, device)

    # Per-volume results (skip first N slices per volume — mostly noise)
    all_ssims = []
    print(f"{'Volume':<45s}  {'Slices':>6s}  {'SSIM':>8s}")
    print("-" * 65)
    for fname in sorted(volume_ssims.keys()):
        entries = volume_ssims[fname]
        vals = [ssim for sl, ssim in entries if sl >= skip]
        if not vals:
            continue
        vol_mean = np.mean(vals)
        all_ssims.extend(vals)
        vol_name = Path(fname).stem
        print(f"{vol_name:<45s}  {len(vals):>6d}  {vol_mean:>8.4f}")

    # Summary
    print("-" * 65)
    print(f"{'TOTAL':<45s}  {len(all_ssims):>6d}  {np.mean(all_ssims):>8.4f}")
    print(f"\nMean SSIM: {np.mean(all_ssims):.4f} +/- {np.std(all_ssims):.4f}")

    # Save figures if requested
    if args.save_figures:
        print(f"\nSaving {args.num_figures} example figures to {args.save_figures}/ ...")
        save_example_figures(model, loader, device, args.save_figures, args.num_figures)
        print("Done.")


if __name__ == "__main__":
    main()
