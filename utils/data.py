"""
FastMRI knee multicoil dataset.

Returned sample:
    masked_kspace  : (coils, H, W, 2)  float32
    mask           : (1, 1, W, 1)      float32  -- 0/1 sampling mask
    target         : (rH, rW)          float32  -- RSS target
    scale          : float             -- per-volume max RSS value
    fname          : str
    slice_num      : int
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.transforms import (
    to_tensor,
    apply_mask,
    RandomMaskFunc,
    EquispacedMaskFractionFunc,
    MaskFunc,
)


class FastMRIKneeDataset(Dataset):
    """
    Multicoil fastMRI knee dataset backed by HDF5 files.

    Args:
        data_path:       Directory containing the *.h5 files.
        split_csv:       Path to fastMRI_paired_knee.csv.
        split:           One of 'train', 'val', 'test'.
        mask_type:       'random' or 'equispaced'.
        center_fractions: ACS fraction(s), e.g. [0.08].
        accelerations:   Acceleration factor(s), e.g. [4].
        use_seed:        If True, mask seed is derived from fname for
                         reproducible val/test masks.
        max_slices:      Cap the dataset at this many slices (for debugging).
    """

    def __init__(
        self,
        data_path: str,
        split_csv: str,
        split: str = "train",
        mask_type: str = "random",
        center_fractions: List[float] = None,
        accelerations: List[int] = None,
        use_seed: bool = False,
        max_slices: Optional[int] = None,
    ):
        if center_fractions is None:
            center_fractions = [0.08]
        if accelerations is None:
            accelerations = [4]

        self.data_path = Path(data_path)
        self.split = split
        self.use_seed = use_seed
        self.mask_type = mask_type

        self.mask_func_random = RandomMaskFunc(center_fractions, accelerations)
        self.mask_func_equispaced = EquispacedMaskFractionFunc(center_fractions, accelerations)

        # Load split CSV and collect file list (both PD and PDFS contrasts)
        df = pd.read_csv(split_csv)
        pdfs_files = df.loc[df["pdfs_split"] == split, "pdfs"].dropna().tolist()
        pd_files = df.loc[df["pd_split"] == split, "pd"].dropna().tolist()
        split_files = list(set(pdfs_files + pd_files))

        # Build (fname, slice_num) index
        self.examples: List[Tuple[str, int]] = []
        for fname in split_files:
            full_path = str(self.data_path / fname)
            if not os.path.exists(full_path):
                continue
            try:
                with h5py.File(full_path, "r") as f:
                    num_slices = f["kspace"].shape[0]
            except Exception:
                continue
            self.examples.extend((full_path, sl) for sl in range(num_slices))

        if max_slices is not None:
            self.examples = self.examples[:max_slices]

    def __len__(self) -> int:
        return len(self.examples)

    def _pick_mask_func(self) -> MaskFunc:
        if self.mask_type == "random":
            return self.mask_func_random
        elif self.mask_type == "equispaced":
            return self.mask_func_equispaced
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

    def __getitem__(self, idx: int):
        fname, slice_num = self.examples[idx]

        with h5py.File(fname, "r") as f:
            kspace_np = f["kspace"][slice_num]
            target_np = f["reconstruction_rss"][slice_num]
            max_value = f.attrs["max"]

        kspace = to_tensor(kspace_np)
        target = torch.from_numpy(target_np.astype(np.float32))

        mask_func = self._pick_mask_func()
        seed = tuple(map(ord, os.path.basename(fname))) if self.use_seed else None
        masked_kspace, mask, num_low_freqs = apply_mask(kspace, mask_func, seed=seed)

        return masked_kspace, mask, target, float(max_value), fname, slice_num, num_low_freqs


def collate_fn(batch):
    """Stack batch items into tensors."""
    masked_kspaces, masks, targets, scales, fnames, slice_nums, nlfs = zip(*batch)
    return (
        torch.stack(masked_kspaces),
        torch.stack(masks),
        torch.stack(targets),
        list(scales),
        list(fnames),
        list(slice_nums),
        list(nlfs),
    )
