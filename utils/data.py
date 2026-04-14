"""
FastMRI knee multicoil dataset.

FastMRIKneeDataset (single-slice):
    masked_kspace  : (coils, H, W, 2)  float32
    mask           : (1, 1, W, 1)      float32  -- 0/1 sampling mask
    target         : (rH, rW)          float32  -- RSS target
    scale          : float             -- per-volume max RSS value
    fname          : str
    slice_num      : int

MultiSliceDataset (neighbouring-slice):
    masked_kspace  : (3*coils, H, W, 2)  float32  -- 3 adjacent slices stacked on coil dim
    mask           : (1, 1, W, 1)        float32
    target         : (rH, rW)            float32  -- center slice RSS target
    scale          : float
    fname          : str
    slice_num      : int

PairedContrastDataset (joint PD+PDFS, multi-slice):
    masked_kspace  : (6*coils, H, W, 2)  float32  -- 3 PD slices + 3 PDFS slices on coil dim
    mask           : (1, 1, W, 1)        float32
    target_pd      : (rH, rW)            float32  -- PD center slice RSS target
    target_pdfs    : (rH, rW)            float32  -- PDFS center slice RSS target
    scale          : float               -- max of both volumes
    pd_fname       : str
    pdfs_fname     : str
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
        split_files = sorted(set(pdfs_files + pd_files))

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


class MultiSliceDataset(Dataset):
    """
    Multicoil fastMRI knee dataset that loads 3 adjacent slices per sample.

    Slices [s-1, s, s+1] are stacked along the coil dimension so the output
    k-space has shape (3*coils, H, W, 2). The target is the RSS
    reconstruction of the center slice s only.  Edge slices are
    replicate-padded (slice 0 duplicates itself, last slice likewise).

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
        split_files = sorted(set(pdfs_files + pd_files))

        # Build (fname, slice_num, num_slices) index
        self.examples: List[Tuple[str, int, int]] = []
        for fname in split_files:
            full_path = str(self.data_path / fname)
            if not os.path.exists(full_path):
                continue
            try:
                with h5py.File(full_path, "r") as f:
                    num_slices = f["kspace"].shape[0]
            except Exception:
                continue
            self.examples.extend(
                (full_path, sl, num_slices) for sl in range(num_slices)
            )

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
        fname, slice_num, num_slices = self.examples[idx]

        # Compute neighbour indices with replicate padding at edges
        prev_sl = max(slice_num - 1, 0)
        next_sl = min(slice_num + 1, num_slices - 1)

        with h5py.File(fname, "r") as f:
            kspace_prev = f["kspace"][prev_sl]
            kspace_curr = f["kspace"][slice_num]
            kspace_next = f["kspace"][next_sl]
            target_np = f["reconstruction_rss"][slice_num]
            max_value = f.attrs["max"]

        # Stack 3 slices along coil dimension: (3*coils, H, W, 2)
        kspace = np.concatenate([kspace_prev, kspace_curr, kspace_next], axis=0)
        kspace = to_tensor(kspace)
        target = torch.from_numpy(target_np.astype(np.float32))

        mask_func = self._pick_mask_func()
        seed = tuple(map(ord, os.path.basename(fname))) if self.use_seed else None
        masked_kspace, mask, num_low_freqs = apply_mask(kspace, mask_func, seed=seed)

        return masked_kspace, mask, target, float(max_value), fname, slice_num, num_low_freqs


class PairedContrastDataset(Dataset):
    """
    Joint PD + PDFS dataset with multi-slice loading.

    Each sample loads 3 adjacent slices from both the PD and PDFS volumes
    of the same exam.  The slices are stacked along the coil dimension:
    [PD_prev, PD_curr, PD_next, PDFS_prev, PDFS_curr, PDFS_next] giving
    shape (6*coils, H, W, 2).  Targets are the center-slice RSS images
    from both contrasts.

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

        # Load split CSV — only keep rows where both PD and PDFS exist
        df = pd.read_csv(split_csv)
        paired = df.loc[
            (df["pd_split"] == split) & (df["pdfs_split"] == split)
        ].dropna(subset=["pd", "pdfs"])

        # Build (pd_path, pdfs_path, slice_num, num_slices) index
        self.examples: List[Tuple[str, str, int, int]] = []
        for _, row in paired.iterrows():
            pd_path = str(self.data_path / row["pd"])
            pdfs_path = str(self.data_path / row["pdfs"])
            if not (os.path.exists(pd_path) and os.path.exists(pdfs_path)):
                continue
            try:
                with h5py.File(pd_path, "r") as f:
                    num_slices = f["kspace"].shape[0]
            except Exception:
                continue
            self.examples.extend(
                (pd_path, pdfs_path, sl, num_slices) for sl in range(num_slices)
            )

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

    def _load_3_slices(self, fname: str, slice_num: int, num_slices: int):
        """Load 3 adjacent slices, replicate-padding at edges."""
        prev_sl = max(slice_num - 1, 0)
        next_sl = min(slice_num + 1, num_slices - 1)
        with h5py.File(fname, "r") as f:
            kspace = np.concatenate([
                f["kspace"][prev_sl],
                f["kspace"][slice_num],
                f["kspace"][next_sl],
            ], axis=0)
            target = f["reconstruction_rss"][slice_num]
            max_value = float(f.attrs["max"])
        return kspace, target, max_value

    def __getitem__(self, idx: int):
        pd_fname, pdfs_fname, slice_num, num_slices = self.examples[idx]

        pd_kspace, pd_target, pd_max = self._load_3_slices(pd_fname, slice_num, num_slices)
        pdfs_kspace, pdfs_target, pdfs_max = self._load_3_slices(pdfs_fname, slice_num, num_slices)

        # Stack both contrasts along coil dim: (6*coils, H, W, 2)
        kspace = np.concatenate([pd_kspace, pdfs_kspace], axis=0)
        kspace = to_tensor(kspace)
        target_pd = torch.from_numpy(pd_target.astype(np.float32))
        target_pdfs = torch.from_numpy(pdfs_target.astype(np.float32))

        # Same mask for both contrasts
        mask_func = self._pick_mask_func()
        seed = tuple(map(ord, os.path.basename(pd_fname))) if self.use_seed else None
        masked_kspace, mask, num_low_freqs = apply_mask(kspace, mask_func, seed=seed)

        return (masked_kspace, mask, target_pd, target_pdfs,
                pd_max, pdfs_max, pd_fname, pdfs_fname, slice_num, num_low_freqs)


def collate_fn(batch):
    """Stack batch items into tensors (single-slice and multi-slice datasets)."""
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


def paired_collate_fn(batch):
    """Stack batch items into tensors (paired-contrast dataset)."""
    (masked_kspaces, masks, targets_pd, targets_pdfs,
     pd_maxes, pdfs_maxes, pd_fnames, pdfs_fnames, slice_nums, nlfs) = zip(*batch)
    return (
        torch.stack(masked_kspaces),
        torch.stack(masks),
        torch.stack(targets_pd),
        torch.stack(targets_pdfs),
        list(pd_maxes),
        list(pdfs_maxes),
        list(pd_fnames),
        list(pdfs_fnames),
        list(slice_nums),
        list(nlfs),
    )
