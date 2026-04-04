"""
Self-contained transforms for fastMRI knee VarNet.

Includes complex math ops, FFT utilities, mask functions, and normalization.
Adapted from the fastMRI repository (MIT License,
https://github.com/facebookresearch/fastMRI).
"""

import contextlib
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.fft


# ---------------------------------------------------------------------------
# Complex math helpers
# ---------------------------------------------------------------------------

def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise complex multiplication. Last dim must be 2 (re, im)."""
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """Complex conjugate. Last dim must be 2."""
    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """Magnitude of complex tensor. Last dim must be 2."""
    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """Squared magnitude of complex tensor. Last dim must be 2."""
    return (data ** 2).sum(dim=-1)


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Root-sum-of-squares along `dim` (coil combination for magnitude images)."""
    return torch.sqrt((data ** 2).sum(dim))


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Root-sum-of-squares for complex tensor (last dim = 2)."""
    return torch.sqrt(complex_abs_sq(data).sum(dim))


# ---------------------------------------------------------------------------
# FFT utilities
# ---------------------------------------------------------------------------

def _roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def _roll(x: torch.Tensor, shift: List[int], dim: List[int]) -> torch.Tensor:
    for s, d in zip(shift, dim):
        x = _roll_one_dim(x, s, d)
    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    if dim is None:
        dim = list(range(x.dim()))
    shift = [x.shape[d] // 2 for d in dim]
    return _roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    if dim is None:
        dim = list(range(x.dim()))
    shift = [(x.shape[d] + 1) // 2 for d in dim]
    return _roll(x, shift, dim)


def fft2c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Centered 2D FFT. Input: (..., H, W, 2). Output: (..., H, W, 2)."""
    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(torch.view_as_complex(data), dim=(-2, -1), norm=norm)
    )
    return fftshift(data, dim=[-3, -2])


def ifft2c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Centered 2D IFFT. Input: (..., H, W, 2). Output: (..., H, W, 2)."""
    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(torch.view_as_complex(data), dim=(-2, -1), norm=norm)
    )
    return fftshift(data, dim=[-3, -2])


# ---------------------------------------------------------------------------
# Spatial crop helpers
# ---------------------------------------------------------------------------

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Center crop last two dimensions of a real tensor to `shape`."""
    h, w = data.shape[-2], data.shape[-1]
    h_from = (h - shape[0]) // 2
    w_from = (w - shape[1]) // 2
    return data[..., h_from: h_from + shape[0], w_from: w_from + shape[1]]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Center crop for complex tensor (last dim = 2, spatial at -3, -2)."""
    h, w = data.shape[-3], data.shape[-2]
    h_from = (h - shape[0]) // 2
    w_from = (w - shape[1]) // 2
    return data[..., h_from: h_from + shape[0], w_from: w_from + shape[1], :]


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to torch tensor.

    Complex arrays become float tensors with shape (..., 2) where
    the last dimension holds (real, imag).
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


# ---------------------------------------------------------------------------
# Mask functions  (from fastMRI, MIT License)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    if seed is None:
        yield
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """Base class for k-space sub-sampling masks."""

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        seed: Optional[int] = None,
    ):
        if len(center_fractions) != len(accelerations):
            raise ValueError("center_fractions and accelerations must have equal length.")
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions.")
        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_freqs = self.sample_mask(shape, offset)
        return torch.max(center_mask, accel_mask), num_low_freqs

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        num_cols = shape[-2]
        center_fraction, acceleration = self._choose_acceleration()
        num_low_freqs = round(num_cols * center_fraction)
        center_mask = self._reshape_mask(self._center_mask(shape, num_low_freqs), shape)
        accel_mask = self._reshape_mask(
            self._accel_mask(num_cols, acceleration, offset, num_low_freqs), shape
        )
        return center_mask, accel_mask, num_low_freqs

    def _reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        num_cols = shape[-2]
        mask_shape = [1] * len(shape)
        mask_shape[-2] = num_cols
        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def _center_mask(self, shape: Sequence[int], num_low_freqs: int) -> np.ndarray:
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad: pad + num_low_freqs] = 1
        return mask

    def _accel_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_freqs: int,
    ) -> np.ndarray:
        raise NotImplementedError

    def _choose_acceleration(self):
        choice = self.rng.randint(len(self.center_fractions))
        return self.center_fractions[choice], self.accelerations[choice]


class RandomMaskFunc(MaskFunc):
    """
    Random k-space undersampling mask.

    Densely samples the center (ACS lines), then randomly selects exactly
    the right number of outer k-space lines to hit the target acceleration.
    """

    def _accel_mask(self, num_cols, acceleration, offset, num_low_freqs):
        num_outer = num_cols - num_low_freqs
        num_to_sample = round(num_outer / acceleration)
        outer_indices = self.rng.choice(num_outer, size=num_to_sample, replace=False)
        mask = np.zeros(num_cols)
        center_start = (num_cols - num_low_freqs + 1) // 2
        center_end = center_start + num_low_freqs
        outer_cols = np.concatenate([np.arange(0, center_start), np.arange(center_end, num_cols)])
        mask[outer_cols[outer_indices]] = 1.0
        return mask


class EquispacedMaskFractionFunc(MaskFunc):
    """
    Equispaced undersampling mask.

    8% dense center + every Rth line outside, where R = acceleration.
    """

    def _accel_mask(self, num_cols, acceleration, offset, num_low_freqs):
        if offset is None:
            offset = self.rng.randint(0, high=acceleration)
        mask = np.zeros(num_cols)
        mask[offset::acceleration] = 1.0
        return mask


# ---------------------------------------------------------------------------
# Apply mask to k-space
# ---------------------------------------------------------------------------

def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Apply a sub-sampling mask to k-space data.

    Args:
        data: k-space tensor of shape (coils, rows, cols, 2).
        mask_func: A MaskFunc instance.
        offset: Optional offset for equispaced masks.
        seed: Optional seed for reproducibility.

    Returns:
        (masked_kspace, mask, num_low_frequencies)
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_freqs = mask_func(shape, offset=offset, seed=seed)
    mask = mask.to(data.device)
    masked_data = data * mask + 0.0
    return masked_data, mask, num_low_freqs


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_kspace(kspace: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Normalize k-space by the 99th percentile of the zero-filled RSS image.

    Args:
        kspace: Tensor of shape (coils, H, W, 2).

    Returns:
        (normalized_kspace, scale_factor)
    """
    image = ifft2c(kspace)
    image_crop = complex_center_crop(image, (320, 320))
    rss_zf = rss(complex_abs(image_crop), dim=0)
    scale = float(torch.quantile(rss_zf, 0.99))
    if scale < 1e-8:
        scale = 1.0
    return kspace / scale, scale
