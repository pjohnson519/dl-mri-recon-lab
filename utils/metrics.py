"""
SSIM loss and metric for fastMRI evaluation.

Reference:
  Wang et al., "Image Quality Assessment: From Error Visibility to
  Structural Similarity," IEEE TIP 2004.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1-D Gaussian kernel, normalised to sum to 1."""
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _ssim_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    SSIM between two batched images.

    Args:
        pred:       (B, 1, H, W)
        target:     (B, 1, H, W)
        data_range: (B,) or scalar

    Returns:
        Mean SSIM over the batch, shape ().
    """
    k1d = _gaussian_kernel(kernel_size, sigma).to(pred.device)
    kernel_2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

    pad = kernel_size // 2

    mu_x = F.conv2d(pred, kernel_2d, padding=pad)
    mu_y = F.conv2d(target, kernel_2d, padding=pad)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy   = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, kernel_2d, padding=pad) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, kernel_2d, padding=pad) - mu_y_sq
    sigma_xy   = F.conv2d(pred * target, kernel_2d, padding=pad) - mu_xy

    if isinstance(data_range, (int, float)):
        dr = torch.tensor(data_range, dtype=pred.dtype, device=pred.device)
    else:
        dr = data_range.to(pred.device).to(pred.dtype)
    if dr.dim() == 0:
        dr = dr.view(1, 1, 1, 1)
    else:
        dr = dr.view(-1, 1, 1, 1)

    c1 = (k1 * dr) ** 2
    c2 = (k2 * dr) ** 2

    numerator   = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / denominator
    return ssim_map.mean()


class SSIMLoss(nn.Module):
    """1 - SSIM  (so minimising this maximises structural similarity)."""

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        max_value: torch.Tensor = None,
    ) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        dr = max_value if max_value is not None else self.data_range
        return 1.0 - _ssim_2d(pred, target, data_range=dr)


def ssim_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_value: float = 1.0,
) -> float:
    """
    Compute SSIM as a scalar metric (no gradient).

    Args:
        pred, target: (H, W) or (B, H, W) tensors.
        max_value:    Data range.

    Returns:
        SSIM value in [0, 1].
    """
    with torch.no_grad():
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        return float(_ssim_2d(pred, target, data_range=max_value))
