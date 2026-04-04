"""
Simplified End-to-End VarNet for fastMRI knee reconstruction.

Adapted from the fastMRI repository (MIT License,
https://github.com/facebookresearch/fastMRI).

Architecture:
  SensitivityModel (NormUnet) -> N x VarNetBlock -> RSS output

Each VarNetBlock:
  1. Sensitivity-expand current estimate to multi-coil kspace
  2. Run NormUnet regularizer on sensitivity-reduced image
  3. Soft data-consistency:
       kspace <- kspace - dc_weight * (A*A^H*kspace - masked_kspace) - model_term
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .unet import NormUnet
from utils.transforms import (
    fft2c,
    ifft2c,
    complex_mul,
    complex_conj,
    complex_abs,
    rss,
    rss_complex,
    complex_center_crop,
)


def _batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """Zero out k-space outside the center band [mask_from, mask_to)."""
    if mask_from.shape[0] == 1:
        mask = torch.zeros_like(x)
        mask[:, :, :, int(mask_from): int(mask_to)] = x[:, :, :, int(mask_from): int(mask_to)]
        return mask
    mask = torch.zeros_like(x)
    for i, (start, end) in enumerate(zip(mask_from, mask_to)):
        mask[i, :, :, start:end] = x[i, :, :, start:end]
    return mask


class SensitivityModel(nn.Module):
    """
    Learns coil sensitivity maps from the ACS (auto-calibration signal) lines.

    The ACS region is isolated, transformed to image space, refined by a
    NormUnet, then normalised to unit RSS so the maps can be used to
    combine / expand multi-coil images.
    """

    def __init__(
        self,
        chans: int = 8,
        num_pools: int = 4,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.norm_unet = NormUnet(chans, num_pools, in_chans, out_chans, drop_prob)

    def _chans_to_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    def _batch_to_chans(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)

    def _divide_rss(self, x: torch.Tensor) -> torch.Tensor:
        return x / (rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1) + 1e-8)

    def _get_acs_pad(
        self,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            squeezed = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed.shape[1] // 2
            left = torch.argmin(squeezed[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed[:, cent:], dim=1)
            num_lf = torch.max(2 * torch.min(left, right), torch.ones_like(left))
        else:
            num_lf = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )
        pad = (mask.shape[-2] - num_lf + 1) // 2
        return pad, num_lf

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        pad, num_lf = self._get_acs_pad(mask, num_low_frequencies)
        acs_kspace = _batched_mask_center(masked_kspace, pad, pad + num_lf)
        images, b = self._chans_to_batch(ifft2c(acs_kspace))
        return self._divide_rss(self._batch_to_chans(self.norm_unet(images), b))


class VarNetBlock(nn.Module):
    """
    One unrolled gradient step.

    Applies:
      1. Regulariser  (NormUnet in image space)
      2. Soft data consistency  (k-space domain)
    """

    def __init__(self, model: nn.Module, use_dc: bool = True):
        super().__init__()
        self.model = model
        if use_dc:
            self.dc_weight = nn.Parameter(torch.ones(1))
        else:
            self.dc_weight = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fft2c(complex_mul(x, sens_maps))

    def _sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return complex_mul(ifft2c(x), complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1, device=current_kspace.device,
                           dtype=current_kspace.dtype)
        soft_dc = torch.where(mask.bool(), current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self._sens_expand(
            self.model(self._sens_reduce(current_kspace, sens_maps)), sens_maps
        )
        return current_kspace - soft_dc - model_term


class SimpleVarNet(nn.Module):
    """
    Simplified End-to-End VarNet.

    Args:
        num_cascades:   Number of unrolled gradient steps.
        chans:          U-Net channels in each cascade (regulariser).
        pools:          U-Net pool layers in each cascade.
        sens_chans:     U-Net channels in the sensitivity model.
        sens_pools:     U-Net pool layers in the sensitivity model.
        use_dc:         If False, sets all DC weights to 0 and freezes them.
        recon_size:     Spatial size of the output image (H, W).
    """

    def __init__(
        self,
        num_cascades: int = 8,
        chans: int = 18,
        pools: int = 4,
        sens_chans: int = 8,
        sens_pools: int = 4,
        dc_weight_init: float = 1.0,
        learn_dc: bool = True,
        use_dc: bool = True,
        recon_size: Tuple[int, int] = (320, 320),
    ):
        super().__init__()
        self.recon_size = recon_size

        self.sens_net = SensitivityModel(chans=sens_chans, num_pools=sens_pools)

        self.cascades = nn.ModuleList([
            VarNetBlock(NormUnet(chans, pools), use_dc=use_dc)
            for _ in range(num_cascades)
        ])

        for block in self.cascades:
            if use_dc:
                with torch.no_grad():
                    block.dc_weight.fill_(dc_weight_init)
                if not learn_dc:
                    block.dc_weight.requires_grad_(False)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        image = ifft2c(kspace_pred)
        image = complex_center_crop(image, self.recon_size)
        return rss(complex_abs(image), dim=1)
