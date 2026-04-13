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

from typing import List, Optional, Tuple, Union

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
        s, e = int(start), int(end)
        mask[i, :, :, s:e] = x[i, :, :, s:e]
    return mask


class SensitivityModel(nn.Module):
    """
    Learns coil sensitivity maps from the ACS (auto-calibration signal) lines.

    The ACS region is isolated, transformed to image space, refined by a
    NormUnet, then normalised to unit RSS so the maps can be used to
    combine / expand multi-coil images.

    For multi-slice / multi-contrast inputs, sensitivity maps are estimated
    and normalised independently for each group of ``num_coils`` physical
    coils.  This avoids mixing spatial content from different slices or
    contrasts during RSS normalisation.
    """

    def __init__(
        self,
        chans: int = 8,
        num_pools: int = 4,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        num_coils: int = 15,
        num_groups: int = 1,
    ):
        super().__init__()
        self.norm_unet = NormUnet(chans, num_pools, in_chans, out_chans, drop_prob)
        self.num_coils = num_coils
        self.num_groups = num_groups

    def _chans_to_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    def _batch_to_chans(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)

    def _divide_rss(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise to unit RSS within each coil group independently."""
        if self.num_groups == 1:
            return x / (rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1) + 1e-8)
        b, c, h, w, comp = x.shape
        nc = self.num_coils
        # (B, G, nc, H, W, 2)
        x = x.view(b, self.num_groups, nc, h, w, comp)
        # RSS per group over coil dim → (B, G, 1, H, W, 1)
        rss_val = rss_complex(x, dim=2).unsqueeze(-1).unsqueeze(2)
        x = x / (rss_val + 1e-8)
        return x.view(b, c, h, w, comp)

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

    For multi-group inputs (multi-slice / multi-contrast), sens_reduce
    and sens_expand operate independently on each group of ``num_coils``
    physical coils.  The U-Net regulariser sees one image per group,
    enabling cross-slice / cross-contrast feature learning.
    """

    def __init__(self, model: nn.Module, use_dc: bool = True,
                 num_coils: int = 15, num_groups: int = 1):
        super().__init__()
        self.model = model
        self.num_coils = num_coils
        self.num_groups = num_groups
        if use_dc:
            self.dc_weight = nn.Parameter(torch.ones(1))
        else:
            self.dc_weight = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        if self.num_groups == 1:
            return fft2c(complex_mul(x, sens_maps))
        b = x.shape[0]
        nc, g = self.num_coils, self.num_groups
        # x: (B, G, H, W, 2) → (B*G, 1, H, W, 2)
        x = x.reshape(b * g, 1, *x.shape[2:])
        s = sens_maps.reshape(b * g, nc, *sens_maps.shape[2:])
        out = fft2c(complex_mul(x, s))  # (B*G, nc, H, W, 2)
        return out.reshape(b, g * nc, *out.shape[2:])

    def _sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        if self.num_groups == 1:
            return complex_mul(ifft2c(x), complex_conj(sens_maps)).sum(dim=1, keepdim=True)
        b = x.shape[0]
        nc, g = self.num_coils, self.num_groups
        # x: (B, G*nc, H, W, 2) → (B*G, nc, H, W, 2)
        x = x.reshape(b * g, nc, *x.shape[2:])
        s = sens_maps.reshape(b * g, nc, *sens_maps.shape[2:])
        # Per-group: ifft, multiply by conj(sens), sum over coils → (B*G, 1, H, W, 2)
        reduced = complex_mul(ifft2c(x), complex_conj(s)).sum(dim=1, keepdim=True)
        # → (B, G, H, W, 2)
        return reduced.reshape(b, g, *reduced.shape[2:])

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
        num_cascades:    Number of unrolled gradient steps.
        chans:           U-Net channels in each cascade (regulariser).
        pools:           U-Net pool layers in each cascade.
        sens_chans:      U-Net channels in the sensitivity model.
        sens_pools:      U-Net pool layers in the sensitivity model.
        use_dc:          If False, sets all DC weights to 0 and freezes them.
        recon_size:      Spatial size of the output image (H, W).
        num_input_slices: Number of adjacent slices per contrast (1 = single-slice).
        num_coils:       Physical coils per slice (default 15).
        num_contrasts:   Number of contrasts (1 = single, 2 = joint PD+PDFS).
                         When >1, forward() returns a list of images, one per contrast.
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
        num_input_slices: int = 1,
        num_coils: int = 15,
        num_contrasts: int = 1,
    ):
        super().__init__()
        self.recon_size = recon_size
        self.num_input_slices = num_input_slices
        self.num_coils = num_coils
        self.num_contrasts = num_contrasts

        num_groups = num_input_slices * num_contrasts

        self.sens_net = SensitivityModel(
            chans=sens_chans, num_pools=sens_pools,
            num_coils=num_coils, num_groups=num_groups,
        )

        # Cascade U-Nets see one image per group → in_chans = 2 * num_groups
        cascade_chans = 2 * num_groups
        self.cascades = nn.ModuleList([
            VarNetBlock(
                NormUnet(chans, pools, in_chans=cascade_chans, out_chans=cascade_chans),
                use_dc=use_dc, num_coils=num_coils, num_groups=num_groups,
            )
            for _ in range(num_cascades)
        ])

        for block in self.cascades:
            if use_dc:
                with torch.no_grad():
                    block.dc_weight.fill_(dc_weight_init)
                if not learn_dc:
                    block.dc_weight.requires_grad_(False)

    def _extract_center_coils(self, kspace: torch.Tensor, contrast_idx: int = 0) -> torch.Tensor:
        """Extract the center-slice coils for a given contrast from the full k-space."""
        if self.num_input_slices == 1 and self.num_contrasts == 1:
            return kspace
        # Layout: [contrast_0_slice_0, ..., contrast_0_slice_N, contrast_1_slice_0, ...]
        slices_per_contrast = self.num_input_slices * self.num_coils
        center_slice_offset = (self.num_input_slices // 2) * self.num_coils
        start = contrast_idx * slices_per_contrast + center_slice_offset
        end = start + self.num_coils
        return kspace[:, start:end]

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        if self.num_contrasts > 1:
            outputs = []
            for c in range(self.num_contrasts):
                kc = self._extract_center_coils(kspace_pred, contrast_idx=c)
                img = ifft2c(kc)
                img = complex_center_crop(img, self.recon_size)
                outputs.append(rss(complex_abs(img), dim=1))
            return outputs

        kspace_out = self._extract_center_coils(kspace_pred, contrast_idx=0)
        image = ifft2c(kspace_out)
        image = complex_center_crop(image, self.recon_size)
        return rss(complex_abs(image), dim=1)
