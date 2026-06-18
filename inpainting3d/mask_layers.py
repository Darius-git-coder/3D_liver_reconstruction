# inpainting3d/mask_layers.py
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv3d(nn.Module):
    """
    Apply a 3D partial convolution and propagate an updated validity mask.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
        eps: float = 1e-8,
    ):
        """
        Initialize the partial 3D convolution layer.
        
        Parameters
        ----------
        in_channels : int
            Number of input feature channels.
        out_channels : int
            Number of output feature channels.
        kernel_size : int
            Convolution kernel size. Defaults to 3.
        stride : int
            Convolution stride. Defaults to 1.
        padding : int
            Convolution padding. Defaults to 1.
        dilation : int
            Convolution dilation. Defaults to 1.
        bias : bool
            Whether the convolution uses a learnable bias. Defaults to True.
        eps : float
            Small constant that avoids division by zero. Defaults to 1e-8.
        """
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.eps = eps

        k = kernel_size
        weight = torch.ones(1, 1, k, k, k)
        self.register_buffer("weight_mask", weight)
        self.kernel_volume = float(k * k * k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the partial convolution to an input tensor and mask.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor or array.
        mask : Optional[torch.Tensor]
            Binary mask that marks valid or selected voxels.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output produced by the model or workflow.
        """
        if mask is None:
            mask = torch.ones((x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=x.dtype)

        if mask.shape[1] != 1:
            mask = mask[:, :1, ...]

        x_masked = x * mask.to(dtype=x.dtype)

        out = self.conv(x_masked)

        with torch.no_grad():
            mask_f = mask.float()
            # mask_sum: wie viele valid Voxels im lokalen Kernel-Fenster?
            mask_sum = F.conv3d(
                mask_f,
                self.weight_mask.to(dtype=torch.float32, device=mask.device),
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation
            )

        valid = mask_sum > 0.0
        mask_ratio = torch.where(
            valid,
            self.kernel_volume / mask_sum.clamp_min(self.eps),
            torch.zeros_like(mask_sum),
        )

        # Die Renormalisierung laeuft in float32, damit AMP nicht ueber
        # inf * 0 in NaNs kippt, wenn mask_sum in komplett invalid Fenstern 0 ist.
        out_f = out.float()
        valid_f = valid.expand(-1, out_f.shape[1], -1, -1, -1)
        mask_ratio_f = mask_ratio.expand(-1, out_f.shape[1], -1, -1, -1)

        if self.conv.bias is not None:
            bias = self.conv.bias.float().view(1, -1, 1, 1, 1)
            scaled = (out_f - bias) * mask_ratio_f + bias
        else:
            scaled = out_f * mask_ratio_f

        out = torch.where(valid_f, scaled, torch.zeros_like(scaled)).to(dtype=out.dtype)
        new_mask = valid.to(mask.dtype)
        return out, new_mask


class PartialResidualBlock3D(nn.Module):
    """
    Apply two partial convolution layers with a residual shortcut.
    """
    def __init__(self, in_f: int, out_f: int):
        """
        Initialize the partial residual block.
        
        Parameters
        ----------
        in_f : int
            Number of input feature channels.
        out_f : int
            Number of output feature channels.
        """
        super().__init__()
        self.c1 = PartialConv3d(in_f, out_f, kernel_size=3, padding=1)
        self.n1 = nn.InstanceNorm3d(out_f, affine=True)
        self.c2 = PartialConv3d(out_f, out_f, kernel_size=3, padding=1)
        self.n2 = nn.InstanceNorm3d(out_f, affine=True)

        self.shortcut = PartialConv3d(in_f, out_f, kernel_size=1, padding=0) if in_f != out_f else None
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the partial residual block on features and masks.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor or array.
        mask : torch.Tensor
            Binary mask that marks valid or selected voxels.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output produced by the model or workflow.
        """
        y, m = self.c1(x, mask)
        y = self.act(self.n1(y))

        y, m = self.c2(y, m)
        y = self.n2(y)

        if self.shortcut is not None:
            s, ms = self.shortcut(x, mask)
        else:
            s, ms = x, mask

        m_out = torch.clamp(m + ms, 0, 1)
        out = self.act(y + s)
        return out, m_out


