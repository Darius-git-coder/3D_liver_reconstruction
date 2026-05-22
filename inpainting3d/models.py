# inpainting3d/models.py
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_layers import PartialResidualBlock3D, GatedResidualBlock3D


class ResidualBlock3D(nn.Module):
    """vanilla conv mit Res-Connection """
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_f, out_f, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_f, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_f, out_f, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_f, affine=True),
        )
        self.shortcut = nn.Conv3d(in_f, out_f, kernel_size=1) if in_f != out_f else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class DeepResUNet3D(nn.Module):
    """Baseline: Input [B,2,D,H,W] -> Output [B,1,D,H,W]."""
    def __init__(self, in_channels: int = 2, out_channels: int = 1, init_feat: int = 32):
        super().__init__()
        f = init_feat
        self.enc1 = ResidualBlock3D(in_channels, f)
        self.enc2 = ResidualBlock3D(f, f * 2)
        self.enc3 = ResidualBlock3D(f * 2, f * 4)
        self.enc4 = ResidualBlock3D(f * 4, f * 8)
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = ResidualBlock3D(f * 8, f * 16)

        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock3D(f * 16, f * 8)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock3D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock3D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock3D(f * 2, f)

        self.final = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class GatedResUNet3D(nn.Module):
    """Drop-in Alternative: ersetzt Conv-Blöcke durch GatedResidualBlock3D (Maske bleibt als Input-Kanal erhalten)."""
    def __init__(self, in_channels: int = 2, out_channels: int = 1, init_feat: int = 32):
        super().__init__()
        f = init_feat
        self.enc1 = GatedResidualBlock3D(in_channels, f)
        self.enc2 = GatedResidualBlock3D(f, f * 2)
        self.enc3 = GatedResidualBlock3D(f * 2, f * 4)
        self.enc4 = GatedResidualBlock3D(f * 4, f * 8)
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = GatedResidualBlock3D(f * 8, f * 16)

        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = GatedResidualBlock3D(f * 16, f * 8)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = GatedResidualBlock3D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = GatedResidualBlock3D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = GatedResidualBlock3D(f * 2, f)

        self.final = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class PartialResUNet3D(nn.Module):
    """
    3D U-Net with residual partial convolution blocks for sparse volumetric data.

    This model extends the standard 3D U-Net architecture by incorporating
    Partial Convolutions and residual connections in all encoder and decoder
    blocks. It is specifically designed to handle inputs with missing or
    invalid regions, which are explicitly indicated via a binary mask.

    The network processes both the input volume and its corresponding mask
    throughout the entire forward pass. The mask is updated at each layer and
    ensures that convolutions only consider valid voxels.

    Architecture Overview:
        - Encoder: 4 levels of downsampling using PartialResidualBlock3D + max pooling
        - Bottleneck: deepest feature representation
        - Decoder: 4 levels of upsampling using transposed convolutions
        - Skip connections: concatenate encoder features with decoder features
        - Mask propagation: masks are pooled, upsampled, and merged at each stage

    Input Format:
        x2 (torch.Tensor):
            Shape: [B, 2, D, H, W]
            B = batch size 
            Channel 0: input volume (e.g., sparse or corrupted data)
            Channel 1: binary mask (1 = valid voxel, 0 = invalid/missing voxel)
            D = depth of the 3D volume
            H = height of the 3D volume
            W = width of the 3D volume
    Output:
        torch.Tensor:
            Shape: [B, out_channels, D, H, W]
            The predicted dense volume based on the input and mask.

    Mask Handling:
        - Encoder: masks are downsampled using max pooling
          (a voxel is valid if any voxel in the pooling region is valid)
        - Decoder: masks are upsampled via nearest-neighbor interpolation
        - Skip connections: masks are merged using logical OR
          (implemented as clamp(m_up + m_skip, 0, 1))
        - PartialResidualBlock3D uses the mask to ignore invalid inputs
          during convolution

    Args:
        out_channels (int, optional):
            Number of output channels. Default: 1.

        init_feat (int, optional):
            Number of feature channels in the first encoder layer.
            This value is doubled at each downsampling step.
            Default: 32.

    Forward Args:
        x2 (torch.Tensor):
            Input tensor containing both image and mask.
            Shape: [B, 2, D, H, W].

    Returns:
        torch.Tensor:
            Output tensor with shape [B, out_channels, D, H, W].

    Notes:
        - Spatial dimensions (D, H, W) should ideally be divisible by 16
          due to four downsampling operations.
        - The final layer is a standard 3D convolution and does not use
          partial convolution.
        - This architecture is well-suited for:
            * Sparse 3D reconstruction
            * Medical imaging with missing data
            * Volumetric inpainting tasks

    Example:
        >>> model = PartialResUNet3D(out_channels=1, init_feat=32)
        >>> x = torch.randn(2, 2, 64, 64, 64)  # [B, image+mask, D, H, W]
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([2, 1, 64, 64, 64])
    """
        
    def __init__(self, out_channels: int = 1, init_feat: int = 32):
        super().__init__()
        f = init_feat

        # Wir splitten im forward: image=[B,1,...], mask=[B,1,...]
        self.enc1 = PartialResidualBlock3D(1, f)
        self.enc2 = PartialResidualBlock3D(f, f * 2)
        self.enc3 = PartialResidualBlock3D(f * 2, f * 4)
        self.enc4 = PartialResidualBlock3D(f * 4, f * 8)
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = PartialResidualBlock3D(f * 8, f * 16)

        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = PartialResidualBlock3D(f * 16, f * 8)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = PartialResidualBlock3D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = PartialResidualBlock3D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = PartialResidualBlock3D(f * 2, f)

        self.final = nn.Conv3d(f, out_channels, kernel_size=1)

    @staticmethod
    def _pool_mask(mask: torch.Tensor) -> torch.Tensor:
        # Max-Pooling: wenn irgendein valid in 2x2x2 -> valid
        return F.max_pool3d(mask, kernel_size=2, stride=2)

    @staticmethod
    def _up_mask(mask: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
        return F.interpolate(mask, size=size, mode="nearest")

    def forward(self, x2: torch.Tensor) -> torch.Tensor:
        img = x2[:, 0:1, ...]
        mask = x2[:, 1:2, ...]

        e1, m1 = self.enc1(img, mask)
        e2_in = self.pool(e1)
        m2_in = self._pool_mask(m1)
        e2, m2 = self.enc2(e2_in, m2_in)

        e3_in = self.pool(e2)
        m3_in = self._pool_mask(m2)
        e3, m3 = self.enc3(e3_in, m3_in)

        e4_in = self.pool(e3)
        m4_in = self._pool_mask(m3)
        e4, m4 = self.enc4(e4_in, m4_in)

        b_in = self.pool(e4)
        mb_in = self._pool_mask(m4)
        b, mb = self.bottleneck(b_in, mb_in)

        up4 = self.up4(b)
        m_up4 = self._up_mask(mb, up4.shape[2:])
        m_cat4 = torch.clamp(m_up4 + m4, 0, 1)
        d4, md4 = self.dec4(torch.cat([up4, e4], dim=1), m_cat4)

        up3 = self.up3(d4)
        m_up3 = self._up_mask(md4, up3.shape[2:])
        m_cat3 = torch.clamp(m_up3 + m3, 0, 1)
        d3, md3 = self.dec3(torch.cat([up3, e3], dim=1), m_cat3)

        up2 = self.up2(d3)
        m_up2 = self._up_mask(md3, up2.shape[2:])
        m_cat2 = torch.clamp(m_up2 + m2, 0, 1)
        d2, md2 = self.dec2(torch.cat([up2, e2], dim=1), m_cat2)

        up1 = self.up1(d2)
        m_up1 = self._up_mask(md2, up1.shape[2:])
        m_cat1 = torch.clamp(m_up1 + m1, 0, 1)
        d1, _ = self.dec1(torch.cat([up1, e1], dim=1), m_cat1)

        return self.final(d1)
