# inpainting3d/losses.py
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the mean of a tensor over a masked region.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor or array.
    mask : torch.Tensor
        Binary mask that marks valid or selected voxels.
    eps : float
        Small constant that avoids division by zero. Defaults to 1e-8.
    
    Returns
    -------
    torch.Tensor
        Mean value aggregated over the selected mask.
    """
    return (x * mask).sum() / (mask.sum() + eps)


class MaskedInpaintingLossNormalized(nn.Module):
    """
    Compute a mask-aware L1 loss with separate known and missing terms.
    """

    def __init__(
        self,
        hole_weight: float = 10.0,
        valid_weight: float = 1.0,
        hole_fg_weight: Optional[float] = None,
        hole_bg_weight: Optional[float] = None,
        foreground_threshold: float = 0.1,
        eps: float = 1e-8,
    ):
        """
        Initialize the normalized masked inpainting loss.
        
        Parameters
        ----------
        hole_weight : float
            Weight assigned to missing-voxel reconstruction errors. Defaults to 10.0.
        valid_weight : float
            Weight assigned to known-voxel reconstruction errors. Defaults to 1.0.
        hole_fg_weight : Optional[float]
            Optional weight for missing foreground voxels. Defaults to None.
        hole_bg_weight : Optional[float]
            Optional weight for missing background voxels. Defaults to None.
        foreground_threshold : float
            Threshold used to identify foreground voxels. Defaults to 0.1.
        eps : float
            Small constant that avoids division by zero. Defaults to 1e-8.
        """
        super().__init__()
        self.hole_weight = float(hole_weight)
        self.valid_weight = float(valid_weight)
        self.hole_fg_weight = float(hole_weight if hole_fg_weight is None else hole_fg_weight)
        self.hole_bg_weight = float(hole_weight if hole_bg_weight is None else hole_bg_weight)
        self.foreground_threshold = float(foreground_threshold)
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred/target: [B,1,D,H,W], mask: [B,1,D,H,W]
        """
        Evaluate the normalized masked inpainting loss.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted reconstruction tensor or array.
        target : torch.Tensor
            Reference tensor or array used as supervision.
        mask : torch.Tensor
            Binary mask that marks valid or selected voxels.
        
        Returns
        -------
        torch.Tensor
            Output produced by the model or workflow.
        """
        err = torch.abs(pred - target)
        valid = mask
        hole = 1.0 - mask
        loss_valid = _masked_mean(err, valid, eps=self.eps)
        loss_hole = _masked_mean(err, hole, eps=self.eps)
        return self.valid_weight * loss_valid + self.hole_weight * loss_hole


class GradientLoss3D(nn.Module):
    """
    Compare local 3D gradients between prediction and target volumes.
    """

    def __init__(self, region: str = "missing", eps: float = 1e-8):
        """
        Initialize the 3D gradient loss.
        
        Parameters
        ----------
        region : str
            Region selector used by the loss. Defaults to "missing".
        eps : float
            Small constant that avoids division by zero. Defaults to 1e-8.
        """
        super().__init__()
        self.region = region
        self.eps = eps

    @staticmethod
    def _grad3d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute forward finite differences along all spatial axes.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor or array.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Forward finite-difference tensors along the three spatial axes.
        """
        dx = x[..., 1:, :, :] - x[..., :-1, :, :]
        dy = x[..., :, 1:, :] - x[..., :, :-1, :]
        dz = x[..., :, :, 1:] - x[..., :, :, :-1]
        return dx, dy, dz

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate the 3D gradient loss.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted reconstruction tensor or array.
        target : torch.Tensor
            Reference tensor or array used as supervision.
        mask : Optional[torch.Tensor]
            Binary mask that marks valid or selected voxels. Defaults to None.
        region_mask : Optional[torch.Tensor]
            Binary mask that selects the region used for aggregation. Defaults to None.
        
        Returns
        -------
        torch.Tensor
            Output produced by the model or workflow.
        """
        gpx, gpy, gpz = self._grad3d(pred)
        gtx, gty, gtz = self._grad3d(target)

        diffx = torch.abs(gpx - gtx)
        diffy = torch.abs(gpy - gty)
        diffz = torch.abs(gpz - gtz)

        if self.region == "all" or (mask is None and region_mask is None):
            return (diffx.mean() + diffy.mean() + diffz.mean()) / 3.0

        # Backward-compatible API:
        # - when region_mask is given, it explicitly defines the voxels whose
        #   gradients should be compared.
        # - otherwise we fall back to the original "missing = 1 - mask"
        #   interpretation used for known/missing inpainting masks.
        if region_mask is not None:
            hole = region_mask.float()
        else:
            hole = 1.0 - mask
        hx = hole[..., 1:, :, :] * hole[..., :-1, :, :]
        hy = hole[..., :, 1:, :] * hole[..., :, :-1, :]
        hz = hole[..., :, :, 1:] * hole[..., :, :, :-1]

        lx = _masked_mean(diffx, hx, eps=self.eps)
        ly = _masked_mean(diffy, hy, eps=self.eps)
        lz = _masked_mean(diffz, hz, eps=self.eps)
        return (lx + ly + lz) / 3.0


def _gaussian_kernel_1d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """
    Create a normalized one-dimensional Gaussian kernel.
    
    Parameters
    ----------
    kernel_size : int
        Convolution kernel size.
    sigma : float
        Standard deviation of the Gaussian kernel.
    device : Any
        Torch device on which tensors should be created or evaluated.
    dtype : Any
        Data type used when constructing arrays or tensors.
    
    Returns
    -------
    torch.Tensor
        One-dimensional normalized Gaussian kernel.
    """
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / (g.sum() + 1e-12)
    return g


def _gaussian_kernel_3d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """
    Create a normalized separable three-dimensional Gaussian kernel.
    
    Parameters
    ----------
    kernel_size : int
        Convolution kernel size.
    sigma : float
        Standard deviation of the Gaussian kernel.
    device : Any
        Torch device on which tensors should be created or evaluated.
    dtype : Any
        Data type used when constructing arrays or tensors.
    
    Returns
    -------
    torch.Tensor
        Three-dimensional normalized Gaussian kernel.
    """
    g1 = _gaussian_kernel_1d(kernel_size, sigma, device, dtype)
    g3 = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    return g3


def _autocast_disabled(x: torch.Tensor):
    """
    Return a context manager that temporarily disables autocasting.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor or array.
    
    Returns
    -------
    Any
        Context manager that keeps the enclosed computation out of autocast mode.
    """
    if x.is_cuda:
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def ssim3d(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute differentiable 3D SSIM values for volumetric tensors.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor or array.
    y : torch.Tensor
        Reference tensor or array.
    data_range : float
        Dynamic value range used by PSNR or SSIM computations. Defaults to 1.0.
    kernel_size : int
        Convolution kernel size. Defaults to 11.
    sigma : float
        Standard deviation of the Gaussian kernel. Defaults to 1.5.
    k1 : float
        First SSIM stabilization constant factor. Defaults to 0.01.
    k2 : float
        Second SSIM stabilization constant factor. Defaults to 0.03.
    mask : Optional[torch.Tensor]
        Binary mask that marks valid or selected voxels. Defaults to None.
    eps : float
        Small constant that avoids division by zero. Defaults to 1e-8.
    
    Returns
    -------
    torch.Tensor
        Scalar SSIM score averaged over the selected region.
    """

    with _autocast_disabled(x):
        x = x.float()
        y = y.float()
        if mask is not None:
            mask = mask.float()

        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2

        device, dtype = x.device, x.dtype
        win = _gaussian_kernel_3d(kernel_size, sigma, device, dtype).view(
            1, 1, kernel_size, kernel_size, kernel_size
        )
        pad = kernel_size // 2

        mu_x = F.conv3d(x, win, padding=pad, groups=1)
        mu_y = F.conv3d(y, win, padding=pad, groups=1)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv3d(x * x, win, padding=pad, groups=1) - mu_x2
        sigma_y2 = F.conv3d(y * y, win, padding=pad, groups=1) - mu_y2
        sigma_xy = F.conv3d(x * y, win, padding=pad, groups=1) - mu_xy

        num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        ssim_map = num / (den + eps)

        if mask is None:
            return ssim_map.mean()
        return _masked_mean(ssim_map, mask, eps=eps)


def ms_ssim3d(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    weights: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute differentiable multi-scale 3D SSIM values.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor or array.
    y : torch.Tensor
        Reference tensor or array.
    data_range : float
        Dynamic value range used by PSNR or SSIM computations. Defaults to 1.0.
    weights : Sequence[float]
        Checkpoint path or weighting coefficients used by the helper. Defaults to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).
    kernel_size : int
        Convolution kernel size. Defaults to 11.
    sigma : float
        Standard deviation of the Gaussian kernel. Defaults to 1.5.
    k1 : float
        First SSIM stabilization constant factor. Defaults to 0.01.
    k2 : float
        Second SSIM stabilization constant factor. Defaults to 0.03.
    mask : Optional[torch.Tensor]
        Binary mask that marks valid or selected voxels. Defaults to None.
    eps : float
        Small constant that avoids division by zero. Defaults to 1e-8.
    
    Returns
    -------
    torch.Tensor
        Scalar multi-scale SSIM score averaged over the selected region.
    """

    with _autocast_disabled(x):
        x = x.float()
        y = y.float()
        if mask is not None:
            mask = mask.float()

        device, dtype = x.device, x.dtype
        weights_t = torch.tensor(weights, device=device, dtype=dtype)

        mcs = []
        for _ in range(len(weights) - 1):
            c2 = (k2 * data_range) ** 2
            win = _gaussian_kernel_3d(kernel_size, sigma, device, dtype).view(
                1, 1, kernel_size, kernel_size, kernel_size
            )
            pad = kernel_size // 2

            mu_x = F.conv3d(x, win, padding=pad)
            mu_y = F.conv3d(y, win, padding=pad)

            sigma_x2 = F.conv3d(x * x, win, padding=pad) - mu_x * mu_x
            sigma_y2 = F.conv3d(y * y, win, padding=pad) - mu_y * mu_y
            sigma_xy = F.conv3d(x * y, win, padding=pad) - mu_x * mu_y

            cs_map = (2 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2 + eps)

            if mask is None:
                cs = cs_map.mean()
            else:
                cs = _masked_mean(cs_map, mask, eps=eps)

            # Vermeidet instabile Ableitungen bei x**w fuer x==0.
            mcs.append(torch.clamp(cs, min=1e-6))

            x = F.avg_pool3d(x, kernel_size=2, stride=2)
            y = F.avg_pool3d(y, kernel_size=2, stride=2)
            if mask is not None:
                mask = F.avg_pool3d(mask, kernel_size=2, stride=2)
                mask = (mask > 0).to(mask.dtype)

        ssim_last = ssim3d(
            x, y, data_range=data_range, kernel_size=kernel_size, sigma=sigma, k1=k1, k2=k2, mask=mask, eps=eps
        )
        ssim_last = torch.clamp(ssim_last, min=1e-6)

        mcs_t = torch.stack(mcs, dim=0)
        ms = torch.prod(mcs_t ** weights_t[:-1]) * (ssim_last ** weights_t[-1])
        return ms


@dataclass
class CombinedLossConfig:
    """
    Store weighting coefficients for the combined inpainting loss.
    
    Attributes
    ----------
    hole_weight : float
        Weight assigned to the missing-region reconstruction term.
    valid_weight : float
        Weight assigned to the observed-region reconstruction term.
    hole_fg_weight : Optional[float]
        Optional foreground-specific weight inside missing regions.
    hole_bg_weight : Optional[float]
        Optional background-specific weight inside missing regions.
    foreground_threshold : float
        Threshold used to separate foreground from background.
    w_grad : float
        Weight assigned to the gradient-consistency term.
    w_ssim : float
        Weight assigned to the single-scale SSIM term.
    w_ms_ssim : float
        Weight assigned to the multi-scale SSIM term.
    data_range : float
        Value range used by structural similarity losses.
    """
    hole_weight: float = 10.0
    valid_weight: float = 1.0
    hole_fg_weight: Optional[float] = None
    hole_bg_weight: Optional[float] = None
    foreground_threshold: float = 0.1
    w_grad: float = 0.05
    w_ssim: float = 0.0
    w_ms_ssim: float = 0.2
    data_range: float = 1.0


class CombinedInpaintingLoss(nn.Module):
    """
    Combine reconstruction, gradient, and structural similarity losses.
    """

    def __init__(self, cfg: CombinedLossConfig):
        """
        Initialize the combined inpainting loss.
        
        Parameters
        ----------
        cfg : CombinedLossConfig
            Configuration object that stores the loss weights.
        """
        super().__init__()
        self.cfg = cfg
        self.base = MaskedInpaintingLossNormalized(
            hole_weight=cfg.hole_weight,
            valid_weight=cfg.valid_weight,
            hole_fg_weight=cfg.hole_fg_weight,
            hole_bg_weight=cfg.hole_bg_weight,
            foreground_threshold=cfg.foreground_threshold,
        )
        self.grad = GradientLoss3D(region="missing")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the combined inpainting loss.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted reconstruction tensor or array.
        target : torch.Tensor
            Reference tensor or array used as supervision.
        mask : torch.Tensor
            Binary mask that marks valid or selected voxels.
        
        Returns
        -------
        torch.Tensor
            Output produced by the model or workflow.
        """
        l_base = self.base(pred, target, mask)
        foreground = (target > self.cfg.foreground_threshold).to(dtype=target.dtype)
        fg_missing = (1.0 - mask) * foreground

        l_grad = (
            self.grad(pred, target, region_mask=fg_missing)
            if self.cfg.w_grad > 0
            else pred.new_tensor(0.0)
        )

        l_ssim = pred.new_tensor(0.0)
        if self.cfg.w_ssim > 0 and bool(torch.count_nonzero(fg_missing).item()):
            l_ssim = 1.0 - ssim3d(pred, target, data_range=self.cfg.data_range, mask=fg_missing)

        l_ms = pred.new_tensor(0.0)
        if self.cfg.w_ms_ssim > 0 and bool(torch.count_nonzero(fg_missing).item()):
            l_ms = 1.0 - ms_ssim3d(pred, target, data_range=self.cfg.data_range, mask=fg_missing)

        return l_base + self.cfg.w_grad * l_grad + self.cfg.w_ssim * l_ssim + self.cfg.w_ms_ssim * l_ms
