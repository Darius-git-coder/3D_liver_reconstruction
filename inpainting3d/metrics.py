from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .losses import ssim3d


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the mean squared error over an optional mask.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted reconstruction tensor or array.
    target : torch.Tensor
        Reference tensor or array used as supervision.
    mask : Optional[torch.Tensor]
        Binary mask that marks valid or selected voxels.
    eps : float
        Small constant that avoids division by zero. Defaults to 1e-8.
    
    Returns
    -------
    torch.Tensor
        Mean squared error aggregated over the selected mask.
    """
    diff2 = (pred - target) ** 2
    if mask is None:
        return diff2.mean()
    return (diff2 * mask).sum() / (mask.sum() + eps)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the mean absolute error over an optional mask.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted reconstruction tensor or array.
    target : torch.Tensor
        Reference tensor or array used as supervision.
    mask : Optional[torch.Tensor]
        Binary mask that marks valid or selected voxels.
    eps : float
        Small constant that avoids division by zero. Defaults to 1e-8.
    
    Returns
    -------
    torch.Tensor
        Mean absolute error aggregated over the selected mask.
    """
    diff = torch.abs(pred - target)
    if mask is None:
        return diff.mean()
    return (diff * mask).sum() / (mask.sum() + eps)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Compute the root mean squared error over an optional mask.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted reconstruction tensor or array.
    target : torch.Tensor
        Reference tensor or array used as supervision.
    mask : Optional[torch.Tensor]
        Binary mask that marks valid or selected voxels.
    
    Returns
    -------
    torch.Tensor
        Root mean squared error aggregated over the selected mask.
    """
    return torch.sqrt(_masked_mse(pred, target, mask) + 1e-12)


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], data_range: float = 1.0) -> torch.Tensor:
    """
    Compute PSNR over an optional mask.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted reconstruction tensor or array.
    target : torch.Tensor
        Reference tensor or array used as supervision.
    mask : Optional[torch.Tensor]
        Binary mask that marks valid or selected voxels.
    data_range : float
        Dynamic value range used by PSNR or SSIM computations. Defaults to 1.0.
    
    Returns
    -------
    torch.Tensor
        PSNR value aggregated over the selected mask.
    """
    mse = _masked_mse(pred, target, mask)
    return 10.0 * torch.log10((data_range ** 2) / (mse + 1e-12))


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    known_mask: torch.Tensor,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """
    Compute reconstruction metrics for full, known, and missing regions.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted reconstruction tensor or array.
    target : torch.Tensor
        Reference tensor or array used as supervision.
    known_mask : torch.Tensor
        Binary mask that marks voxels observed in the sparse input.
    data_range : float
        Dynamic value range used by PSNR or SSIM computations. Defaults to 1.0.
    
    Returns
    -------
    Dict[str, float]
        Computed summary values.
    """
    pred = pred.float()
    target = target.float()
    known_mask = known_mask.float()
    missing = 1.0 - known_mask

    out: Dict[str, float] = {}
    for region_name, region_mask in [
        ("all", None),
        ("known", known_mask),
        ("missing", missing),
    ]:
        out[f"mae_{region_name}"] = float(masked_mae(pred, target, region_mask).item())
        out[f"rmse_{region_name}"] = float(masked_rmse(pred, target, region_mask).item())
        out[f"psnr_{region_name}"] = float(masked_psnr(pred, target, region_mask, data_range=data_range).item())
        out[f"ssim_{region_name}"] = float(ssim3d(pred, target, data_range=data_range, mask=region_mask).item())
    return out


def compute_metrics_per_case(
    pred: torch.Tensor,
    target: torch.Tensor,
    known_mask: torch.Tensor,
    data_range: float = 1.0,
    case_ids: Optional[Sequence[str]] = None,
    paths: Optional[Sequence[str]] = None,
) -> List[Dict[str, float | str]]:
    """
    Compute per-case reconstruction metrics for a batched prediction.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted reconstruction tensor or array.
    target : torch.Tensor
        Reference tensor or array used as supervision.
    known_mask : torch.Tensor
        Binary mask that marks voxels observed in the sparse input.
    data_range : float
        Dynamic value range used by PSNR or SSIM computations. Defaults to 1.0.
    case_ids : Optional[Sequence[str]]
        Case identifiers aligned with the batch dimension. Defaults to None.
    paths : Optional[Sequence[str]]
        Sequence of filesystem paths to process. Defaults to None.
    
    Returns
    -------
    List[Dict[str, float | str]]
        Computed summary values.
    """
    if pred.shape[0] != target.shape[0] or pred.shape[0] != known_mask.shape[0]:
        raise ValueError("pred, target and known_mask must have the same batch size.")

    rows: List[Dict[str, float | str]] = []
    for batch_index in range(pred.shape[0]):
        row: Dict[str, float | str] = compute_metrics(
            pred[batch_index:batch_index + 1],
            target[batch_index:batch_index + 1],
            known_mask[batch_index:batch_index + 1],
            data_range=data_range,
        )
        row["index"] = batch_index
        if case_ids is not None:
            row["case_id"] = str(case_ids[batch_index])
        if paths is not None:
            row["path"] = str(paths[batch_index])
        rows.append(row)
    return rows
