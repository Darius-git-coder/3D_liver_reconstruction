from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .losses import ssim3d


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    diff2 = (pred - target) ** 2
    if mask is None:
        return diff2.mean()
    return (diff2 * mask).sum() / (mask.sum() + eps)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    diff = torch.abs(pred - target)
    if mask is None:
        return diff.mean()
    return (diff * mask).sum() / (mask.sum() + eps)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(_masked_mse(pred, target, mask) + 1e-12)


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], data_range: float = 1.0) -> torch.Tensor:
    mse = _masked_mse(pred, target, mask)
    return 10.0 * torch.log10((data_range ** 2) / (mse + 1e-12))


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    known_mask: torch.Tensor,
    data_range: float = 1.0,
) -> Dict[str, float]:
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
