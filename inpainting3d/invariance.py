# inpainting3d/invariance.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence

import numpy as np
from scipy.ndimage import rotate as ndi_rotate
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import affine_transform


@dataclass
class PCATransform:
    R: np.ndarray       # 3x3 Rotation
    center: np.ndarray  # 3,


def pca_align_stable(
    vol: np.ndarray,
    mask: Optional[np.ndarray] = None,
    thr: float = 0.1,
    eig_gap_tol: float = 0.03,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[PCATransform]]:
    """
    PCA-basierte Ausrichtung mit Stabilitäts-Heuristik:
    - Falls Eigenwerte nahe gleich (degeneriert): skip align (zu instabil)
    - Determinant positiv (right-handed)
    - Identische Transformation auf Maske (order=0)

    Rückgabe: aligned_vol, aligned_mask, transform oder None
    """
    if mask is None:
        coords = np.argwhere(vol > thr)
    else:
        coords = np.argwhere(mask > 0.5)

    if coords.shape[0] < 20:
        return vol, mask, None

    center = coords.mean(axis=0).astype(np.float32)
    coords_c = coords.astype(np.float32) - center[None, :]

    cov = np.cov(coords_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # asc
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Degeneranzcheck: wenn Achsen nicht eindeutig -> skip
    gap = float((eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-8))
    if gap < eig_gap_tol:
        return vol, mask, None

    # Sign-Konsistenz (simple Heuristik)
    for i in range(3):
        if eigvecs[:, i].sum() < 0:
            eigvecs[:, i] *= -1.0

    R = eigvecs.astype(np.float32)
    # right-handed erzwingen
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0

    # affine_transform: output_index -> input_index = R^T @ out + offset
    # wir wollen um center rotieren
    offset = center - R.T @ center

    aligned = affine_transform(vol, R.T, offset=offset, order=1, mode="constant", cval=0.0)
    aligned_mask = None
    if mask is not None:
        aligned_mask = affine_transform(mask, R.T, offset=offset, order=0, mode="constant", cval=0.0)
        aligned_mask = (aligned_mask > 0.5).astype(np.float32)

    return aligned.astype(np.float32), aligned_mask, PCATransform(R=R, center=center)


def apply_rotation(vol: np.ndarray, mask: np.ndarray, angle_deg: float, axes=(1, 2)) -> Tuple[np.ndarray, np.ndarray]:
    v = ndi_rotate(vol, angle=angle_deg, axes=axes, reshape=False, order=1, mode="constant", cval=0.0)
    m = ndi_rotate(mask, angle=angle_deg, axes=axes, reshape=False, order=0, mode="constant", cval=0.0)
    m = (m > 0.5).astype(np.float32)
    return v.astype(np.float32), m


def apply_translation(vol: np.ndarray, mask: np.ndarray, shift_xyz: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    v = ndi_shift(vol, shift=shift_xyz, order=1, mode="constant", cval=0.0)
    m = ndi_shift(mask, shift=shift_xyz, order=0, mode="constant", cval=0.0)
    m = (m > 0.5).astype(np.float32)
    return v.astype(np.float32), m


def apply_intensity(vol: np.ndarray, mask: np.ndarray, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    v = np.clip(a * vol + b, 0.0, 1.0).astype(np.float32)
    return v, mask.astype(np.float32)
