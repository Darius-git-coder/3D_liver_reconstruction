# inpainting3d/data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import zoom


def load_nifti_volume(
    path: str,
    canonical: bool = True,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lädt ein NIfTI-Volumen als numpy array. Optional: as_closest_canonical.
    Rückgabe: (volume, affine)
    """
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "load_nifti_volume requires the optional dependency 'nibabel'. "
            "Install it to use NIfTI-based 3D training or evaluation."
        ) from exc

    img = nib.load(path)
    if canonical:
        img = nib.as_closest_canonical(img)
    vol = img.get_fdata(dtype=dtype)  # Achtung: kann cachen; ok für Training/Eval
    return vol, img.affine


def normalize_volume(vol: np.ndarray, method: str = "clip01") -> np.ndarray:
    """
    Vereinheitlicht Intensitäten. Du kannst das an deine Preprocessing-Pipeline anpassen.
    - clip01: clip auf [0,1]
    - minmax: skaliert auf [0,1]
    - percentile: clip auf [p1,p99] und minmax
    """
    vol = vol.astype(np.float32)

    if method == "clip01":
        return np.clip(vol, 0.0, 1.0)

    if method == "minmax":
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax - vmin < 1e-8:
            return np.zeros_like(vol)
        return (vol - vmin) / (vmax - vmin)

    if method.startswith("percentile"):
        # z.B. "percentile:1,99"
        parts = method.split(":")
        p1, p2 = (1.0, 99.0) if len(parts) == 1 else tuple(map(float, parts[1].split(",")))
        lo, hi = np.percentile(vol, [p1, p2])
        vol = np.clip(vol, lo, hi)
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax - vmin < 1e-8:
            return np.zeros_like(vol)
        return (vol - vmin) / (vmax - vmin)

    raise ValueError(f"Unbekannte Normalisierung: {method}")


def resample_to_shape(vol: np.ndarray, target_shape: Tuple[int, int, int], order: int = 1) -> np.ndarray:
    """
    Resampling auf Zielshape. order=1 für Volumen (trilinear), order=0 für Masken.
    """
    if vol.shape == target_shape:
        return vol
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol, factors, order=order)


def center_of_mass_threshold(vol: np.ndarray, thr: float = 0.1) -> np.ndarray:
    coords = np.argwhere(vol > thr)
    if coords.shape[0] < 10:
        return np.array([(s - 1) / 2 for s in vol.shape], dtype=np.float32)
    return coords.mean(axis=0).astype(np.float32)


def fibonacci_normals(n: int) -> np.ndarray:
    """
    Approximativ gleichmäßige Richtungsverteilung auf der Kugel.
    """
    idx = np.arange(n, dtype=np.float32) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    nx = np.sin(phi) * np.cos(theta)
    ny = np.sin(phi) * np.sin(theta)
    nz = np.cos(phi)
    normals = np.stack([nx, ny, nz], axis=1)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    return normals.astype(np.float32)


def random_normals(n: int, rng: np.random.Generator) -> np.ndarray:
    normals = rng.normal(size=(n, 3)).astype(np.float32)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    return normals


def simulate_sparse_acquisition(
    vol: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    thickness_vox: float = 1.0,
    chunk: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Kern-Fix: Einheitliche Sparse-Acquisition (Training/Inference/Eval).
    plane: |(x-center)·n| <= thickness_vox

    Rückgabe:
      sparse_vol: recon/hits
      mask: (hits>0) float32
      hits: float32
    """
    vol = vol.astype(np.float32)
    D, H, W = vol.shape

    coords = np.indices((D, H, W)).astype(np.float32)
    dx = coords[0] - center[0]
    dy = coords[1] - center[1]
    dz = coords[2] - center[2]

    recon = np.zeros_like(vol, dtype=np.float32)
    hits = np.zeros_like(vol, dtype=np.float32)

    for start in range(0, normals.shape[0], chunk):
        n = normals[start:start + chunk]  # (c,3)
        dist = (
            dx[None, ...] * n[:, 0, None, None, None] +
            dy[None, ...] * n[:, 1, None, None, None] +
            dz[None, ...] * n[:, 2, None, None, None]
        )
        m = np.abs(dist) <= thickness_vox
        hits += m.sum(axis=0).astype(np.float32)
        recon += (m * vol[None, ...]).sum(axis=0).astype(np.float32)

    sparse = np.divide(recon, hits, out=np.zeros_like(recon), where=hits > 0)
    mask = (hits > 0).astype(np.float32)
    return sparse, mask, hits


@dataclass
class InpaintingSample:
    x: np.ndarray        # [2,D,H,W]  (sparse + mask)
    y: np.ndarray        # [1,D,H,W]  (gt)
    meta: dict
