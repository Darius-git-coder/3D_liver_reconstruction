# inpainting3d/data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import zoom


def load_nifti_volume(
    path: str,
    canonical: bool = True,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI volume as numpy array.
    Returns: (volume, affine)
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
    vol = img.get_fdata(dtype=dtype)
    return vol, img.affine


def normalize_volume(vol: np.ndarray, method: str = "clip01") -> np.ndarray:
    """
    Normalize intensities into a stable [0, 1]-like range.
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
        parts = method.split(":")
        p1, p2 = (1.0, 99.0) if len(parts) == 1 else tuple(map(float, parts[1].split(",")))
        lo, hi = np.percentile(vol, [p1, p2])
        vol = np.clip(vol, lo, hi)
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax - vmin < 1e-8:
            return np.zeros_like(vol)
        return (vol - vmin) / (vmax - vmin)

    raise ValueError(f"Unknown normalization: {method}")


def resample_to_shape(vol: np.ndarray, target_shape: Tuple[int, int, int], order: int = 1) -> np.ndarray:
    """
    Resample to target shape. order=1 for volumes, order=0 for masks.
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


def foreground_bbox_threshold(vol: np.ndarray, thr: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(vol > thr)
    if coords.shape[0] < 10:
        bbox_min = np.zeros((3,), dtype=np.float32)
        bbox_max = (np.asarray(vol.shape, dtype=np.float32) - 1.0).astype(np.float32)
        return bbox_min, bbox_max
    return coords.min(axis=0).astype(np.float32), coords.max(axis=0).astype(np.float32)


def fibonacci_normals(n: int) -> np.ndarray:
    """
    Approximately uniform directions on the sphere.
    """
    idx = np.arange(n, dtype=np.float32) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    nx = np.sin(phi) * np.cos(theta)
    ny = np.sin(phi) * np.sin(theta)
    nz = np.cos(phi)
    normals = np.stack([nx, ny, nz], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals.astype(np.float32)


def random_normals(n: int, rng: np.random.Generator) -> np.ndarray:
    normals = rng.normal(size=(n, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals


def _normalize_vector(vec: Sequence[float], *, fallback: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Expected a 3D vector, got shape {tuple(arr.shape)}.")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        arr = np.asarray(fallback, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(arr))
    return (arr / max(norm, 1e-8)).astype(np.float32)


def _orthonormal_basis(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    direction = _normalize_vector(direction)
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(direction, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    axis_u = np.cross(direction, reference)
    axis_u = _normalize_vector(axis_u, fallback=(0.0, 1.0, 0.0))
    axis_v = np.cross(direction, axis_u)
    axis_v = _normalize_vector(axis_v, fallback=(0.0, 0.0, 1.0))
    return axis_u.astype(np.float32), axis_v.astype(np.float32)


def sample_ultrasound_fan_normals(
    n: int,
    rng: np.random.Generator,
    *,
    preferred_axis: Sequence[float] | None = None,
    axis_jitter_deg: float = 25.0,
    fan_half_angle_deg: float = 35.0,
    elevation_jitter_deg: float = 6.0,
    sweep_jitter_deg: float = 2.5,
) -> np.ndarray:
    """
    Limited fan of directions around one dominant axis.
    """
    if n <= 0:
        raise ValueError(f"n must be >= 1, got {n}.")

    anchor_source = (0.0, 0.0, 1.0) if preferred_axis is None else preferred_axis
    anchor = _normalize_vector(anchor_source)
    base_direction = anchor.copy()
    if axis_jitter_deg > 0.0:
        basis_u, basis_v = _orthonormal_basis(anchor)
        yaw = np.deg2rad(float(rng.normal(0.0, axis_jitter_deg)))
        pitch = np.deg2rad(float(rng.normal(0.0, axis_jitter_deg * 0.35)))
        base_direction = anchor + np.tan(yaw) * basis_u + np.tan(pitch) * basis_v
        base_direction = _normalize_vector(base_direction, fallback=anchor)

    sweep_half_angle = np.deg2rad(max(0.0, float(fan_half_angle_deg)))
    basis_u, basis_v = _orthonormal_basis(base_direction)
    if n == 1 or sweep_half_angle <= 0.0:
        sweep_angles = np.zeros((n,), dtype=np.float32)
    else:
        sweep_angles = np.linspace(-sweep_half_angle, sweep_half_angle, n, dtype=np.float32)
        if sweep_jitter_deg > 0.0:
            max_jitter = sweep_half_angle / max(1, n - 1)
            sweep_jitter = np.deg2rad(float(sweep_jitter_deg))
            sweep_angles += rng.normal(0.0, min(sweep_jitter, max_jitter), size=n).astype(np.float32)
            sweep_angles = np.clip(sweep_angles, -sweep_half_angle, sweep_half_angle)
            sweep_angles.sort()

    if elevation_jitter_deg > 0.0:
        elevation_angles = np.deg2rad(rng.normal(0.0, elevation_jitter_deg, size=n)).astype(np.float32)
    else:
        elevation_angles = np.zeros((n,), dtype=np.float32)

    normals = (
        base_direction[None, :]
        + np.tan(sweep_angles)[:, None] * basis_u[None, :]
        + np.tan(elevation_angles)[:, None] * basis_v[None, :]
    ).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals.astype(np.float32)


def _bbox_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
        ],
        dtype=np.float32,
    )


def _sample_axis_positions(
    count: int,
    lo: float,
    hi: float,
    rng: np.random.Generator,
    *,
    ordered: bool,
) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.float32)
    if count == 1:
        return np.asarray([0.5 * (lo + hi)], dtype=np.float32)
    if ordered:
        return np.linspace(lo, hi, count, dtype=np.float32)
    return rng.uniform(lo, hi, size=count).astype(np.float32)


def sample_cross_axis_probe_geometry_bbox(
    n: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    center: np.ndarray,
    rng: np.random.Generator,
    *,
    preferred_axis: Sequence[float] | None = None,
    axis_jitter_deg: float = 20.0,
    pos_sigma_vox: float = 1.0,
    depth_sigma_vox: float = 6.0,
    tilt_sigma: float = 0.08,
    ordered: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate a translated freehand ultrasound sweep with two principal passes.

    Each plane is defined by a point p_i and a normal n_i.
    """
    if n <= 0:
        raise ValueError(f"n must be >= 1, got {n}.")

    bbox_min = np.asarray(bbox_min, dtype=np.float32).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float32).reshape(3)
    center = np.asarray(center, dtype=np.float32).reshape(3)

    anchor_source = (0.0, 0.0, 1.0) if preferred_axis is None else preferred_axis
    anchor = _normalize_vector(anchor_source)
    depth_dir = anchor.copy()
    if axis_jitter_deg > 0.0:
        jitter_u, jitter_v = _orthonormal_basis(anchor)
        yaw = np.deg2rad(float(rng.normal(0.0, axis_jitter_deg)))
        pitch = np.deg2rad(float(rng.normal(0.0, axis_jitter_deg * 0.35)))
        depth_dir = anchor + np.tan(yaw) * jitter_u + np.tan(pitch) * jitter_v
        depth_dir = _normalize_vector(depth_dir, fallback=anchor)

    sweep_u, sweep_v = _orthonormal_basis(depth_dir)
    rel_corners = _bbox_corners(bbox_min, bbox_max) - center[None, :]
    proj_u = rel_corners @ sweep_u
    proj_v = rel_corners @ sweep_v
    proj_d = rel_corners @ depth_dir

    u_min, u_max = float(proj_u.min()), float(proj_u.max())
    v_min, v_max = float(proj_v.min()), float(proj_v.max())
    d_min, d_max = float(proj_d.min()), float(proj_d.max())

    k_u = n // 2
    k_v = n - k_u
    points: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    axis_ids: List[int] = []

    if k_u > 0:
        sweep_vals = _sample_axis_positions(k_u, u_min, u_max, rng, ordered=ordered)
        cross_vals = np.clip(rng.normal(0.0, pos_sigma_vox, size=k_u).astype(np.float32), v_min, v_max)
        depth_vals = np.clip(rng.normal(0.0, depth_sigma_vox, size=k_u).astype(np.float32), d_min, d_max)
        tilt_sweep = rng.normal(0.0, tilt_sigma, size=k_u).astype(np.float32)
        tilt_depth = rng.normal(0.0, tilt_sigma * 0.35, size=k_u).astype(np.float32)
        base_normal = _normalize_vector(np.cross(depth_dir, sweep_u), fallback=sweep_v)
        for sweep_pos, cross_pos, depth_pos, tilt_a, tilt_b in zip(
            sweep_vals,
            cross_vals,
            depth_vals,
            tilt_sweep,
            tilt_depth,
        ):
            point = center + sweep_pos * sweep_u + cross_pos * sweep_v + depth_pos * depth_dir
            point = np.clip(point, bbox_min, bbox_max).astype(np.float32)
            normal = _normalize_vector(base_normal + tilt_a * sweep_u + tilt_b * depth_dir, fallback=base_normal)
            points.append(point)
            normals.append(normal)
            axis_ids.append(0)

    if k_v > 0:
        sweep_vals = _sample_axis_positions(k_v, v_min, v_max, rng, ordered=ordered)
        cross_vals = np.clip(rng.normal(0.0, pos_sigma_vox, size=k_v).astype(np.float32), u_min, u_max)
        depth_vals = np.clip(rng.normal(0.0, depth_sigma_vox, size=k_v).astype(np.float32), d_min, d_max)
        tilt_sweep = rng.normal(0.0, tilt_sigma, size=k_v).astype(np.float32)
        tilt_depth = rng.normal(0.0, tilt_sigma * 0.35, size=k_v).astype(np.float32)
        base_normal = _normalize_vector(np.cross(depth_dir, sweep_v), fallback=sweep_u)
        for sweep_pos, cross_pos, depth_pos, tilt_a, tilt_b in zip(
            sweep_vals,
            cross_vals,
            depth_vals,
            tilt_sweep,
            tilt_depth,
        ):
            point = center + cross_pos * sweep_u + sweep_pos * sweep_v + depth_pos * depth_dir
            point = np.clip(point, bbox_min, bbox_max).astype(np.float32)
            normal = _normalize_vector(base_normal + tilt_a * sweep_v + tilt_b * depth_dir, fallback=base_normal)
            points.append(point)
            normals.append(normal)
            axis_ids.append(1)

    points_arr = np.stack(points, axis=0).astype(np.float32)
    normals_arr = np.stack(normals, axis=0).astype(np.float32)
    axis_ids_arr = np.asarray(axis_ids, dtype=np.int64)
    if ordered:
        return points_arr, normals_arr, axis_ids_arr
    perm = rng.permutation(n)
    return points_arr[perm], normals_arr[perm], axis_ids_arr[perm]


def sample_slice_normals(
    n: int,
    rng: np.random.Generator,
    *,
    strategy: str = "random",
    preferred_axis: Sequence[float] | None = None,
    axis_jitter_deg: float = 25.0,
    fan_half_angle_deg: float = 35.0,
    elevation_jitter_deg: float = 6.0,
    sweep_jitter_deg: float = 2.5,
) -> np.ndarray:
    strategy = str(strategy).strip().lower()
    if strategy == "random":
        return random_normals(n, rng)
    if strategy == "fibonacci":
        return fibonacci_normals(n)
    if strategy == "ultrasound_fan":
        return sample_ultrasound_fan_normals(
            n,
            rng,
            preferred_axis=preferred_axis,
            axis_jitter_deg=axis_jitter_deg,
            fan_half_angle_deg=fan_half_angle_deg,
            elevation_jitter_deg=elevation_jitter_deg,
            sweep_jitter_deg=sweep_jitter_deg,
        )
    raise ValueError(f"Unknown slice normal strategy: {strategy}")


def sample_slice_planes(
    n: int,
    rng: np.random.Generator,
    *,
    strategy: str = "random",
    center: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    preferred_axis: Sequence[float] | None = None,
    axis_jitter_deg: float = 25.0,
    fan_half_angle_deg: float = 35.0,
    elevation_jitter_deg: float = 6.0,
    sweep_jitter_deg: float = 2.5,
    probe_pos_sigma_vox: float = 1.0,
    probe_depth_sigma_vox: float = 6.0,
    probe_tilt_sigma: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    strategy = str(strategy).strip().lower()
    center = np.asarray(center, dtype=np.float32).reshape(3)
    if strategy == "ultrasound_probe":
        points, normals, _axis_ids = sample_cross_axis_probe_geometry_bbox(
            n,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            center=center,
            rng=rng,
            preferred_axis=preferred_axis,
            axis_jitter_deg=axis_jitter_deg,
            pos_sigma_vox=probe_pos_sigma_vox,
            depth_sigma_vox=probe_depth_sigma_vox,
            tilt_sigma=probe_tilt_sigma,
            ordered=True,
        )
        return points.astype(np.float32), normals.astype(np.float32)

    normals = sample_slice_normals(
        n,
        rng,
        strategy=strategy,
        preferred_axis=preferred_axis,
        axis_jitter_deg=axis_jitter_deg,
        fan_half_angle_deg=fan_half_angle_deg,
        elevation_jitter_deg=elevation_jitter_deg,
        sweep_jitter_deg=sweep_jitter_deg,
    )
    points = np.repeat(center[None, :], n, axis=0).astype(np.float32)
    return points, normals.astype(np.float32)


def simulate_sparse_acquisition(
    vol: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray | None = None,
    points: np.ndarray | None = None,
    thickness_vox: float = 1.0,
    chunk: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified sparse acquisition for centered or translated slice planes.

    plane: |(x - p_i) dot n_i| <= thickness_vox
    """
    vol = vol.astype(np.float32)
    D, H, W = vol.shape

    coords = np.indices((D, H, W)).astype(np.float32)
    if points is None:
        if center is None:
            raise ValueError("Either center or points must be provided.")
        center = np.asarray(center, dtype=np.float32).reshape(3)
        points = np.repeat(center[None, :], normals.shape[0], axis=0).astype(np.float32)
    else:
        points = np.asarray(points, dtype=np.float32)
        if points.shape != normals.shape:
            raise ValueError(f"points and normals must match, got {points.shape} vs {normals.shape}.")

    recon = np.zeros_like(vol, dtype=np.float32)
    hits = np.zeros_like(vol, dtype=np.float32)

    for start in range(0, normals.shape[0], chunk):
        n = normals[start:start + chunk]
        p = points[start:start + chunk]
        dist = (
            (coords[0][None, ...] - p[:, 0, None, None, None]) * n[:, 0, None, None, None]
            + (coords[1][None, ...] - p[:, 1, None, None, None]) * n[:, 1, None, None, None]
            + (coords[2][None, ...] - p[:, 2, None, None, None]) * n[:, 2, None, None, None]
        )
        m = np.abs(dist) <= thickness_vox
        hits += m.sum(axis=0).astype(np.float32)
        recon += (m * vol[None, ...]).sum(axis=0).astype(np.float32)

    sparse = np.divide(recon, hits, out=np.zeros_like(recon), where=hits > 0)
    mask = (hits > 0).astype(np.float32)
    return sparse, mask, hits


@dataclass
class InpaintingSample:
    x: np.ndarray  # [2,D,H,W] (sparse + mask)
    y: np.ndarray  # [1,D,H,W] (gt)
    meta: dict
