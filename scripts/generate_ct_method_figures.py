from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, map_coordinates

from inpainting3d.data import (
    center_of_mass_threshold,
    foreground_bbox_threshold,
    load_nifti_volume,
    normalize_volume,
    resample_to_shape,
    sample_slice_planes,
    simulate_sparse_acquisition,
)
from inpainting3d.models import DeepResUNet3D, PartialResUNet3D
from inpainting3d.utils import (
    ensure_dir,
    finalize_inpainting_prediction,
    strip_dataparallel_prefix,
    torch_load_weights_compat,
)


DEFAULT_VOLUME = Path(r"E:\Bachelorarbeit_Daten\pipeline_preprocessed_data_256\liver_11.nii.gz")
DEFAULT_OUT_DIR = ROOT / "results" / "thesis_ct_method_figures"
DEFAULT_WEIGHTS = (
    ROOT.parent
    / "liver_sparse_reconstruction_core"
    / "runs"
    / "ablation_final_128"
    / "standard_unet3d_no_partialconv"
    / "train"
    / "best_model.pt"
)
AXIS_NAME_TO_ZYX = {"axial": 0, "coronal": 1, "sagittal": 2}
PROJECT_SLICE_THRESHOLD = 0.075
PROJECT_SLICE_SIGMA = 0.85


def configure_matplotlib() -> None:
    """
    Configure matplotlib defaults for figure generation.
    
    Returns
    -------
    None
        Matplotlib global state is updated in place.
    """
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments for this script.
    
    Returns
    -------
    argparse.Namespace
        Parsed args.
    """
    parser = argparse.ArgumentParser(
        description="Generate CT-based method figures for Sections 5.2, 5.3, and 5.6."
    )
    parser.add_argument("--volume", type=str, default=str(DEFAULT_VOLUME))
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--slice-count", type=int, default=6)
    parser.add_argument("--thickness-vox", type=float, default=1.0)
    parser.add_argument(
        "--slice-geometry",
        type=str,
        default="uniform_axis",
        choices=["uniform_axis", "ultrasound_probe", "random", "fibonacci", "ultrasound_fan"],
    )
    parser.add_argument(
        "--uniform-axis",
        type=str,
        default="coronal",
        choices=["axial", "coronal", "sagittal"],
    )
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "partial"])
    parser.add_argument("--init-feat", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--formats", type=str, default="png,pdf")
    return parser.parse_args()


def resolve_formats(raw: str) -> list[str]:
    """
    Resolve and normalize the requested output figure formats.
    
    Parameters
    ----------
    raw : str
        Raw input value or array before normalization.
    
    Returns
    -------
    list[str]
        Resolved value or selection.
    """
    formats = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return formats or ["png"]


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Sequence[str]) -> list[str]:
    """
    Save a matplotlib figure to one or more output files.
    
    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure to save or annotate.
    out_dir : Path
        Destination directory for generated outputs.
    stem : str
        Filename stem used when building output paths.
    formats : Sequence[str]
        Output formats requested by the user.
    
    Returns
    -------
    None
        The requested artifact is written to disk.
    """
    ensure_dir(str(out_dir))
    saved: list[str] = []
    for fmt in formats:
        out_path = out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, format=fmt, dpi=220, bbox_inches="tight", pad_inches=0.05)
        saved.append(str(out_path))
    plt.close(fig)
    return saved


def normalize01(arr: np.ndarray) -> np.ndarray:
    """
    Normalize values into the unit interval.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array to normalize or transform.
    
    Returns
    -------
    np.ndarray
        Array normalized into the unit interval.
    """
    arr = arr.astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def load_volume(path: Path, dim: int) -> np.ndarray:
    """
    Load a volumetric image from disk.
    
    Parameters
    ----------
    path : Path
        Filesystem path to the input artifact.
    dim : int
        Target cubic side length of the processed volume.
    
    Returns
    -------
    np.ndarray
        Loaded data structure.
    """
    volume, _affine = load_nifti_volume(str(path), canonical=True, dtype=np.float32)
    volume = normalize_volume(volume, method="minmax")
    volume = resample_to_shape(volume, (dim, dim, dim), order=1)
    return np.clip(volume.astype(np.float32), 0.0, 1.0)


def orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct an orthonormal basis around a direction vector.
    
    Parameters
    ----------
    direction : np.ndarray
        Direction vector used to build a local basis.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two orthonormal basis vectors spanning the plane perpendicular to the input direction.
    """
    direction = np.asarray(direction, dtype=np.float32).reshape(3)
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(direction, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    axis_u = np.cross(direction, reference)
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-8)
    axis_v = np.cross(direction, axis_u)
    axis_v /= max(float(np.linalg.norm(axis_v)), 1e-8)
    return axis_u.astype(np.float32), axis_v.astype(np.float32)


def sample_volume_points(
    volume: np.ndarray,
    *,
    threshold: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample foreground points from a volume for visualization.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume array.
    threshold : float
        Threshold used to define the foreground region.
    max_points : int
        Maximum number of points sampled for visualization.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sampled values in the requested representation.
    """
    coords = np.argwhere(volume > threshold)
    if coords.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    values = volume[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
    if coords.shape[0] > max_points:
        stride = max(1, coords.shape[0] // max_points)
        coords = coords[::stride]
        values = values[::stride]
    xyz = coords[:, [2, 1, 0]].astype(np.float32)
    return xyz, values


def draw_bounding_box(ax: plt.Axes, shape: tuple[int, int, int], color: str = "#808080") -> None:
    """
    Draw a 3D bounding box onto an axis.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis used for plotting.
    shape : tuple[int, int, int]
        Spatial shape of the volume or plotting region.
    color : str
        Color used for plotting or annotation. Defaults to "#808080".
    
    Returns
    -------
    None
        The bounding box is drawn onto the provided axis.
    """
    depth, height, width = shape
    corners = np.array(
        [
            [0, 0, 0],
            [width - 1, 0, 0],
            [width - 1, height - 1, 0],
            [0, height - 1, 0],
            [0, 0, depth - 1],
            [width - 1, 0, depth - 1],
            [width - 1, height - 1, depth - 1],
            [0, height - 1, depth - 1],
        ],
        dtype=np.float32,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for start, end in edges:
        xs, ys, zs = zip(corners[start], corners[end])
        ax.plot(xs, ys, zs, color=color, linewidth=0.8, alpha=0.8)


def set_equal_axes(ax: plt.Axes, shape: tuple[int, int, int]) -> None:
    """
    Set equal scaling across all axes of a 3D plot.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis used for plotting.
    shape : tuple[int, int, int]
        Spatial shape of the volume or plotting region.
    
    Returns
    -------
    None
        The axis limits are updated in place.
    """
    depth, height, width = shape
    max_dim = max(depth, height, width)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    cz = (depth - 1) / 2.0
    radius = max_dim / 2.0
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_zlim(cz - radius, cz + radius)
    try:
        ax.set_box_aspect((width, height, depth))
    except Exception:
        pass


def style_axis(ax: plt.Axes, title: str) -> None:
    """
    Apply consistent styling to a matplotlib axis.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis used for plotting.
    title : str
        Plot or figure title.
    
    Returns
    -------
    None
        The axis styling is updated in place.
    """
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def draw_intensity_cloud(
    ax: plt.Axes,
    volume: np.ndarray,
    *,
    threshold: float,
    max_points: int,
    alpha: float,
    title: str,
) -> None:
    """
    Draw a point cloud colored by intensity values.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis used for plotting.
    volume : np.ndarray
        Input volume array.
    threshold : float
        Threshold used to define the foreground region.
    max_points : int
        Maximum number of points sampled for visualization.
    alpha : float
        Opacity used when rendering a plotted element.
    title : str
        Plot or figure title.
    
    Returns
    -------
    None
        The point cloud is drawn onto the provided axis.
    """
    xyz, values = sample_volume_points(volume, threshold=threshold, max_points=max_points)
    if xyz.shape[0] > 0:
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=values,
            cmap="gray",
            s=1.6,
            alpha=alpha,
            linewidths=0,
            vmin=0.0,
            vmax=1.0,
        )
    draw_bounding_box(ax, volume.shape)
    set_equal_axes(ax, volume.shape)
    style_axis(ax, title)


def draw_intensity_voxels(
    ax: plt.Axes,
    volume: np.ndarray,
    *,
    threshold: float,
    stride: int,
    alpha: float,
    title: str,
) -> None:
    """
    Draw sparse voxels colored by intensity values.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis used for plotting.
    volume : np.ndarray
        Input volume array.
    threshold : float
        Threshold used to define the foreground region.
    stride : int
        Convolution stride.
    alpha : float
        Opacity used when rendering a plotted element.
    title : str
        Plot or figure title.
    
    Returns
    -------
    None
        The voxels are drawn onto the provided axis.
    """
    stride = max(int(stride), 1)
    depth, height, width = volume.shape
    vol_small = volume[::stride, ::stride, ::stride].astype(np.float32)
    filled = vol_small > float(threshold)
    if np.any(filled):
        values = normalize01(vol_small)
        facecolors = plt.get_cmap("gray")(values)
        facecolors[..., 3] = alpha * filled.astype(np.float32)
        edge_rgba = (0.18, 0.18, 0.18, min(alpha * 0.18, 0.08))
        x_edges = np.linspace(0.0, float(width), vol_small.shape[2] + 1, dtype=np.float32)
        y_edges = np.linspace(0.0, float(height), vol_small.shape[1] + 1, dtype=np.float32)
        z_edges = np.linspace(0.0, float(depth), vol_small.shape[0] + 1, dtype=np.float32)
        xx, yy, zz = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")
        ax.voxels(
            xx,
            yy,
            zz,
            filled.transpose(2, 1, 0),
            facecolors=facecolors.transpose(2, 1, 0, 3),
            edgecolors=edge_rgba,
            linewidth=0.02,
        )
    draw_bounding_box(ax, volume.shape)
    set_equal_axes(ax, volume.shape)
    style_axis(ax, title)


def draw_mask_cloud(
    ax: plt.Axes,
    mask: np.ndarray,
    *,
    max_points: int,
    title: str,
) -> None:
    """
    Draw a point cloud from a binary mask.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis used for plotting.
    mask : np.ndarray
        Binary mask that marks valid or selected voxels.
    max_points : int
        Maximum number of points sampled for visualization.
    title : str
        Plot or figure title.
    
    Returns
    -------
    None
        The point cloud is drawn onto the provided axis.
    """
    xyz, values = sample_volume_points(mask.astype(np.float32), threshold=0.5, max_points=max_points)
    if xyz.shape[0] > 0:
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c="#222222",
            s=1.2,
            alpha=0.08,
            linewidths=0,
        )
    draw_bounding_box(ax, mask.shape)
    set_equal_axes(ax, mask.shape)
    style_axis(ax, title)


def axis_name_to_normal(axis_name: str) -> np.ndarray:
    """
    Map an anatomical axis label to its normal vector.
    
    Parameters
    ----------
    axis_name : str
        Axis name used to select an anatomical direction.
    
    Returns
    -------
    np.ndarray
        Unit normal vector associated with the requested axis name.
    """
    axis = str(axis_name).strip().lower()
    if axis == "axial":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if axis == "coronal":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if axis == "sagittal":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    raise ValueError(f"Unknown axis: {axis_name}")


def build_project_foreground_mask(
    volume: np.ndarray,
    *,
    threshold: float = PROJECT_SLICE_THRESHOLD,
    sigma: float = PROJECT_SLICE_SIGMA,
) -> np.ndarray:
    """
    Build a foreground mask for projected visualization geometry.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume array.
    threshold : float
        Threshold used to define the foreground region. Defaults to PROJECT_SLICE_THRESHOLD.
    sigma : float
        Standard deviation of the Gaussian kernel. Defaults to PROJECT_SLICE_SIGMA.
    
    Returns
    -------
    np.ndarray
        Constructed object ready for downstream use.
    """
    smoothed = gaussian_filter(volume.astype(np.float32), sigma=float(sigma))
    return smoothed > float(threshold)


def build_project_slice_mask(
    foreground_mask: np.ndarray,
    axis_name: str,
    slice_count: int,
) -> tuple[np.ndarray, list[int], int]:
    """
    Build a mask that marks the sampled slice support.
    
    Parameters
    ----------
    foreground_mask : np.ndarray
        Binary mask that marks foreground voxels.
    axis_name : str
        Axis name used to select an anatomical direction.
    slice_count : int
        Number of slice planes represented by the observation.
    
    Returns
    -------
    tuple[np.ndarray, list[int], int]
        Constructed object ready for downstream use.
    """
    axis = AXIS_NAME_TO_ZYX[str(axis_name).strip().lower()]
    coords = np.argwhere(foreground_mask)
    if coords.shape[0] == 0:
        return np.zeros_like(foreground_mask, dtype=bool), [], axis

    lo = int(coords[:, axis].min())
    hi = int(coords[:, axis].max())
    indices = np.linspace(lo, hi, max(int(slice_count), 1))
    indices = np.unique(np.rint(indices).astype(np.int32))
    indices = np.clip(indices, 0, foreground_mask.shape[axis] - 1)

    slice_mask = np.zeros_like(foreground_mask, dtype=bool)
    slicer: list[slice | np.ndarray] = [slice(None), slice(None), slice(None)]
    slicer[axis] = indices
    slice_mask[tuple(slicer)] = True
    return slice_mask, [int(i) for i in indices.tolist()], axis


def sparse_volume_from_axis_slices(volume: np.ndarray, axis: int, indices: list[int]) -> np.ndarray:
    """
    Construct a sparse volume from axis-aligned slices.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume array.
    axis : int
        Axis index or plotting axis used by the helper.
    indices : list[int]
        Slice indices or sampled indices used by the helper.
    
    Returns
    -------
    np.ndarray
        Sparse volume assembled from the requested axis-aligned slices.
    """
    sparse = np.zeros_like(volume, dtype=np.float32)
    slicer: list[slice | list[int]] = [slice(None), slice(None), slice(None)]
    slicer[axis] = indices
    sparse[tuple(slicer)] = volume[tuple(slicer)]
    return sparse


def points_normals_from_axis_indices(
    center: np.ndarray,
    *,
    axis_name: str,
    indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert axis-aligned slice indices into plane points and normals.
    
    Parameters
    ----------
    center : np.ndarray
        Foreground center expressed in voxel coordinates.
    axis_name : str
        Axis name used to select an anatomical direction.
    indices : Sequence[int]
        Slice indices or sampled indices used by the helper.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Plane points and normals derived from the selected slice indices.
    """
    axis_index = AXIS_NAME_TO_ZYX[str(axis_name).strip().lower()]
    normal = axis_name_to_normal(axis_name)
    points = np.repeat(np.asarray(center, dtype=np.float32)[None, :], len(indices), axis=0)
    if len(indices) > 0:
        points[:, axis_index] = np.asarray(indices, dtype=np.float32)
    normals = np.repeat(normal[None, :], len(indices), axis=0)
    return points.astype(np.float32), normals.astype(np.float32)


def build_project_axis_observation(
    volume: np.ndarray,
    center: np.ndarray,
    *,
    axis_name: str,
    slice_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """
    Build a projected sparse observation for one axis configuration.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume array.
    center : np.ndarray
        Foreground center expressed in voxel coordinates.
    axis_name : str
        Axis name used to select an anatomical direction.
    slice_count : int
        Number of slice planes represented by the observation.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]
        Constructed object ready for downstream use.
    """
    foreground_mask = build_project_foreground_mask(volume)
    slice_mask, selected_indices, axis = build_project_slice_mask(foreground_mask, axis_name, slice_count)
    masked_volume = volume * foreground_mask.astype(np.float32)
    sparse = sparse_volume_from_axis_slices(masked_volume, axis, selected_indices)
    mask = (foreground_mask & slice_mask).astype(np.float32)
    hits = mask.copy()
    points, normals = points_normals_from_axis_indices(center, axis_name=axis_name, indices=selected_indices)
    meta = {
        "selected_slice_indices": [int(i) for i in selected_indices],
        "selected_slice_count": int(len(selected_indices)),
        "slice_axis_zyx_index": int(axis),
        "foreground_voxels_full": int(np.count_nonzero(foreground_mask)),
        "foreground_voxels_selected_slices": int(np.count_nonzero(mask > 0.5)),
        "slice_method": "project_render_sparse_volume",
    }
    return points, normals, sparse.astype(np.float32), mask, hits, meta


def build_observation_geometry(
    *,
    geometry: str,
    slice_count: int,
    center: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    rng: np.random.Generator,
    uniform_axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble sparse-observation geometry for a visualization figure.
    
    Parameters
    ----------
    geometry : str
        Sparse-acquisition geometry identifier or configuration.
    slice_count : int
        Number of slice planes represented by the observation.
    center : np.ndarray
        Foreground center expressed in voxel coordinates.
    bbox_min : np.ndarray
        Lower corner of a foreground bounding box.
    bbox_max : np.ndarray
        Upper corner of a foreground bounding box.
    rng : np.random.Generator
        Random number generator used for stochastic sampling.
    uniform_axis : str
        Axis used for uniform axis-aligned slice placement.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Constructed object ready for downstream use.
    """
    geometry_name = str(geometry).strip().lower()
    return sample_slice_planes(
        int(slice_count),
        rng,
        strategy=geometry_name,
        center=center,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        preferred_axis=(0.0, 0.0, 1.0),
        axis_jitter_deg=18.0,
        probe_pos_sigma_vox=1.2,
        probe_depth_sigma_vox=5.5,
        probe_tilt_sigma=0.06,
    )


def draw_textured_plane(
    ax: plt.Axes,
    volume: np.ndarray,
    *,
    point: np.ndarray,
    normal: np.ndarray,
    plane_size: float,
    resolution: int,
    alpha: float,
    edgecolor: str = "#5c5c5c",
    intensity_threshold: float = 0.06,
) -> None:
    """
    Draw a textured plane patch in 3D.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis used for plotting.
    volume : np.ndarray
        Input volume array.
    point : np.ndarray
        Point lying on a sampling plane.
    normal : np.ndarray
        Normal vector of a sampling plane.
    plane_size : float
        Side length of the rendered support plane.
    resolution : int
        Sampling resolution of the rendered support plane.
    alpha : float
        Opacity used when rendering a plotted element.
    edgecolor : str
        Edge color used for plotted geometry. Defaults to "#5c5c5c".
    intensity_threshold : float
        Threshold used when filtering low-intensity points or planes. Defaults to 0.06.
    
    Returns
    -------
    None
        The textured plane is drawn onto the provided axis.
    """
    axis_u, axis_v = orthonormal_basis(normal)
    offsets = np.linspace(-plane_size / 2.0, plane_size / 2.0, resolution, dtype=np.float32)
    uu, vv = np.meshgrid(offsets, offsets, indexing="xy")
    coords = point[:, None, None] + axis_u[:, None, None] * uu[None, :, :] + axis_v[:, None, None] * vv[None, :, :]
    shape = np.array(volume.shape, dtype=np.float32)
    valid = np.all((coords >= 0.0) & (coords <= (shape[:, None, None] - 1.0)), axis=0)
    image = map_coordinates(volume, coords, order=1, mode="constant", cval=0.0).astype(np.float32)
    image = np.where(valid, image, 0.0)
    support = valid & (image > intensity_threshold)
    facecolors = plt.get_cmap("gray")(normalize01(image))
    facecolors[..., 3] = alpha * support.astype(np.float32)
    xyz = coords[[2, 1, 0], :, :]
    ax.plot_surface(
        xyz[0],
        xyz[1],
        xyz[2],
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    corners = np.stack(
        [
            point - plane_size / 2.0 * axis_u - plane_size / 2.0 * axis_v,
            point + plane_size / 2.0 * axis_u - plane_size / 2.0 * axis_v,
            point + plane_size / 2.0 * axis_u + plane_size / 2.0 * axis_v,
            point - plane_size / 2.0 * axis_u + plane_size / 2.0 * axis_v,
        ],
        axis=0,
    )
    corners_xyz = corners[:, [2, 1, 0]]
    closed = np.vstack([corners_xyz, corners_xyz[:1]])
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=edgecolor, linewidth=1.0, alpha=0.8)


def draw_support_planes(
    ax: plt.Axes,
    *,
    volume_shape: tuple[int, int, int],
    points: np.ndarray,
    normals: np.ndarray,
    plane_size: float,
    color: str,
    alpha: float,
    linewidth: float = 0.9,
) -> None:
    """
    Draw the supporting slice planes for an observation.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis used for plotting.
    volume_shape : tuple[int, int, int]
        Spatial shape of the represented volume.
    points : np.ndarray
        Points through which the sampled slice planes pass.
    normals : np.ndarray
        Plane normal vectors that define sampled slices.
    plane_size : float
        Side length of the rendered support plane.
    color : str
        Color used for plotting or annotation.
    alpha : float
        Opacity used when rendering a plotted element.
    linewidth : float
        Line width used for plotted geometry. Defaults to 0.9.
    
    Returns
    -------
    None
        The support planes are drawn onto the provided axis.
    """
    facecolor = mpl.colors.to_rgba(color, alpha=alpha)
    for point, normal in zip(points, normals):
        axis_u, axis_v = orthonormal_basis(normal)
        corners = np.stack(
            [
                point - plane_size / 2.0 * axis_u - plane_size / 2.0 * axis_v,
                point + plane_size / 2.0 * axis_u - plane_size / 2.0 * axis_v,
                point + plane_size / 2.0 * axis_u + plane_size / 2.0 * axis_v,
                point - plane_size / 2.0 * axis_u + plane_size / 2.0 * axis_v,
            ],
            axis=0,
        )
        corners_xyz = corners[:, [2, 1, 0]]
        poly = Poly3DCollection(
            [corners_xyz],
            facecolors=[facecolor],
            edgecolors=[mpl.colors.to_rgba(color, alpha=min(alpha + 0.18, 0.95))],
            linewidths=linewidth,
        )
        ax.add_collection3d(poly)
    draw_bounding_box(ax, volume_shape)
    set_equal_axes(ax, volume_shape)


def create_oriented_slices_figure(
    volume: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    geometry_label: str,
    observation_title: str,
    out_dir: Path,
    formats: Sequence[str],
) -> dict[str, object]:
    """
    Create the oriented-slices method figure.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume array.
    points : np.ndarray
        Points through which the sampled slice planes pass.
    normals : np.ndarray
        Plane normal vectors that define sampled slices.
    geometry_label : str
        Display label for the selected acquisition geometry.
    observation_title : str
        Title shown for the rendered observation panel.
    out_dir : Path
        Destination directory for generated outputs.
    formats : Sequence[str]
        Output formats requested by the user.
    
    Returns
    -------
    dict[str, object]
        Newly created object or artifact.
    """
    fig = plt.figure(figsize=(13.5, 6.2))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")

    draw_intensity_voxels(
        ax_left,
        volume,
        threshold=0.08,
        stride=2,
        alpha=0.22,
        title="CT-Referenzvolumen",
    )
    draw_intensity_cloud(
        ax_right,
        volume,
        threshold=0.06,
        max_points=26000,
        alpha=0.05,
        title=observation_title,
    )

    plane_size = 0.88 * float(np.max(volume.shape))
    for point, normal in zip(points[:8], normals[:8]):
        draw_textured_plane(
            ax_right,
            volume,
            point=point,
            normal=normal,
            plane_size=plane_size,
            resolution=96,
            alpha=0.48,
            intensity_threshold=0.06,
        )

    for ax in (ax_left, ax_right):
        ax.view_init(elev=22, azim=-56)

    fig.suptitle(geometry_label, y=0.97, fontweight="bold")
    fig.text(
        0.5,
        0.03,
        "Rechts sind beispielhafte Beobachtungsebenen mit den zugehörigen CT-Grauwertintensitäten in das gemeinsame Volumen eingebettet.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    saved = save_figure(fig, out_dir, "figure_52_oriented_nonconsecutive_slices_ct", formats)
    return {"files": saved}


def create_sparse_volume_figure(
    volume: np.ndarray,
    sparse: np.ndarray,
    mask: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    mask_render_mode: str,
    out_dir: Path,
    formats: Sequence[str],
) -> dict[str, object]:
    """
    Create the sparse-volume method figure.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume array.
    sparse : np.ndarray
        Sparse observation tensor or array.
    mask : np.ndarray
        Binary mask that marks valid or selected voxels.
    points : np.ndarray
        Points through which the sampled slice planes pass.
    normals : np.ndarray
        Plane normal vectors that define sampled slices.
    mask_render_mode : str
        Rendering mode used for masks in the figure.
    out_dir : Path
        Destination directory for generated outputs.
    formats : Sequence[str]
        Output formats requested by the user.
    
    Returns
    -------
    dict[str, object]
        Newly created object or artifact.
    """
    fig = plt.figure(figsize=(18.2, 6.2))
    ax_a = fig.add_subplot(1, 3, 1, projection="3d")
    ax_b = fig.add_subplot(1, 3, 2, projection="3d")
    ax_c = fig.add_subplot(1, 3, 3, projection="3d")

    draw_intensity_voxels(
        ax_a,
        volume,
        threshold=0.08,
        stride=2,
        alpha=0.22,
        title="CT-Referenzvolumen",
    )
    draw_intensity_cloud(
        ax_b,
        sparse,
        threshold=0.02,
        max_points=26000,
        alpha=0.16,
        title="Sparse Volume",
    )
    if str(mask_render_mode).strip().lower() == "cloud":
        draw_mask_cloud(
            ax_c,
            mask,
            max_points=24000,
            title="Known Mask",
        )
    else:
        plane_size = 0.88 * float(np.max(volume.shape))
        draw_support_planes(
            ax_c,
            volume_shape=volume.shape,
            points=points,
            normals=normals,
            plane_size=plane_size,
            color="#1d4f91",
            alpha=0.14,
        )
        style_axis(ax_c, "Known Mask")

    for ax in (ax_a, ax_b, ax_c):
        ax.view_init(elev=22, azim=-56)

    fig.suptitle("Von beobachteten CT-Intensitäten zu Sparse Volume und Known Mask", y=0.97, fontweight="bold")
    fig.text(
        0.5,
        0.03,
        "Das Sparse Volume speichert die beobachteten CT-Grauwerte an bekannten Positionen; die Known Mask markiert die entsprechende Stütze explizit.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    saved = save_figure(fig, out_dir, "figure_53_sparse_volume_known_mask_ct", formats)
    return {"files": saved}


def choose_midplane(mask: np.ndarray, center: np.ndarray) -> tuple[str, int]:
    """
    Choose a representative mid-plane index.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask that marks valid or selected voxels.
    center : np.ndarray
        Foreground center expressed in voxel coordinates.
    
    Returns
    -------
    tuple[str, int]
        Resolved value or selection.
    """
    z = int(np.clip(round(float(center[0])), 0, mask.shape[0] - 1))
    y = int(np.clip(round(float(center[1])), 0, mask.shape[1] - 1))
    x = int(np.clip(round(float(center[2])), 0, mask.shape[2] - 1))
    candidates = {
        "axial": (z, int(np.count_nonzero(mask[z] > 0.5))),
        "coronal": (y, int(np.count_nonzero(mask[:, y, :] > 0.5))),
        "sagittal": (x, int(np.count_nonzero(mask[:, :, x] > 0.5))),
    }
    plane = max(candidates.items(), key=lambda item: item[1][1])[0]
    return plane, candidates[plane][0]


def view_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    """
    Extract a 2D view from a volume for the requested orientation.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume array.
    plane : str
        Requested plane.
    index : int
        Zero-based index used for deterministic naming or selection.
    
    Returns
    -------
    np.ndarray
        Extracted slice.
    """
    if plane == "axial":
        return volume[index, :, :]
    if plane == "coronal":
        return volume[:, index, :]
    if plane == "sagittal":
        return volume[:, :, index]
    raise ValueError(f"Unknown plane: {plane}")


def build_model(kind: str, init_feat: int) -> torch.nn.Module:
    """
    Build a model instance for the requested architecture identifier.
    
    Parameters
    ----------
    kind : str
        Architecture identifier used to build a model.
    init_feat : int
        Base number of feature channels in the first stage.
    
    Returns
    -------
    torch.nn.Module
        Constructed object ready for downstream use.
    """
    model_kind = str(kind).strip().lower()
    if model_kind == "baseline":
        return DeepResUNet3D(in_channels=2, init_feat=init_feat)
    if model_kind == "partial":
        return PartialResUNet3D(init_feat=init_feat)
    raise ValueError(f"Unknown model kind: {kind}")


def resolve_device(raw: str) -> torch.device:
    """
    Resolve the torch device used for inference or plotting.
    
    Parameters
    ----------
    raw : str
        Raw input value or array before normalization.
    
    Returns
    -------
    torch.device
        Resolved value or selection.
    """
    choice = str(raw).strip().lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_real_inference(
    sparse: np.ndarray,
    mask: np.ndarray,
    *,
    weights: Path,
    model_kind: str,
    init_feat: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model inference for a real sparse observation.
    
    Parameters
    ----------
    sparse : np.ndarray
        Sparse observation tensor or array.
    mask : np.ndarray
        Binary mask that marks valid or selected voxels.
    weights : Path
        Checkpoint path or weighting coefficients used by the helper.
    model_kind : str
        Architecture identifier used to build a model.
    init_feat : int
        Base number of feature channels in the first stage.
    device : torch.device
        Torch device on which tensors should be created or evaluated.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Output produced by the model or workflow.
    """
    model = build_model(model_kind, init_feat=init_feat).to(device)
    state = torch_load_weights_compat(str(weights), map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    state = strip_dataparallel_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    x = np.stack([sparse, mask], axis=0)[None, ...].astype(np.float32)
    xt = torch.from_numpy(x).to(device)
    sparse_t = xt[:, 0:1, ...]
    mask_t = xt[:, 1:2, ...]
    with torch.no_grad():
        raw_t = torch.clamp(model(xt), 0.0, 1.0)
        constrained_t = finalize_inpainting_prediction(raw_t, sparse=sparse_t, known_mask=mask_t)
    raw = raw_t[0, 0].detach().cpu().numpy().astype(np.float32)
    constrained = constrained_t[0, 0].detach().cpu().numpy().astype(np.float32)
    return raw, constrained


def create_hard_constraint_figure(
    volume: np.ndarray,
    sparse: np.ndarray,
    mask: np.ndarray,
    raw: np.ndarray,
    constrained: np.ndarray,
    out_dir: Path,
    formats: Sequence[str],
) -> dict[str, object]:
    """
    Create a figure that illustrates the hard inpainting constraint.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume array.
    sparse : np.ndarray
        Sparse observation tensor or array.
    mask : np.ndarray
        Binary mask that marks valid or selected voxels.
    raw : np.ndarray
        Raw input value or array before normalization.
    constrained : np.ndarray
        Prediction generated after enforcing the hard constraint.
    out_dir : Path
        Destination directory for generated outputs.
    formats : Sequence[str]
        Output formats requested by the user.
    
    Returns
    -------
    dict[str, object]
        Newly created object or artifact.
    """
    center = center_of_mass_threshold(volume, thr=0.12)
    plane, index = choose_midplane(mask, center)

    gt_img = view_slice(volume, plane, index)
    sparse_img = view_slice(sparse, plane, index)
    mask_img = view_slice(mask, plane, index)
    raw_img = view_slice(raw, plane, index)
    constrained_img = view_slice(constrained, plane, index)

    err_before = np.abs(raw_img - sparse_img) * mask_img
    err_after = np.abs(constrained_img - sparse_img) * mask_img
    changed = np.abs(constrained_img - raw_img)

    gray_vmax = max(float(np.quantile(gt_img, 0.995)), 1e-6)
    err_vmax = max(float(np.quantile(err_before, 0.995)), 1e-6)
    change_vmax = max(float(np.quantile(changed, 0.995)), 1e-6)

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.0))
    panels = [
        ("Referenzslice", gt_img, "gray", 0.0, gray_vmax),
        ("Sparse Volume", sparse_img, "gray", 0.0, gray_vmax),
        ("Known Mask", mask_img, "gray", 0.0, 1.0),
        ("Rohe Netzvorhersage $\\tilde{u}$", raw_img, "gray", 0.0, gray_vmax),
        ("Hard-Constraint $\\hat{u}$", constrained_img, "gray", 0.0, gray_vmax),
        ("Fehler auf beobachteten Voxeln\nvor HC", err_before, "inferno", 0.0, err_vmax),
        ("Fehler auf beobachteten Voxeln\nnach HC", err_after, "inferno", 0.0, max(err_vmax, 1e-6)),
        ("Durch HC geänderte Voxel", changed, "viridis", 0.0, change_vmax),
    ]

    changed_cmap = plt.get_cmap("ocean").copy()
    changed_cmap.set_bad(color="black")
    changed_threshold = max(change_vmax * 1e-4, 1e-8)

    for ax, (title, img, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        if title.startswith("Durch HC"):
            masked = np.ma.masked_where(img <= changed_threshold, img)
            ax.imshow(masked.T, cmap=changed_cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_facecolor("black")
        else:
            ax.imshow(img.T, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        rf"Hard-Constraint im {plane.capitalize()}-Schnitt ({plane}={index}): $\hat{{u}} = m \odot s + (1-m)\odot \tilde{{u}}$",
        y=0.98,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.03,
        "Auf beobachteten Voxeln werden die Werte der rohen Vorhersage exakt durch die bekannten Eingabewerte ersetzt; "
        "nur fehlende Regionen bleiben modellbasiert.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.08, wspace=0.04, hspace=0.14)
    saved = save_figure(fig, out_dir, "figure_56_hard_constraint_slice_ct", formats)
    return {"files": saved, "plane": plane, "index": int(index)}


def main() -> None:
    """
    Execute the command-line entry point for this script.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    args = parse_args()
    configure_matplotlib()

    out_dir = Path(args.out).resolve()
    ensure_dir(str(out_dir))
    formats = resolve_formats(args.formats)
    device = resolve_device(args.device)

    volume_path = Path(args.volume)
    volume = load_volume(volume_path, dim=int(args.dim))
    center = center_of_mass_threshold(volume, thr=0.12)
    bbox_min, bbox_max = foreground_bbox_threshold(volume, thr=0.12)
    rng = np.random.default_rng(int(args.seed))
    observation_meta: dict[str, object] = {}
    if str(args.slice_geometry).strip().lower() == "uniform_axis":
        points, normals, sparse, mask, hits, observation_meta = build_project_axis_observation(
            volume,
            center,
            axis_name=str(args.uniform_axis),
            slice_count=int(args.slice_count),
        )
    else:
        points, normals = build_observation_geometry(
            geometry=str(args.slice_geometry),
            slice_count=int(args.slice_count),
            center=center,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            rng=rng,
            uniform_axis=str(args.uniform_axis),
        )
        sparse, mask, hits = simulate_sparse_acquisition(
            volume,
            normals=normals,
            points=points,
            thickness_vox=float(args.thickness_vox),
        )
    weights_path = Path(args.weights)
    raw_pred, constrained_pred = run_real_inference(
        sparse,
        mask,
        weights=weights_path,
        model_kind=str(args.model),
        init_feat=int(args.init_feat),
        device=device,
    )
    geometry_label = (
        f"Gleichverteilte {str(args.uniform_axis).capitalize()}-Schnitte im CT-Volumen"
        if str(args.slice_geometry).strip().lower() == "uniform_axis"
        else "Orientierte nicht-konsekutive Beobachtungsschnitte im CT-Volumen"
    )

    manifest = {
        "volume": str(volume_path),
        "dim": int(args.dim),
        "slice_count": int(args.slice_count),
        "thickness_vox": float(args.thickness_vox),
        "slice_geometry": str(args.slice_geometry),
        "uniform_axis": str(args.uniform_axis),
        "weights": str(weights_path),
        "model": str(args.model),
        "device": str(device),
        "oriented_slices": create_oriented_slices_figure(
            volume,
            points,
            normals,
            geometry_label,
            "Beobachtungsschnitte" if str(args.slice_geometry).strip().lower() == "uniform_axis" else "Orientierte nicht-konsekutive Schnitte",
            out_dir,
            formats,
        ),
        "sparse_volume_known_mask": create_sparse_volume_figure(
            volume,
            sparse,
            mask,
            points,
            normals,
            "cloud" if str(args.slice_geometry).strip().lower() == "uniform_axis" else "planes",
            out_dir,
            formats,
        ),
        "hard_constraint": create_hard_constraint_figure(volume, sparse, mask, raw_pred, constrained_pred, out_dir, formats),
        "num_known_voxels": int(np.count_nonzero(mask > 0.5)),
        "num_sparse_nonzero_voxels": int(np.count_nonzero(sparse > 0.0)),
        "num_hit_voxels": int(np.count_nonzero(hits > 0.0)),
    }
    manifest.update(observation_meta)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "manifest": str(manifest_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
