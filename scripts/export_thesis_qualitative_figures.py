from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from inpainting3d.utils import ensure_dir
from scripts.generate_qualitative_casebook import (
    build_dataset_sample,
    choose_focus_indices,
    load_case_mean_metrics,
    load_eval_artifacts,
    parse_models,
    pick_default_compare_label,
    pick_default_focus_label,
    resolve_model_map,
    select_cases,
    series_metrics_to_dict,
    validate_metric,
    view_slice,
)
from scripts.visualize_model_comparison import resolve_eval_files, run_prediction


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
            "figure.titlesize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def slugify(text: str) -> str:
    """
    Convert a label into a filesystem-friendly slug.
    
    Parameters
    ----------
    text : str
        Text content to render or transform.
    
    Returns
    -------
    str
        Filesystem-friendly slug.
    """
    text = text.strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._") or "unbenannt"


def extract_run_name(eval_dir: str) -> str:
    """
    Extract the run name from an artifact path.
    
    Parameters
    ----------
    eval_dir : str
        Requested evaluation dir.
    
    Returns
    -------
    str
        Resolved name or identifier.
    """
    eval_dir = os.path.normpath(eval_dir)
    base = os.path.basename(eval_dir)
    if base.lower() == "eval":
        return os.path.basename(os.path.dirname(eval_dir))
    return base


def human_archetype(name: str) -> str:
    """
    Map an internal archetype identifier to a readable label.
    
    Parameters
    ----------
    name : str
        Human-readable name or identifier.
    
    Returns
    -------
    str
        Human-readable archetype label.
    """
    mapping = {
        "best": "Best Case",
        "average": "Durchschnittsfall",
        "worst": "Worst Case",
    }
    return mapping.get(name, name)


def compute_gray_limit(*volumes: np.ndarray) -> float:
    """
    Compute grayscale display limits for a volume slice.
    
    Parameters
    ----------
    *volumes : np.ndarray
        Collection of input volumes.
    
    Returns
    -------
    float
        Computed summary values.
    """
    hi = max(float(np.quantile(vol, 0.995)) for vol in volumes)
    return max(hi, 1e-3)


def add_overlay(
    ax: plt.Axes,
    *,
    background: np.ndarray,
    overlay: np.ndarray,
    gray_vmax: float,
    overlay_vmax: float,
    cmap: str,
    symmetric: bool,
) -> Any:
    """
    Overlay a mask or highlight layer onto an axis.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis used for plotting.
    background : np.ndarray
        Background image used for an overlay.
    overlay : np.ndarray
        Overlay image or mask drawn on top of a background.
    gray_vmax : float
        Upper grayscale display limit.
    overlay_vmax : float
        Upper display limit used for the overlay.
    cmap : str
        Matplotlib colormap name used for rendering.
    symmetric : bool
        Whether overlay limits should be symmetric around zero.
    
    Returns
    -------
    None
        The overlay is drawn onto the provided axis.
    """
    ax.imshow(background.T, cmap="gray", origin="lower", vmin=0.0, vmax=gray_vmax)
    if symmetric:
        im = ax.imshow(
            overlay.T,
            cmap=cmap,
            origin="lower",
            vmin=-overlay_vmax,
            vmax=overlay_vmax,
            alpha=np.clip(np.abs(overlay) / max(overlay_vmax, 1e-6), 0.0, 1.0).T * 0.80,
        )
    else:
        im = ax.imshow(
            overlay.T,
            cmap=cmap,
            origin="lower",
            vmin=0.0,
            vmax=overlay_vmax,
            alpha=np.clip(overlay / max(overlay_vmax, 1e-6), 0.0, 1.0).T * 0.85,
        )
    ax.axis("off")
    return im


def annotate_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    """
    Annotate a figure column header.
    
    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure to save or annotate.
    title : str
        Plot or figure title.
    subtitle : str
        Supporting subtitle shown below a title.
    
    Returns
    -------
    None
        The annotation is drawn onto the provided axis.
    """
    wrapped_title = "\n".join(textwrap.wrap(title, width=76, break_long_words=False, break_on_hyphens=False))
    wrapped_subtitle = "\n".join(textwrap.wrap(subtitle, width=120, break_long_words=False, break_on_hyphens=False))
    title_lines = max(1, wrapped_title.count("\n") + 1)
    subtitle_y = 0.938 - 0.032 * (title_lines - 1)
    fig.text(0.5, 0.978, wrapped_title, ha="center", va="top", fontsize=12.4, fontweight="bold")
    fig.text(0.5, subtitle_y, wrapped_subtitle, ha="center", va="top", fontsize=9.2)


def build_metrics_subtitle(
    *,
    base_mean: Dict[str, float],
    focus_mean: Dict[str, float],
    metric: str,
) -> str:
    """
    Build a compact metric subtitle for a figure panel.
    
    Parameters
    ----------
    base_mean : Dict[str, float]
        Mean metric value of the baseline or reference model.
    focus_mean : Dict[str, float]
        Mean metric value of the focus model.
    metric : str
        Metric name used for lookup or reporting.
    
    Returns
    -------
    str
        Constructed object ready for downstream use.
    """
    base_value = float(base_mean.get(metric, np.nan))
    focus_value = float(focus_mean.get(metric, np.nan))
    rel = 100.0 * (base_value - focus_value) / max(base_value, 1e-12)
    return (
        f"{metric}: Basismodell {base_value:.5f}  |  "
        f"Nachtrainiert {focus_value:.5f}  |  "
        f"Verbesserung {rel:.1f}%"
    )


def resolve_slice_descriptor(args_payload: Dict[str, Any]) -> tuple[int | None, str, str]:
    """
    Describe a slice index relative to the shown orientation.
    
    Parameters
    ----------
    args_payload : Dict[str, Any]
        Serialized command-line or configuration payload stored with the artifact.
    
    Returns
    -------
    tuple[int | None, str, str]
        Resolved value or selection.
    """
    raw_slices = args_payload.get("slices")
    if raw_slices is not None:
        slice_count = int(raw_slices)
        return slice_count, f"{slice_count} Schnitte", f"slices_{slice_count}"

    slices_min = args_payload.get("slices_min")
    slices_max = args_payload.get("slices_max")
    if slices_min is not None and slices_max is not None and int(slices_min) == int(slices_max):
        slice_count = int(slices_min)
        return slice_count, f"{slice_count} Schnitte", f"slices_{slice_count}"

    if slices_min is not None and slices_max is not None:
        return None, f"{int(slices_min)}-{int(slices_max)} Schnitte", f"slices_{int(slices_min)}_{int(slices_max)}"

    return None, "variable Schnittzahl", "slices_variabel"


def create_large_matrix_figure(
    *,
    out_path: str,
    case_id: str,
    archetype: str,
    slice_label: str,
    eval_seed: int,
    gt: np.ndarray,
    sparse: np.ndarray,
    mask: np.ndarray,
    pred_base: np.ndarray,
    pred_focus: np.ndarray,
    base_mean: Dict[str, float],
    focus_mean: Dict[str, float],
) -> None:
    """
    Create the large matrix figure for qualitative comparisons.
    
    Parameters
    ----------
    out_path : str
        Destination path for the written artifact.
    case_id : str
        Case identifier.
    archetype : str
        Qualitative archetype label used in the figure.
    slice_label : str
        Human-readable description of the shown slice.
    eval_seed : int
        Seed used for one evaluation repetition.
    gt : np.ndarray
        Ground-truth tensor or array.
    sparse : np.ndarray
        Sparse observation tensor or array.
    mask : np.ndarray
        Binary mask that marks valid or selected voxels.
    pred_base : np.ndarray
        Prediction generated by the baseline or reference model.
    pred_focus : np.ndarray
        Prediction generated by the focus model.
    base_mean : Dict[str, float]
        Mean metric value of the baseline or reference model.
    focus_mean : Dict[str, float]
        Mean metric value of the focus model.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    indices = choose_focus_indices(gt, mask)
    gray_vmax = compute_gray_limit(gt, sparse, pred_base, pred_focus)
    base_err = np.abs(pred_base - gt).astype(np.float32)
    focus_err = np.abs(pred_focus - gt).astype(np.float32)
    err_vmax = max(
        float(np.quantile(base_err, 0.995)),
        float(np.quantile(focus_err, 0.995)),
        1e-6,
    )

    fig, axes = plt.subplots(3, 6, figsize=(16.8, 8.6), constrained_layout=False)
    columns = [
        "Sparse Volume",
        "Referenzvolumen",
        "Basismodell",
        "Nachtrainiertes Modell",
        "Absoluter Fehler\nBasismodell",
        "Absoluter Fehler\nNachtrainiert",
    ]
    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontweight="bold")

    axis_map = {"Axial": "Axial", "Koronal": "Coronal", "Sagittal": "Sagittal"}
    error_im = None
    for row, view_name in enumerate(["Axial", "Koronal", "Sagittal"]):
        key = axis_map[view_name]
        index = indices[key]
        sparse_img = view_slice(sparse, key, index)
        gt_img = view_slice(gt, key, index)
        base_img = view_slice(pred_base, key, index)
        focus_img = view_slice(pred_focus, key, index)
        base_err_img = np.abs(base_img - gt_img)
        focus_err_img = np.abs(focus_img - gt_img)

        panels = [sparse_img, gt_img, base_img, focus_img]
        for col, img in enumerate(panels):
            axes[row, col].imshow(img.T, cmap="gray", origin="lower", vmin=0.0, vmax=gray_vmax)
            axes[row, col].axis("off")

        error_im = axes[row, 4].imshow(base_err_img.T, cmap="inferno", origin="lower", vmin=0.0, vmax=err_vmax)
        axes[row, 4].axis("off")
        axes[row, 5].imshow(focus_err_img.T, cmap="inferno", origin="lower", vmin=0.0, vmax=err_vmax)
        axes[row, 5].axis("off")
        axes[row, 0].set_ylabel(f"{view_name}\nSchnitt {index}", fontweight="bold")

    annotate_header(
        fig,
        title=f"{human_archetype(archetype)}  |  Fall {case_id}  |  {slice_label}  |  Seed {eval_seed}",
        subtitle=build_metrics_subtitle(base_mean=base_mean, focus_mean=focus_mean, metric="rmse_missing"),
    )
    fig.subplots_adjust(left=0.035, right=0.925, top=0.82, bottom=0.06, wspace=0.03, hspace=0.08)
    cax = fig.add_axes([0.935, 0.17, 0.012, 0.55])
    cbar = fig.colorbar(error_im, cax=cax)
    cbar.set_label("Absoluter Fehler")
    fig.savefig(os.path.normpath(out_path), format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def create_comparison_figure(
    *,
    out_path: str,
    case_id: str,
    archetype: str,
    slice_label: str,
    eval_seed: int,
    gt: np.ndarray,
    mask: np.ndarray,
    pred_base: np.ndarray,
    pred_focus: np.ndarray,
    base_mean: Dict[str, float],
    focus_mean: Dict[str, float],
) -> None:
    """
    Create the core qualitative comparison figure.
    
    Parameters
    ----------
    out_path : str
        Destination path for the written artifact.
    case_id : str
        Case identifier.
    archetype : str
        Qualitative archetype label used in the figure.
    slice_label : str
        Human-readable description of the shown slice.
    eval_seed : int
        Seed used for one evaluation repetition.
    gt : np.ndarray
        Ground-truth tensor or array.
    mask : np.ndarray
        Binary mask that marks valid or selected voxels.
    pred_base : np.ndarray
        Prediction generated by the baseline or reference model.
    pred_focus : np.ndarray
        Prediction generated by the focus model.
    base_mean : Dict[str, float]
        Mean metric value of the baseline or reference model.
    focus_mean : Dict[str, float]
        Mean metric value of the focus model.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    indices = choose_focus_indices(gt, mask)
    improvement = np.abs(pred_base - gt) - np.abs(pred_focus - gt)
    gray_vmax = compute_gray_limit(gt, pred_base, pred_focus)
    diff_vmax = max(float(np.quantile(np.abs(improvement), 0.995)), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.9), constrained_layout=False)
    axis_map = {"Axial": "Axial", "Koronal": "Coronal", "Sagittal": "Sagittal"}
    im = None
    for col, view_name in enumerate(["Axial", "Koronal", "Sagittal"]):
        key = axis_map[view_name]
        index = indices[key]
        gt_img = view_slice(gt, key, index)
        improvement_img = view_slice(improvement, key, index)
        im = add_overlay(
            axes[col],
            background=gt_img,
            overlay=improvement_img,
            gray_vmax=gray_vmax,
            overlay_vmax=diff_vmax,
            cmap="coolwarm",
            symmetric=True,
        )
        axes[col].set_title(f"{view_name} (Schnitt {index})", fontweight="bold")

    annotate_header(
        fig,
        title=f"Fehlervergleich  |  {human_archetype(archetype)}  |  Fall {case_id}  |  {slice_label}",
        subtitle=(
            "Warme Farben: Finetuning besser. Kalte Farben: Basismodell besser.  |  "
            + build_metrics_subtitle(base_mean=base_mean, focus_mean=focus_mean, metric="mae_missing")
        ),
    )
    fig.subplots_adjust(left=0.03, right=0.90, top=0.76, bottom=0.12, wspace=0.06)
    cax = fig.add_axes([0.915, 0.21, 0.015, 0.46])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Verbesserung des absoluten Fehlers")
    fig.text(0.5, 0.03, f"Seed {eval_seed}", ha="center", va="bottom", fontsize=9)
    fig.savefig(os.path.normpath(out_path), format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _face_colors(face_vertices: np.ndarray, color: str) -> np.ndarray:
    """
    Generate face colors for volumetric surface rendering.
    
    Parameters
    ----------
    face_vertices : np.ndarray
        Vertex coordinates of mesh faces.
    color : str
        Color used for plotting or annotation.
    
    Returns
    -------
    np.ndarray
        Face colors prepared for surface rendering.
    """
    base_color = np.asarray(to_rgb(color), dtype=np.float32)
    light_dir = np.asarray([0.30, -0.45, 0.84], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir) + 1e-8
    u = face_vertices[:, 1] - face_vertices[:, 0]
    v = face_vertices[:, 2] - face_vertices[:, 0]
    normals = np.cross(u, v)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    intensity = np.clip(normals @ light_dir, 0.0, 1.0)
    shade = 0.40 + 0.60 * intensity[:, None]
    return np.clip(base_color[None, :] * shade, 0.0, 1.0)


def extract_surface_mesh(volume: np.ndarray, threshold: float) -> Any:
    """
    Extract a surface mesh from a volumetric foreground mask.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume array.
    threshold : float
        Threshold used to define the foreground region.
    
    Returns
    -------
    Any
        Extracted surface mesh.
    """
    smoothed = gaussian_filter(volume.astype(np.float32), sigma=0.8)
    vmin = float(smoothed.min())
    vmax = float(smoothed.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-6:
        return None
    level = float(np.clip(threshold, vmin + 1e-4, vmax - 1e-4))
    try:
        verts, faces, _, _ = marching_cubes(smoothed, level=level, step_size=2, allow_degenerate=False)
    except ValueError:
        return None
    verts = verts[:, [2, 1, 0]].astype(np.float32)
    faces = faces.astype(np.int32)
    if faces.shape[0] > 22000:
        step = max(1, faces.shape[0] // 22000)
        faces = faces[::step]
    return verts, faces


def compute_mesh_bounds(meshes: Sequence[Any]) -> np.ndarray:
    """
    Compute axis-aligned bounds for a mesh.
    
    Parameters
    ----------
    meshes : Sequence[Any]
        Meshes that should be rendered into the figure.
    
    Returns
    -------
    np.ndarray
        Computed summary values.
    """
    available = [mesh[0] for mesh in meshes if mesh is not None]
    if not available:
        return np.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    all_verts = np.concatenate(available, axis=0)
    mins = all_verts.min(axis=0)
    maxs = all_verts.max(axis=0)
    extent = np.maximum(maxs - mins, 1.0)
    margin = 0.08 * extent
    return np.stack([mins - margin, maxs + margin], axis=1).astype(np.float32)


def add_mesh_surface(
    ax: Any,
    *,
    mesh_data: Any,
    color: str,
    title: str,
    bounds: np.ndarray,
) -> None:
    """
    Add a triangulated mesh surface to a 3D axis.
    
    Parameters
    ----------
    ax : Any
        Matplotlib axis used for plotting.
    mesh_data : Any
        Mesh vertices and faces passed to the renderer.
    color : str
        Color used for plotting or annotation.
    title : str
        Plot or figure title.
    bounds : np.ndarray
        Spatial bounds used to frame a rendered scene.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    if mesh_data is None:
        ax.set_title(f"{title}\n(keine stabile Oberfläche)")
        ax.set_axis_off()
        return

    verts, faces = mesh_data
    face_vertices = verts[faces]
    mesh = Poly3DCollection(
        face_vertices,
        facecolors=_face_colors(face_vertices, color),
        edgecolor="none",
        alpha=0.97,
    )
    mesh.set_rasterized(True)
    ax.add_collection3d(mesh)

    ax.set_xlim(float(bounds[0, 0]), float(bounds[0, 1]))
    ax.set_ylim(float(bounds[1, 0]), float(bounds[1, 1]))
    ax.set_zlim(float(bounds[2, 0]), float(bounds[2, 1]))
    ax.set_box_aspect(
        (
            float(bounds[0, 1] - bounds[0, 0]),
            float(bounds[1, 1] - bounds[1, 0]),
            float(bounds[2, 1] - bounds[2, 0]),
        )
    )
    ax.view_init(elev=18, azim=42)
    ax.set_proj_type("persp", focal_length=0.9)
    ax.set_title(title, fontweight="bold")
    ax.set_axis_off()


def create_volume_render_figure(
    *,
    out_path: str,
    case_id: str,
    archetype: str,
    slice_label: str,
    eval_seed: int,
    sparse: np.ndarray,
    gt: np.ndarray,
    pred_base: np.ndarray,
    pred_focus: np.ndarray,
    base_mean: Dict[str, float],
    focus_mean: Dict[str, float],
    threshold: float,
) -> None:
    """
    Create a qualitative 3D volume-rendering figure.
    
    Parameters
    ----------
    out_path : str
        Destination path for the written artifact.
    case_id : str
        Case identifier.
    archetype : str
        Qualitative archetype label used in the figure.
    slice_label : str
        Human-readable description of the shown slice.
    eval_seed : int
        Seed used for one evaluation repetition.
    sparse : np.ndarray
        Sparse observation tensor or array.
    gt : np.ndarray
        Ground-truth tensor or array.
    pred_base : np.ndarray
        Prediction generated by the baseline or reference model.
    pred_focus : np.ndarray
        Prediction generated by the focus model.
    base_mean : Dict[str, float]
        Mean metric value of the baseline or reference model.
    focus_mean : Dict[str, float]
        Mean metric value of the focus model.
    threshold : float
        Threshold used to define the foreground region.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    positive = gt[gt > max(threshold, 1e-3)]
    if positive.size:
        threshold = max(float(threshold), float(np.quantile(positive, 0.35)))
    sparse_positive = sparse[sparse > 1e-3]
    sparse_threshold = threshold
    if sparse_positive.size:
        sparse_threshold = max(0.06, min(float(threshold), float(np.quantile(sparse_positive, 0.20))))

    sparse_mesh = extract_surface_mesh(sparse, sparse_threshold)
    gt_mesh = extract_surface_mesh(gt, threshold)
    base_mesh = extract_surface_mesh(pred_base, threshold)
    focus_mesh = extract_surface_mesh(pred_focus, threshold)
    bounds = compute_mesh_bounds([sparse_mesh, gt_mesh, base_mesh, focus_mesh])

    fig = plt.figure(figsize=(18.8, 5.2))
    axes = [
        fig.add_subplot(1, 4, 1, projection="3d"),
        fig.add_subplot(1, 4, 2, projection="3d"),
        fig.add_subplot(1, 4, 3, projection="3d"),
        fig.add_subplot(1, 4, 4, projection="3d"),
    ]
    render_color = "#b86a1b"
    add_mesh_surface(axes[0], mesh_data=sparse_mesh, color=render_color, title="Sparse Volume", bounds=bounds)
    add_mesh_surface(axes[1], mesh_data=gt_mesh, color=render_color, title="Referenzvolumen", bounds=bounds)
    add_mesh_surface(axes[2], mesh_data=base_mesh, color=render_color, title="Basismodell", bounds=bounds)
    add_mesh_surface(axes[3], mesh_data=focus_mesh, color=render_color, title="Nachtrainiertes Modell", bounds=bounds)
    annotate_header(
        fig,
        title=f"3D-Volumenrendering  |  {human_archetype(archetype)}  |  Fall {case_id}  |  {slice_label}",
        subtitle=f"Isowertdarstellung  |  Sparse-Schwelle {sparse_threshold:.3f}  |  Volumenschwelle {threshold:.3f}  |  Seed {eval_seed}",
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.74, bottom=0.04, wspace=0.01)
    fig.savefig(os.path.normpath(out_path), format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    """
    Write a JSON document to disk.
    
    Parameters
    ----------
    path : str
        Filesystem path to the input artifact.
    payload : Dict[str, Any]
        Structured data that will be serialized as JSON.
    
    Returns
    -------
    None
        The JSON artifact is written to disk.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """
    Execute the command-line entry point for this script.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--select_label", type=str, default="")
    ap.add_argument("--compare_label", type=str, default="")
    ap.add_argument("--metric", type=str, default="rmse_missing")
    ap.add_argument("--selector", type=str, default="mean", choices=["mean", "median"])
    ap.add_argument("--surface_threshold", type=float, default=0.10)
    ap.add_argument("--no_hard_constraint", dest="hard_constraint", action="store_false")
    ap.set_defaults(hard_constraint=True)
    args = ap.parse_args()

    configure_matplotlib()

    eval_dir = os.path.normpath(os.path.abspath(args.eval_dir))
    frame, repro, summary = load_eval_artifacts(eval_dir)
    validate_metric(frame, args.metric)
    models = parse_models(repro, summary)
    model_map = resolve_model_map(models)

    focus_label = args.select_label.strip() or pick_default_focus_label(models)
    compare_label = args.compare_label.strip() or pick_default_compare_label(models, focus_label)
    if focus_label not in model_map:
        raise RuntimeError(f"Unknown focus label {focus_label!r}.")
    if compare_label and compare_label not in model_map:
        raise RuntimeError(f"Unknown comparison label {compare_label!r}.")
    if not compare_label:
        raise RuntimeError("A comparison label is required for the thesis-quality exports.")

    out_dir = os.path.normpath(os.path.abspath(args.out)) if args.out else os.path.join(eval_dir, "thesis_qualitative_pdfs")
    ensure_dir(out_dir)

    args_payload = repro.get("args")
    if not isinstance(args_payload, dict):
        raise RuntimeError("Evaluation reproducibility.json does not contain an 'args' object.")

    selected = select_cases(
        frame,
        focus_label=focus_label,
        compare_label=compare_label,
        metric=str(args.metric),
        selector=str(args.selector),
        lower_is_better=True,
    )

    files = resolve_eval_files(
        data_glob=str(args_payload.get("data", "")),
        split_file=str(args_payload.get("split_file", "")),
        split=str(args_payload.get("split", "test")),
    )
    run_name = slugify(extract_run_name(eval_dir))
    slice_count, slice_label, slice_slug = resolve_slice_descriptor(args_payload)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outputs: List[Dict[str, Any]] = []
    for item in selected:
        x, y, meta = build_dataset_sample(
            files=files,
            case_id=item.case_id,
            seed=item.eval_seed,
            args_payload=args_payload,
        )
        gt = y[0].numpy().astype(np.float32)
        sparse = x[0].numpy().astype(np.float32)
        mask = x[1].numpy().astype(np.float32)

        focus_model = model_map[focus_label]
        compare_model = model_map[compare_label]
        pred_focus = run_prediction(
            x,
            weights=focus_model.weights,
            model_kind=focus_model.kind,
            init_feat=int(args_payload.get("init_feat", 32)),
            hard_constraint=bool(args.hard_constraint),
            device=device,
        )
        pred_compare = run_prediction(
            x,
            weights=compare_model.weights,
            model_kind=compare_model.kind,
            init_feat=int(args_payload.get("init_feat", 32)),
            hard_constraint=bool(args.hard_constraint),
            device=device,
        )

        focus_mean = load_case_mean_metrics(frame, focus_label, item.case_id)
        compare_mean = load_case_mean_metrics(frame, compare_label, item.case_id)
        focus_row = frame[
            (frame["label"] == focus_label)
            & (frame["case_id"] == item.case_id)
            & (frame["eval_seed"].astype(int) == item.eval_seed)
        ]
        compare_row = frame[
            (frame["label"] == compare_label)
            & (frame["case_id"] == item.case_id)
            & (frame["eval_seed"].astype(int) == item.eval_seed)
        ]
        focus_row_metrics = series_metrics_to_dict(focus_row.iloc[0]) if not focus_row.empty else {}
        compare_row_metrics = series_metrics_to_dict(compare_row.iloc[0]) if not compare_row.empty else {}

        prefix = "__".join(
            [
                run_name,
                slice_slug,
                slugify(item.archetype),
                slugify(item.case_id),
                f"seed_{item.eval_seed}",
                slugify(compare_label),
                "vs",
                slugify(focus_label),
            ]
        )
        matrix_path = os.path.join(out_dir, prefix + "__grosse_matrix.pdf")
        compare_path = os.path.join(out_dir, prefix + "__vergleich.pdf")
        render_path = os.path.join(out_dir, prefix + "__volumenrendering.pdf")

        create_large_matrix_figure(
            out_path=matrix_path,
            case_id=item.case_id,
            archetype=item.archetype,
            slice_label=slice_label,
            eval_seed=item.eval_seed,
            gt=gt,
            sparse=sparse,
            mask=mask,
            pred_base=pred_compare,
            pred_focus=pred_focus,
            base_mean=compare_mean,
            focus_mean=focus_mean,
        )
        create_comparison_figure(
            out_path=compare_path,
            case_id=item.case_id,
            archetype=item.archetype,
            slice_label=slice_label,
            eval_seed=item.eval_seed,
            gt=gt,
            mask=mask,
            pred_base=pred_compare,
            pred_focus=pred_focus,
            base_mean=compare_mean,
            focus_mean=focus_mean,
        )
        create_volume_render_figure(
            out_path=render_path,
            case_id=item.case_id,
            archetype=item.archetype,
            slice_label=slice_label,
            eval_seed=item.eval_seed,
            sparse=sparse,
            gt=gt,
            pred_base=pred_compare,
            pred_focus=pred_focus,
            base_mean=compare_mean,
            focus_mean=focus_mean,
            threshold=float(args.surface_threshold),
        )

        outputs.append(
            {
                "archetype": item.archetype,
                "case_id": item.case_id,
                "eval_seed": int(item.eval_seed),
                "path": item.path,
                "matrix_pdf": os.path.normpath(matrix_path),
                "comparison_pdf": os.path.normpath(compare_path),
                "volume_pdf": os.path.normpath(render_path),
                "focus_label": focus_label,
                "compare_label": compare_label,
                "focus_case_mean": focus_mean,
                "compare_case_mean": compare_mean,
                "focus_row_metrics": focus_row_metrics,
                "compare_row_metrics": compare_row_metrics,
                "slice_count": slice_count,
                "slice_label": slice_label,
                "meta": {
                    "case_id": meta.get("case_id"),
                    "path": meta.get("path"),
                    "slice_geometry": meta.get("slice_geometry"),
                    "dataset_index": meta.get("dataset_index"),
                },
            }
        )

    manifest = {
        "eval_dir": eval_dir,
        "run_name": run_name,
        "focus_label": focus_label,
        "compare_label": compare_label,
        "metric": str(args.metric),
        "selector": str(args.selector),
        "surface_threshold": float(args.surface_threshold),
        "slice_count": slice_count,
        "slice_label": slice_label,
        "outputs": outputs,
    }
    save_json(os.path.join(out_dir, "thesis_qualitative_manifest.json"), manifest)
    print(json.dumps({"out": out_dir, "num_cases": len(outputs), "outputs": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
