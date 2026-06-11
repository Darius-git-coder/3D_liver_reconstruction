from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy.ndimage import binary_erosion

from inpainting3d.utils import ensure_dir
from scripts.visualize_model_comparison import load_case_sample, resolve_eval_files, run_prediction, to_metrics


@dataclass
class ModelEntry:
    label: str
    kind: str
    weights: str


@dataclass
class SelectedCase:
    archetype: str
    case_id: str
    path: str
    eval_seed: int
    focus_metric_mean: float
    focus_metric_row: float
    compare_metric_mean: float | None
    compare_metric_row: float | None


def normalize_label(label: str) -> str:
    return str(label).strip()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path!r}.")
    return payload


def load_eval_artifacts(eval_dir: str) -> tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    metrics_path = os.path.join(eval_dir, "metrics_per_case.csv")
    repro_path = os.path.join(eval_dir, "reproducibility.json")
    summary_path = os.path.join(eval_dir, "metrics_summary.json")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics_per_case.csv not found in {eval_dir!r}.")
    if not os.path.exists(repro_path):
        raise FileNotFoundError(f"reproducibility.json not found in {eval_dir!r}.")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"metrics_summary.json not found in {eval_dir!r}.")

    frame = pd.read_csv(metrics_path)
    repro = load_json(repro_path)
    summary = load_json(summary_path)
    return frame, repro, summary


def parse_models(repro: Dict[str, Any], summary: Dict[str, Any]) -> List[ModelEntry]:
    models_payload = None
    extra = repro.get("extra")
    if isinstance(extra, dict):
        models_payload = extra.get("models")
    if not isinstance(models_payload, list):
        models_payload = summary.get("models")
    if not isinstance(models_payload, list) or not models_payload:
        raise RuntimeError("No model metadata found in evaluation artifacts.")

    models: List[ModelEntry] = []
    for item in models_payload:
        if not isinstance(item, dict):
            continue
        label = normalize_label(item.get("label", ""))
        kind = str(item.get("kind", item.get("model", ""))).strip()
        weights = str(item.get("weights", "")).strip()
        if label and kind and weights:
            models.append(ModelEntry(label=label, kind=kind, weights=weights))
    if not models:
        raise RuntimeError("No valid model entries found in evaluation artifacts.")
    return models


def pick_default_focus_label(models: Sequence[ModelEntry]) -> str:
    return models[-1].label


def pick_default_compare_label(models: Sequence[ModelEntry], focus_label: str) -> str | None:
    for model in models:
        if model.label != focus_label:
            return model.label
    return None


def validate_metric(frame: pd.DataFrame, metric: str) -> None:
    if metric not in frame.columns:
        known = ", ".join(str(col) for col in frame.columns)
        raise RuntimeError(f"Metric {metric!r} not present in metrics_per_case.csv. Available: {known}")


def select_representative_row(case_rows: pd.DataFrame, metric: str, target_value: float) -> pd.Series:
    diffs = (case_rows[metric].astype(float) - float(target_value)).abs()
    order = np.argsort(diffs.to_numpy(), kind="stable")
    return case_rows.iloc[int(order[0])]


def select_cases(
    frame: pd.DataFrame,
    *,
    focus_label: str,
    compare_label: str | None,
    metric: str,
    selector: str,
    lower_is_better: bool,
) -> List[SelectedCase]:
    label_frame = frame[frame["label"] == focus_label].copy()
    if label_frame.empty:
        raise RuntimeError(f"No rows found for focus label {focus_label!r}.")

    aggregated = (
        label_frame.groupby("case_id", as_index=False)
        .agg(
            metric_mean=(metric, "mean"),
            path=("path", "first"),
        )
        .sort_values("metric_mean", ascending=bool(lower_is_better), kind="stable")
        .reset_index(drop=True)
    )
    if aggregated.empty:
        raise RuntimeError("Could not aggregate case metrics.")

    best_idx = 0 if lower_is_better else len(aggregated) - 1
    worst_idx = len(aggregated) - 1 if lower_is_better else 0
    target_stat = float(aggregated["metric_mean"].mean()) if selector == "mean" else float(aggregated["metric_mean"].median())
    aggregated["distance_to_center"] = np.abs(aggregated["metric_mean"].astype(float) - target_stat)

    chosen_case_ids: List[str] = []
    archetype_rows: List[tuple[str, pd.Series]] = []

    best_row = aggregated.iloc[best_idx]
    chosen_case_ids.append(str(best_row["case_id"]))
    archetype_rows.append(("best", best_row))

    center_candidates = aggregated.sort_values(["distance_to_center", "metric_mean"], ascending=[True, lower_is_better], kind="stable")
    center_row = None
    for _, candidate in center_candidates.iterrows():
        case_id = str(candidate["case_id"])
        if case_id not in chosen_case_ids:
            center_row = candidate
            break
    if center_row is None:
        center_row = center_candidates.iloc[0]
    chosen_case_ids.append(str(center_row["case_id"]))
    archetype_rows.append(("average", center_row))

    worst_row = aggregated.iloc[worst_idx]
    if str(worst_row["case_id"]) in chosen_case_ids:
        scan_frame = aggregated.iloc[::-1] if lower_is_better else aggregated
        for _, candidate in scan_frame.iterrows():
            case_id = str(candidate["case_id"])
            if case_id not in chosen_case_ids:
                worst_row = candidate
                break
    archetype_rows.append(("worst", worst_row))

    compare_frame = frame[frame["label"] == compare_label].copy() if compare_label else None
    selected: List[SelectedCase] = []
    for archetype, row in archetype_rows:
        case_id = str(row["case_id"])
        case_mean = float(row["metric_mean"])
        case_rows = label_frame[label_frame["case_id"] == case_id].copy()
        rep = select_representative_row(case_rows, metric, case_mean)
        eval_seed = int(rep["eval_seed"])
        focus_metric_row = float(rep[metric])

        compare_metric_mean = None
        compare_metric_row = None
        if compare_frame is not None and not compare_frame.empty:
            compare_case_rows = compare_frame[compare_frame["case_id"] == case_id].copy()
            if not compare_case_rows.empty:
                compare_metric_mean = float(compare_case_rows[metric].mean())
                same_seed = compare_case_rows[compare_case_rows["eval_seed"].astype(int) == eval_seed]
                if same_seed.empty:
                    same_seed = compare_case_rows
                compare_rep = select_representative_row(same_seed, metric, float(compare_metric_mean))
                compare_metric_row = float(compare_rep[metric])

        selected.append(
            SelectedCase(
                archetype=archetype,
                case_id=case_id,
                path=str(row["path"]),
                eval_seed=eval_seed,
                focus_metric_mean=case_mean,
                focus_metric_row=focus_metric_row,
                compare_metric_mean=compare_metric_mean,
                compare_metric_row=compare_metric_row,
            )
        )
    return selected


def resolve_model_map(models: Sequence[ModelEntry]) -> Dict[str, ModelEntry]:
    return {model.label: model for model in models}


def build_dataset_sample(
    *,
    files: List[str],
    case_id: str,
    seed: int,
    args_payload: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    slices = args_payload.get("slices")
    slices_min = int(args_payload.get("slices_min", 4))
    slices_max = int(args_payload.get("slices_max", 32))
    if slices is not None:
        slices_min = int(slices)
        slices_max = int(slices)

    return load_case_sample(
        files=files,
        case_id=case_id,
        dim=int(args_payload.get("dim", 64)),
        init_feat=int(args_payload.get("init_feat", 32)),
        slices_min=slices_min,
        slices_max=slices_max,
        slice_sampling=str(args_payload.get("slice_sampling", "uniform")),
        slice_mean=args_payload.get("slice_mean"),
        slice_std=args_payload.get("slice_std"),
        slice_geometry=str(args_payload.get("slice_geometry", "random")),
        slice_axis=args_payload.get("slice_axis"),
        slice_axis_jitter_deg=float(args_payload.get("slice_axis_jitter_deg", 25.0)),
        slice_fan_half_angle_deg=float(args_payload.get("slice_fan_half_angle_deg", 35.0)),
        slice_elevation_jitter_deg=float(args_payload.get("slice_elevation_jitter_deg", 6.0)),
        slice_sweep_jitter_deg=float(args_payload.get("slice_sweep_jitter_deg", 2.5)),
        probe_pos_sigma_vox=float(args_payload.get("probe_pos_sigma_vox", 1.0)),
        probe_depth_sigma_vox=float(args_payload.get("probe_depth_sigma_vox", 6.0)),
        probe_tilt_sigma=float(args_payload.get("probe_tilt_sigma", 0.08)),
        thickness_vox=float(args_payload.get("thickness_vox", 1.0)),
        seed=seed,
    )


def choose_focus_indices(gt: np.ndarray, mask: np.ndarray) -> Dict[str, int]:
    foreground = gt > 1e-3
    missing = mask < 0.5
    focus = foreground & missing
    coords = np.argwhere(focus)
    if coords.size == 0:
        coords = np.argwhere(foreground)
    if coords.size == 0:
        coords = np.argwhere(np.ones_like(gt, dtype=bool))
    center = np.round(coords.mean(axis=0)).astype(int)
    return {
        "Axial": int(np.clip(center[0], 0, gt.shape[0] - 1)),
        "Coronal": int(np.clip(center[1], 0, gt.shape[1] - 1)),
        "Sagittal": int(np.clip(center[2], 0, gt.shape[2] - 1)),
    }


def view_slice(volume: np.ndarray, view_name: str, index: int) -> np.ndarray:
    if view_name == "Axial":
        return volume[index, :, :]
    if view_name == "Coronal":
        return volume[:, index, :]
    if view_name == "Sagittal":
        return volume[:, :, index]
    raise ValueError(f"Unknown view {view_name!r}.")


def compute_display_limits(*volumes: np.ndarray) -> tuple[float, float]:
    hi = max(float(np.quantile(vol, 0.995)) for vol in volumes)
    return 0.0, max(hi, 1e-3)


def overlay_signed_error(
    ax: plt.Axes,
    *,
    gt_img: np.ndarray,
    pred_img: np.ndarray,
    vmax_gray: float,
    vmax_err: float,
) -> None:
    diff = pred_img - gt_img
    alpha = np.clip(np.abs(diff) / max(vmax_err, 1e-6), 0.0, 1.0) ** 0.85
    alpha = 0.85 * alpha
    ax.imshow(gt_img.T, cmap="bone", origin="lower", vmin=0.0, vmax=vmax_gray)
    ax.imshow(diff.T, cmap="coolwarm", origin="lower", vmin=-vmax_err, vmax=vmax_err, alpha=alpha.T)
    ax.axis("off")


def format_metric_pair(name: str, value_a: float | None, value_b: float | None) -> str:
    if value_a is None and value_b is None:
        return ""
    if value_b is None:
        return f"{name}: {value_a:.5f}"
    delta = value_b - value_a
    return f"{name}: {value_a:.5f} -> {value_b:.5f} (delta {delta:+.5f})"


def make_surface_points(volume: np.ndarray, threshold: float, max_points: int) -> np.ndarray:
    occupancy = volume >= threshold
    if not np.any(occupancy):
        return np.zeros((0, 3), dtype=np.float32)
    eroded = binary_erosion(occupancy, iterations=1, border_value=0)
    boundary = occupancy & ~eroded
    coords = np.argwhere(boundary)
    if coords.shape[0] > max_points:
        step = max(1, coords.shape[0] // max_points)
        coords = coords[::step]
    return coords.astype(np.float32)


def plot_surface(ax: Any, points: np.ndarray, title: str, color: str) -> None:
    if points.size == 0:
        ax.set_title(f"{title}\n(no surface voxels)")
        ax.set_axis_off()
        return
    ax.scatter(points[:, 2], points[:, 1], points[:, 0], s=1.8, c=color, alpha=0.22, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("W")
    ax.set_ylabel("H")
    ax.set_zlabel("D")
    ax.view_init(elev=24, azim=38)


def make_error_cloud(error_volume: np.ndarray, q: float, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    threshold = float(np.quantile(np.abs(error_volume), q))
    coords = np.argwhere(np.abs(error_volume) >= threshold)
    if coords.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    values = error_volume[np.abs(error_volume) >= threshold].astype(np.float32)
    if coords.shape[0] > max_points:
        step = max(1, coords.shape[0] // max_points)
        coords = coords[::step]
        values = values[::step]
    return coords.astype(np.float32), values


def plot_signed_cloud(ax: Any, coords: np.ndarray, values: np.ndarray, title: str, vmax: float) -> None:
    if coords.size == 0:
        ax.set_title(f"{title}\n(no highlighted voxels)")
        ax.set_axis_off()
        return
    ax.scatter(
        coords[:, 2],
        coords[:, 1],
        coords[:, 0],
        c=values,
        cmap="coolwarm",
        s=2.8,
        alpha=0.34,
        linewidths=0,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("W")
    ax.set_ylabel("H")
    ax.set_zlabel("D")
    ax.view_init(elev=26, azim=42)


def create_cover_page(
    pdf: PdfPages,
    *,
    eval_dir: str,
    focus_label: str,
    compare_label: str | None,
    metric: str,
    selected: Sequence[SelectedCase],
    args_payload: Dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    lines = [
        "Qualitative Casebook",
        "",
        f"Evaluation folder: {os.path.normpath(eval_dir)}",
        f"Focus model: {focus_label}",
        f"Comparison model: {compare_label or 'none'}",
        f"Ranking metric: {metric}",
        f"Slice geometry: {args_payload.get('slice_geometry', 'n/a')}",
        f"Fixed slices: {args_payload.get('slices', 'variable')}",
        "",
        "Case selection:",
    ]
    for item in selected:
        compare_text = ""
        if item.compare_metric_mean is not None:
            compare_text = f", compare mean={item.compare_metric_mean:.5f}"
        lines.append(
            f"  - {item.archetype:7s} case={item.case_id} seed={item.eval_seed} "
            f"focus mean={item.focus_metric_mean:.5f}{compare_text}"
        )
    lines.extend(
        [
            "",
            "Layout:",
            "  - page A: orthogonal 4-column comparison for the baseline model",
            "  - page B: orthogonal 4-column comparison for the finetuned model",
            "  - page C: 3D surfaces and signed error clouds",
        ]
    )
    fig.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_orthogonal_page(
    pdf: PdfPages,
    *,
    archetype: str,
    case_id: str,
    eval_seed: int,
    gt: np.ndarray,
    sparse: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    metrics_row: Dict[str, float],
    metrics_mean: Dict[str, float] | None,
    model_label: str,
    role_label: str,
) -> None:
    indices = choose_focus_indices(gt, mask)
    vmin, vmax = compute_display_limits(gt, sparse, pred)
    vmax_err = max(float(np.quantile(np.abs(pred - gt), 0.995)), 1e-6)

    fig, axes = plt.subplots(3, 4, figsize=(15.5, 11.2))
    column_titles = ["Sparse Input", "Ground Truth", "Prediction", "Error Overlay"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=12)

    for row, view_name in enumerate(["Axial", "Coronal", "Sagittal"]):
        index = indices[view_name]
        sparse_img = view_slice(sparse, view_name, index)
        gt_img = view_slice(gt, view_name, index)
        pred_img = view_slice(pred, view_name, index)

        axes[row, 0].imshow(sparse_img.T, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(gt_img.T, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 2].imshow(pred_img.T, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
        overlay_signed_error(
            axes[row, 3],
            gt_img=gt_img,
            pred_img=pred_img,
            vmax_gray=vmax,
            vmax_err=vmax_err,
        )
        for col in range(4):
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"{view_name}\nidx={index}", fontsize=11)

    title = (
        f"{archetype.upper()} CASE  |  {case_id}  |  seed {eval_seed}  |  "
        f"{role_label}: {model_label}"
    )
    subtitle_lines = [
        format_metric_pair("rmse_missing (row)", metrics_row.get("rmse_missing"), None),
        format_metric_pair("mae_missing (row)", metrics_row.get("mae_missing"), None),
        format_metric_pair("ssim_all (row)", metrics_row.get("ssim_all"), None),
    ]
    if metrics_mean:
        subtitle_lines.extend(
            [
                format_metric_pair("rmse_missing (case mean)", metrics_mean.get("rmse_missing"), None),
                format_metric_pair("ssim_all (case mean)", metrics_mean.get("ssim_all"), None),
            ]
        )
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.text(0.5, 0.943, "  |  ".join(item for item in subtitle_lines if item), ha="center", va="top", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_3d_page(
    pdf: PdfPages,
    *,
    archetype: str,
    case_id: str,
    eval_seed: int,
    gt: np.ndarray,
    pred_compare: np.ndarray | None,
    pred_focus: np.ndarray,
    compare_label: str | None,
    focus_label: str,
    surface_threshold: float,
) -> None:
    gt_points = make_surface_points(gt, surface_threshold, max_points=14000)
    focus_points = make_surface_points(pred_focus, surface_threshold, max_points=14000)
    compare_points = make_surface_points(pred_compare, surface_threshold, max_points=14000) if pred_compare is not None else np.zeros((0, 3), dtype=np.float32)

    focus_diff = (pred_focus - gt).astype(np.float32)
    focus_coords, focus_vals = make_error_cloud(focus_diff, q=0.97, max_points=16000)
    compare_coords = np.zeros((0, 3), dtype=np.float32)
    compare_vals = np.zeros((0,), dtype=np.float32)
    improvement_coords = np.zeros((0, 3), dtype=np.float32)
    improvement_vals = np.zeros((0,), dtype=np.float32)
    if pred_compare is not None:
        compare_diff = (pred_compare - gt).astype(np.float32)
        compare_coords, compare_vals = make_error_cloud(compare_diff, q=0.97, max_points=16000)
        improvement = (np.abs(pred_compare - gt) - np.abs(pred_focus - gt)).astype(np.float32)
        improvement_coords, improvement_vals = make_error_cloud(improvement, q=0.97, max_points=16000)

    vmax_signed = max(
        float(np.quantile(np.abs(focus_vals), 0.995)) if focus_vals.size else 0.0,
        float(np.quantile(np.abs(compare_vals), 0.995)) if compare_vals.size else 0.0,
        float(np.quantile(np.abs(improvement_vals), 0.995)) if improvement_vals.size else 0.0,
        1e-6,
    )

    fig = plt.figure(figsize=(15.5, 10.5))
    gs = GridSpec(2, 3, figure=fig)
    axes = [
        fig.add_subplot(gs[0, 0], projection="3d"),
        fig.add_subplot(gs[0, 1], projection="3d"),
        fig.add_subplot(gs[0, 2], projection="3d"),
        fig.add_subplot(gs[1, 0], projection="3d"),
        fig.add_subplot(gs[1, 1], projection="3d"),
        fig.add_subplot(gs[1, 2], projection="3d"),
    ]

    plot_surface(axes[0], gt_points, "GT surface", "#7a0019")
    plot_surface(axes[1], compare_points, (compare_label or "Reference") + " surface", "#005f73")
    plot_surface(axes[2], focus_points, focus_label + " surface", "#ee9b00")

    plot_signed_cloud(axes[3], compare_coords, compare_vals, (compare_label or "Reference") + " signed error", vmax_signed)
    plot_signed_cloud(axes[4], focus_coords, focus_vals, focus_label + " signed error", vmax_signed)
    plot_signed_cloud(axes[5], improvement_coords, improvement_vals, "abs. error improvement", vmax_signed)

    fig.suptitle(
        f"{archetype.upper()} CASE  |  {case_id}  |  seed {eval_seed}  |  3D qualitative comparison",
        fontsize=16,
        y=0.98,
    )
    fig.text(
        0.5,
        0.945,
        "Blue/red error clouds show signed intensity deviation. Positive improvement means the focus model reduces absolute error.",
        ha="center",
        va="top",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def metrics_to_float_dict(metrics: Dict[str, float]) -> Dict[str, float]:
    return {str(key): float(value) for key, value in metrics.items()}


def load_case_mean_metrics(frame: pd.DataFrame, label: str, case_id: str) -> Dict[str, float]:
    subset = frame[(frame["label"] == label) & (frame["case_id"] == case_id)].copy()
    metric_columns = [col for col in subset.columns if col.startswith(("mae_", "rmse_", "psnr_", "ssim_"))]
    if subset.empty or not metric_columns:
        return {}
    means = subset[metric_columns].mean(numeric_only=True)
    return {str(col): float(means[col]) for col in metric_columns}


def series_metrics_to_dict(row: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in row.items():
        if str(key).startswith(("mae_", "rmse_", "psnr_", "ssim_")):
            out[str(key)] = float(value)
    return out


def main() -> None:
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

    eval_dir = os.path.normpath(os.path.abspath(args.eval_dir))
    frame, repro, summary = load_eval_artifacts(eval_dir)
    validate_metric(frame, args.metric)
    models = parse_models(repro, summary)
    model_map = resolve_model_map(models)

    focus_label = normalize_label(args.select_label) or pick_default_focus_label(models)
    if focus_label not in model_map:
        known = ", ".join(model_map.keys())
        raise RuntimeError(f"Unknown focus label {focus_label!r}. Known labels: {known}")

    compare_label = normalize_label(args.compare_label) or pick_default_compare_label(models, focus_label)
    if compare_label and compare_label not in model_map:
        known = ", ".join(model_map.keys())
        raise RuntimeError(f"Unknown comparison label {compare_label!r}. Known labels: {known}")

    out_dir = os.path.normpath(os.path.abspath(args.out)) if args.out else os.path.join(eval_dir, "qualitative_casebook")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdf_path = os.path.join(out_dir, "qualitative_casebook.pdf")
    selected_manifest: List[Dict[str, Any]] = []

    with PdfPages(pdf_path) as pdf:
        create_cover_page(
            pdf,
            eval_dir=eval_dir,
            focus_label=focus_label,
            compare_label=compare_label,
            metric=str(args.metric),
            selected=selected,
            args_payload=args_payload,
        )

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
            focus_pred = run_prediction(
                x,
                weights=focus_model.weights,
                model_kind=focus_model.kind,
                init_feat=int(args_payload.get("init_feat", 32)),
                hard_constraint=bool(args.hard_constraint),
                device=device,
            )
            focus_row = frame[
                (frame["label"] == focus_label)
                & (frame["case_id"] == item.case_id)
                & (frame["eval_seed"].astype(int) == item.eval_seed)
            ]
            if focus_row.empty:
                raise RuntimeError(f"No matching metrics row found for case {item.case_id!r}, seed {item.eval_seed}.")
            focus_row_metrics = series_metrics_to_dict(focus_row.iloc[0])
            focus_mean_metrics = load_case_mean_metrics(frame, focus_label, item.case_id)

            compare_pred = None
            compare_row_metrics: Dict[str, float] | None = None
            compare_mean_metrics: Dict[str, float] | None = None
            if compare_label:
                compare_model = model_map[compare_label]
                compare_pred = run_prediction(
                    x,
                    weights=compare_model.weights,
                    model_kind=compare_model.kind,
                    init_feat=int(args_payload.get("init_feat", 32)),
                    hard_constraint=bool(args.hard_constraint),
                    device=device,
                )
                compare_row = frame[
                    (frame["label"] == compare_label)
                    & (frame["case_id"] == item.case_id)
                    & (frame["eval_seed"].astype(int) == item.eval_seed)
                ]
                if not compare_row.empty:
                    compare_row_metrics = series_metrics_to_dict(compare_row.iloc[0])
                compare_mean_metrics = load_case_mean_metrics(frame, compare_label, item.case_id)

            if compare_pred is not None:
                create_orthogonal_page(
                    pdf,
                    archetype=item.archetype,
                case_id=item.case_id,
                eval_seed=item.eval_seed,
                gt=gt,
                sparse=sparse,
                mask=mask,
                pred=compare_pred,
                metrics_row=compare_row_metrics or metrics_to_float_dict(to_metrics(compare_pred, gt, mask)),
                metrics_mean=compare_mean_metrics,
                model_label=compare_label or "reference",
                    role_label="baseline/reference",
                )

            create_orthogonal_page(
                pdf,
                archetype=item.archetype,
                case_id=item.case_id,
                eval_seed=item.eval_seed,
                gt=gt,
                sparse=sparse,
                mask=mask,
                pred=focus_pred,
                metrics_row=focus_row_metrics or metrics_to_float_dict(to_metrics(focus_pred, gt, mask)),
                metrics_mean=focus_mean_metrics,
                model_label=focus_label,
                role_label="focus model",
            )

            create_3d_page(
                pdf,
                archetype=item.archetype,
                case_id=item.case_id,
                eval_seed=item.eval_seed,
                gt=gt,
                pred_compare=compare_pred,
                pred_focus=focus_pred,
                compare_label=compare_label,
                focus_label=focus_label,
                surface_threshold=float(args.surface_threshold),
            )

            selected_manifest.append(
                {
                    "archetype": item.archetype,
                    "case_id": item.case_id,
                    "path": item.path,
                    "eval_seed": int(item.eval_seed),
                    "focus_label": focus_label,
                    "compare_label": compare_label,
                    "focus_metric_mean": float(item.focus_metric_mean),
                    "focus_metric_row": float(item.focus_metric_row),
                    "compare_metric_mean": None if item.compare_metric_mean is None else float(item.compare_metric_mean),
                    "compare_metric_row": None if item.compare_metric_row is None else float(item.compare_metric_row),
                    "meta": {key: value for key, value in meta.items() if key in {"case_id", "path", "slice_geometry", "dataset_index"}},
                }
            )

    payload = {
        "eval_dir": eval_dir,
        "pdf": os.path.normpath(pdf_path),
        "focus_label": focus_label,
        "compare_label": compare_label,
        "metric": str(args.metric),
        "selector": str(args.selector),
        "surface_threshold": float(args.surface_threshold),
        "selected_cases": selected_manifest,
    }
    with open(os.path.join(out_dir, "selected_cases.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps({"out": out_dir, "pdf": pdf_path, "selected_cases": selected_manifest}, indent=2))


if __name__ == "__main__":
    main()
