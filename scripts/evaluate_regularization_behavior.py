from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "evaluate_regularization_behavior.py requires 'pandas'. "
        "Use the project's prepared environment, e.g. .\\.venv\\Scripts\\python.exe ..."
    ) from exc

from inpainting3d.data import (
    center_of_mass_threshold,
    fibonacci_normals,
    load_nifti_volume,
    normalize_volume,
    random_normals,
    resample_to_shape,
    simulate_sparse_acquisition,
)
from inpainting3d.metrics import compute_metrics
from inpainting3d.repro import build_repro_metadata
from inpainting3d.splits import (
    case_id_from_path,
    filter_to_known_files,
    get_split_files,
    load_split_manifest,
    normalize_case_path,
    resolve_case_paths,
    write_split_manifest,
)
from inpainting3d.stats import summarize_metric_values
from inpainting3d.utils import ensure_dir, strip_dataparallel_prefix, torch_load_weights_compat
from scripts.train_inpainting import build_model, clamp_prediction


DEFAULT_STUDIES = ["convergence", "noise", "sampling", "sensitivity"]
DEFAULT_CONVERGENCE_SLICES = [4, 8, 16, 32, 64, 128]
DEFAULT_NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05]
DEFAULT_SENSITIVITY_NOISE_LEVELS = [0.005, 0.01, 0.02]
DEFAULT_HEADLINE_METRICS = ["rmse_missing", "ssim_all"]
DEFAULT_PLOT_EXTENSION = ".pdf"
DISPLAY_NAMES = {
    "en": {
        "rmse_missing": "RMSE missing",
        "mae_missing": "MAE missing",
        "psnr_missing": "PSNR missing",
        "ssim_missing": "SSIM missing",
        "ssim_all": "SSIM full volume",
        "stability_ratio_all": "Stability ratio full volume",
        "stability_ratio_missing": "Stability ratio missing",
        "output_delta_rmse_all": "Output delta RMSE full volume",
        "output_delta_rmse_missing": "Output delta RMSE missing",
        "input_delta_rms_known": "Input delta RMS observed",
    },
    "de": {
        "rmse_missing": "RMSE fehlende Voxel",
        "mae_missing": "MAE fehlende Voxel",
        "psnr_missing": "PSNR fehlende Voxel",
        "ssim_missing": "SSIM fehlende Voxel",
        "ssim_all": "SSIM Gesamtvolumen",
        "stability_ratio_all": "Stabilitätsverhältnis Gesamtvolumen",
        "stability_ratio_missing": "Stabilitätsverhältnis fehlende Voxel",
        "output_delta_rmse_all": "Ausgabe-Delta RMSE Gesamtvolumen",
        "output_delta_rmse_missing": "Ausgabe-Delta RMSE fehlende Voxel",
        "input_delta_rms_known": "Eingabe-Delta RMS beobachtete Voxel",
    },
}
UI_TEXT = {
    "en": {
        "repeat_std_suffix": "repeat std",
        "repeat_mean_suffix": "repeat mean",
        "across_models_std_suffix": "across-model std",
        "across_models_mean_suffix": "across-model mean",
        "row_slices": "Slices",
        "row_noise_std": "Noise std",
        "row_checkpoint": "Checkpoint",
        "x_slices": "Number of slices",
        "x_noise_std": "Input noise std",
        "title_information_convergence": "Information convergence",
        "title_noise_stability": "Noise stability",
        "title_sampling_mean": "Sampling stability: mean reconstruction quality",
        "title_sampling_repeat": "Sampling stability: within-case repeat std",
        "title_input_sensitivity": "Input sensitivity",
        "title_prediction_change": "Prediction change under perturbation",
        "ylabel_stability_ratio": "Stability ratio",
        "ylabel_output_delta_rmse": "Output delta RMSE",
        "title_training_robustness": "Training robustness across checkpoints",
        "title_sampling_overview": "Sampling stability",
        "title_input_sensitivity_overview": "Input sensitivity",
    },
    "de": {
        "repeat_std_suffix": "Wiederholungs-Std.",
        "repeat_mean_suffix": "Wiederholungsmittel",
        "across_models_std_suffix": "Std. über Modelle",
        "across_models_mean_suffix": "Mittel über Modelle",
        "row_slices": "Schnitte",
        "row_noise_std": "Rausch-Std.",
        "row_checkpoint": "Checkpoint",
        "x_slices": "Anzahl der Schnitte",
        "x_noise_std": "Standardabw. des Eingangsrauschens",
        "title_information_convergence": "Informationskonvergenz",
        "title_noise_stability": "Rauschstabilität",
        "title_sampling_mean": "Sampling-Stabilität: mittlere Rekonstruktionsqualität",
        "title_sampling_repeat": "Sampling-Stabilität: Wiederholungs-Std. pro Fall",
        "title_input_sensitivity": "Eingabesensitivität",
        "title_prediction_change": "Änderung der Vorhersage unter Störung",
        "ylabel_stability_ratio": "Stabilitätsverhältnis",
        "ylabel_output_delta_rmse": "Ausgabe-Delta RMSE",
        "title_training_robustness": "Robustheit gegenüber Trainings-Checkpoints",
        "title_sampling_overview": "Sampling-Stabilität",
        "title_input_sensitivity_overview": "Eingabesensitivität",
    },
}
ACTIVE_LANGUAGE = "en"
COLOR_PALETTE = {
    "primary": "#0f5c6e",
    "secondary": "#a43d2c",
    "accent": "#d89000",
    "muted": "#54616c",
    "grid": "#d8dee3",
}


@dataclass
class PreparedCase:
    case_id: str
    path: str
    volume: np.ndarray
    center: np.ndarray


@dataclass
class Observation:
    sparse: np.ndarray
    mask: np.ndarray
    hits: np.ndarray
    strategy: str
    num_slices: int
    geometry_seed: int | None
    noise_std: float
    noise_seed: int | None


@dataclass
class ModelSpec:
    label: str
    weights: str
    model_kind: str
    init_feat: int


class ModelRunner:
    def __init__(
        self,
        *,
        model_kind: str,
        weights: str,
        init_feat: int,
        hard_constraint: bool,
        device: torch.device,
    ) -> None:
        self.model_kind = str(model_kind)
        self.weights = normalize_case_path(weights)
        self.init_feat = int(init_feat)
        self.hard_constraint = bool(hard_constraint)
        self.device = device
        self.model = build_model(self.model_kind, init_feat=self.init_feat).to(device)

        state = torch_load_weights_compat(self.weights, map_location=device)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if not (isinstance(state, dict) and any(isinstance(value, torch.Tensor) for value in state.values())):
            raise RuntimeError(f"Gewichte '{self.weights}' haben kein erwartetes state_dict-Format.")
        self.model.load_state_dict(strip_dataparallel_prefix(state), strict=True)
        self.model.eval()

    def predict(self, sparse: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x = np.stack([sparse, mask], axis=0)[None, ...].astype(np.float32)
        x_t = torch.from_numpy(x).to(self.device)
        sparse_t = x_t[:, 0:1, ...]
        mask_t = x_t[:, 1:2, ...]
        with torch.no_grad():
            pred = clamp_prediction(
                self.model(x_t),
                sparse=sparse_t,
                mask=mask_t,
                hard_constraint=self.hard_constraint,
            )
        return pred[0, 0].detach().cpu().numpy().astype(np.float32)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.6, 5.0),
            "figure.dpi": 120,
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": "#73808b",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": COLOR_PALETTE["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_json(path: str, payload: Mapping[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_rows_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path, index=False)


def write_dataframe_csv(path: str, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def write_dataframe_latex(path: str, frame: pd.DataFrame) -> None:
    latex = frame.to_latex(index=False, escape=False)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(latex)


def read_dataframe_csv_if_exists(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def plot_output_path(plot_dir: str, stem: str) -> str:
    return os.path.join(plot_dir, f"{stem}{DEFAULT_PLOT_EXTENSION}")


def ordered_unique(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values))


def set_language(language: str) -> None:
    global ACTIVE_LANGUAGE
    resolved = str(language).strip().lower()
    if resolved not in DISPLAY_NAMES:
        raise ValueError(f"Unsupported language: {language}")
    ACTIVE_LANGUAGE = resolved


def text(key: str) -> str:
    localized = UI_TEXT.get(ACTIVE_LANGUAGE, UI_TEXT["en"])
    return str(localized.get(key, UI_TEXT["en"].get(key, key)))


def metric_label(metric_name: str) -> str:
    if metric_name.endswith("_repeat_std"):
        base = metric_name[: -len("_repeat_std")]
        return f"{metric_label(base)} {text('repeat_std_suffix')}"
    if metric_name.endswith("_repeat_mean"):
        base = metric_name[: -len("_repeat_mean")]
        return f"{metric_label(base)} {text('repeat_mean_suffix')}"
    if metric_name.endswith("_across_models_std"):
        base = metric_name[: -len("_across_models_std")]
        return f"{metric_label(base)} {text('across_models_std_suffix')}"
    if metric_name.endswith("_across_models_mean"):
        base = metric_name[: -len("_across_models_mean")]
        return f"{metric_label(base)} {text('across_models_mean_suffix')}"
    return DISPLAY_NAMES.get(ACTIVE_LANGUAGE, DISPLAY_NAMES["en"]).get(metric_name, metric_name.replace("_", " "))


def validate_positive_int_list(values: Sequence[int], name: str) -> List[int]:
    resolved = [int(value) for value in values]
    if not resolved or any(value <= 0 for value in resolved):
        raise ValueError(f"{name} must contain at least one positive integer.")
    return resolved


def validate_nonnegative_float_list(values: Sequence[float], name: str) -> List[float]:
    resolved = [float(value) for value in values]
    if not resolved or any(value < 0.0 for value in resolved):
        raise ValueError(f"{name} must contain at least one non-negative float.")
    return resolved


def make_seed(base_seed: int, *parts: int) -> int:
    seed_sequence = np.random.SeedSequence([int(base_seed), *[int(part) for part in parts]])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def masked_rms(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    arr = values.astype(np.float32)
    if mask is not None:
        active = arr[mask > 0.5]
        if active.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(active ** 2)))
    return float(np.sqrt(np.mean(arr ** 2)))


def resolve_selected_files(args: argparse.Namespace) -> tuple[list[str], dict[str, object] | None]:
    available_files = resolve_case_paths(args.data)
    if not available_files:
        raise RuntimeError("Keine Dateien fuer das angegebene Daten-Glob gefunden.")

    if args.split_file:
        manifest = load_split_manifest(args.split_file)
        selected = filter_to_known_files(get_split_files(manifest, args.split), available_files)
    else:
        if not args.allow_all_data_eval:
            raise RuntimeError(
                "Diese Auswertung erwartet standardmaessig einen persistierten Split. "
                "Pass --split_file <manifest.json> oder aktiviere bewusst --allow_all_data_eval."
            )
        manifest = None
        selected = list(available_files)

    if args.case_ids:
        requested = set(str(case_id) for case_id in args.case_ids)
        selected = [path for path in selected if case_id_from_path(path) in requested]
        if not selected:
            raise RuntimeError("Keiner der angeforderten case_ids wurde im ausgewaehlten Split gefunden.")

    if args.max_cases is not None:
        limit = int(args.max_cases)
        if limit <= 0:
            raise ValueError("--max_cases must be >= 1 when provided.")
        selected = selected[:limit]

    if not selected:
        raise RuntimeError("Die Fallauswahl ist leer.")
    return selected, manifest


def load_prepared_cases(
    files: Sequence[str],
    *,
    dim: int,
    normalize: str,
    canonical: bool,
) -> List[PreparedCase]:
    prepared: List[PreparedCase] = []
    for path in files:
        volume, _ = load_nifti_volume(path, canonical=canonical, dtype=np.float32)
        volume = normalize_volume(volume, normalize)
        volume = resample_to_shape(volume, (dim, dim, dim), order=1)
        center = center_of_mass_threshold(volume, thr=0.1)
        prepared.append(
            PreparedCase(
                case_id=case_id_from_path(path),
                path=normalize_case_path(path),
                volume=volume.astype(np.float32),
                center=center.astype(np.float32),
            )
        )
    return prepared


def build_observation(
    case: PreparedCase,
    *,
    num_slices: int,
    strategy: str,
    thickness_vox: float,
    geometry_seed: int | None = None,
    noise_std: float = 0.0,
    noise_seed: int | None = None,
) -> Observation:
    if strategy == "fibonacci":
        normals = fibonacci_normals(int(num_slices))
    elif strategy == "random":
        if geometry_seed is None:
            raise ValueError("geometry_seed is required for random sampling.")
        rng = np.random.default_rng(int(geometry_seed))
        normals = random_normals(int(num_slices), rng)
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    sparse, mask, hits = simulate_sparse_acquisition(
        case.volume,
        normals=normals,
        center=case.center,
        thickness_vox=float(thickness_vox),
        chunk=64,
    )
    if noise_std > 0.0:
        if noise_seed is None:
            raise ValueError("noise_seed is required when noise_std > 0.")
        rng = np.random.default_rng(int(noise_seed))
        sparse = np.clip(
            sparse + rng.normal(0.0, float(noise_std), size=sparse.shape).astype(np.float32),
            0.0,
            1.0,
        )

    return Observation(
        sparse=sparse.astype(np.float32),
        mask=mask.astype(np.float32),
        hits=hits.astype(np.float32),
        strategy=strategy,
        num_slices=int(num_slices),
        geometry_seed=None if geometry_seed is None else int(geometry_seed),
        noise_std=float(noise_std),
        noise_seed=None if noise_seed is None else int(noise_seed),
    )


def to_tensor(volume: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(volume[None, None]).float()


def evaluate_prediction(
    pred: np.ndarray,
    *,
    target: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    metrics = compute_metrics(to_tensor(pred), to_tensor(target), known_mask=to_tensor(mask), data_range=1.0)
    return {key: float(value) for key, value in metrics.items()}


def summarize_grouped_metrics(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()

    grouped = frame.groupby(list(group_cols), dropna=False, sort=True) if group_cols else [((), frame)]
    for group_key, group_frame in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_map = {str(col): value for col, value in zip(group_cols, group_key)}
        for metric_name in metric_cols:
            if metric_name not in group_frame.columns:
                continue
            values = group_frame[metric_name].astype(float).tolist()
            if not values:
                continue
            row: Dict[str, Any] = dict(group_map)
            row["metric"] = metric_name
            row.update(summarize_metric_values(values))
            rows.append(row)
    return pd.DataFrame(rows)


def extract_summary_row(summary_frame: pd.DataFrame, selector: Mapping[str, object], metric_name: str) -> pd.Series | None:
    if summary_frame.empty:
        return None
    mask = summary_frame["metric"] == metric_name
    for key, value in selector.items():
        mask &= summary_frame[key] == value
    subset = summary_frame.loc[mask]
    if subset.empty:
        return None
    return subset.iloc[0]


def format_mean_ci(mean: float, ci_low: float, ci_high: float, precision: int = 4) -> str:
    ci_half = max(abs(mean - ci_low), abs(ci_high - mean))
    return f"{mean:.{precision}f} $\\pm$ {ci_half:.{precision}f}"


def build_formatted_metric_table(
    summary_frame: pd.DataFrame,
    *,
    row_key: str,
    metrics: Sequence[str],
    row_label: str,
    precision: int = 4,
) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    unique_keys = summary_frame[row_key].drop_duplicates().tolist()
    for key_value in unique_keys:
        row: Dict[str, Any] = {row_label: key_value}
        for metric_name in metrics:
            stats_row = extract_summary_row(summary_frame, {row_key: key_value}, metric_name)
            if stats_row is None:
                continue
            row[metric_label(metric_name)] = format_mean_ci(
                float(stats_row["mean"]),
                float(stats_row["ci95_low"]),
                float(stats_row["ci95_high"]),
                precision=precision,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_sampling_table(
    raw_summary: pd.DataFrame,
    variability_summary: pd.DataFrame,
    *,
    row_key: str,
    metrics: Sequence[str],
    row_label: str,
    precision: int = 4,
) -> pd.DataFrame:
    if raw_summary.empty or variability_summary.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    unique_keys = raw_summary[row_key].drop_duplicates().tolist()
    for key_value in unique_keys:
        row: Dict[str, Any] = {row_label: key_value}
        for metric_name in metrics:
            mean_row = extract_summary_row(raw_summary, {row_key: key_value}, metric_name)
            std_row = extract_summary_row(variability_summary, {row_key: key_value}, f"{metric_name}_repeat_std")
            if mean_row is not None:
                row[f"{metric_label(metric_name)} mean"] = format_mean_ci(
                    float(mean_row["mean"]),
                    float(mean_row["ci95_low"]),
                    float(mean_row["ci95_high"]),
                    precision=precision,
                )
            if std_row is not None:
                row[f"{metric_label(metric_name)} repeat std"] = format_mean_ci(
                    float(std_row["mean"]),
                    float(std_row["ci95_low"]),
                    float(std_row["ci95_high"]),
                    precision=precision,
                )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_line(
    ax: plt.Axes,
    summary_frame: pd.DataFrame,
    *,
    x_col: str,
    metric_name: str,
    color: str,
    x_label: str,
    title: str,
) -> None:
    subset = summary_frame.loc[summary_frame["metric"] == metric_name].copy()
    if subset.empty:
        ax.set_visible(False)
        return
    subset = subset.sort_values(by=x_col)
    xs = subset[x_col].astype(float).to_numpy()
    means = subset["mean"].astype(float).to_numpy()
    ci_low = subset["ci95_low"].astype(float).to_numpy()
    ci_high = subset["ci95_high"].astype(float).to_numpy()

    ax.plot(xs, means, marker="o", linewidth=2.3, markersize=5.5, color=color)
    ax.fill_between(xs, ci_low, ci_high, color=color, alpha=0.18)
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_label(metric_name))
    ax.set_title(title)


def save_dual_metric_plot(
    summary_frame: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    primary_metric: str,
    secondary_metric: str,
    title_prefix: str,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    plot_metric_line(
        axes[0],
        summary_frame,
        x_col=x_col,
        metric_name=primary_metric,
        color=COLOR_PALETTE["primary"],
        x_label=x_label,
        title=f"{title_prefix}: {metric_label(primary_metric)}",
    )
    plot_metric_line(
        axes[1],
        summary_frame,
        x_col=x_col,
        metric_name=secondary_metric,
        color=COLOR_PALETTE["secondary"],
        x_label=x_label,
        title=f"{title_prefix}: {metric_label(secondary_metric)}",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_sampling_plot(
    raw_summary: pd.DataFrame,
    variability_summary: pd.DataFrame,
    *,
    primary_metric: str,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    plot_metric_line(
        axes[0],
        raw_summary,
        x_col="slice_count",
        metric_name=primary_metric,
        color=COLOR_PALETTE["primary"],
        x_label=text("x_slices"),
        title=text("title_sampling_mean"),
    )
    plot_metric_line(
        axes[1],
        variability_summary,
        x_col="slice_count",
        metric_name=f"{primary_metric}_repeat_std",
        color=COLOR_PALETTE["accent"],
        x_label=text("x_slices"),
        title=text("title_sampling_repeat"),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_sensitivity_plot(summary_frame: pd.DataFrame, *, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for metric_name, color in [
        ("stability_ratio_all", COLOR_PALETTE["primary"]),
        ("stability_ratio_missing", COLOR_PALETTE["secondary"]),
    ]:
        subset = summary_frame.loc[summary_frame["metric"] == metric_name].sort_values(by="noise_std")
        if subset.empty:
            continue
        xs = subset["noise_std"].astype(float).to_numpy()
        means = subset["mean"].astype(float).to_numpy()
        ci_low = subset["ci95_low"].astype(float).to_numpy()
        ci_high = subset["ci95_high"].astype(float).to_numpy()
        axes[0].plot(xs, means, marker="o", linewidth=2.2, color=color, label=metric_label(metric_name))
        axes[0].fill_between(xs, ci_low, ci_high, color=color, alpha=0.16)
    axes[0].set_xlabel(text("x_noise_std"))
    axes[0].set_ylabel(text("ylabel_stability_ratio"))
    axes[0].set_title(text("title_input_sensitivity"))
    axes[0].legend(loc="best")

    for metric_name, color in [
        ("output_delta_rmse_all", COLOR_PALETTE["primary"]),
        ("output_delta_rmse_missing", COLOR_PALETTE["secondary"]),
    ]:
        subset = summary_frame.loc[summary_frame["metric"] == metric_name].sort_values(by="noise_std")
        if subset.empty:
            continue
        xs = subset["noise_std"].astype(float).to_numpy()
        means = subset["mean"].astype(float).to_numpy()
        ci_low = subset["ci95_low"].astype(float).to_numpy()
        ci_high = subset["ci95_high"].astype(float).to_numpy()
        axes[1].plot(xs, means, marker="o", linewidth=2.2, color=color, label=metric_label(metric_name))
        axes[1].fill_between(xs, ci_low, ci_high, color=color, alpha=0.16)
    axes[1].set_xlabel(text("x_noise_std"))
    axes[1].set_ylabel(text("ylabel_output_delta_rmse"))
    axes[1].set_title(text("title_prediction_change"))
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_robustness_plot(
    frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    *,
    primary_metric: str,
    out_path: str,
) -> None:
    ordered_labels = frame["model_label"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for idx, label in enumerate(ordered_labels):
        values = frame.loc[frame["model_label"] == label, primary_metric].astype(float).to_numpy()
        if values.size == 0:
            continue
        x_points = np.full(values.shape, idx, dtype=np.float32)
        ax.scatter(x_points, values, color=COLOR_PALETTE["muted"], alpha=0.35, s=18)
        stats_row = extract_summary_row(summary_frame, {"model_label": label}, primary_metric)
        if stats_row is None:
            continue
        mean = float(stats_row["mean"])
        ci_low = float(stats_row["ci95_low"])
        ci_high = float(stats_row["ci95_high"])
        ax.errorbar(
            [idx],
            [mean],
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="o",
            color=COLOR_PALETTE["primary"],
            markersize=7,
            linewidth=2,
            capsize=4,
        )
    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels, rotation=20, ha="right")
    ax.set_ylabel(metric_label(primary_metric))
    ax.set_title(text("title_training_robustness"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_overview_plot(
    *,
    out_path: str,
    convergence_summary: pd.DataFrame | None,
    noise_summary: pd.DataFrame | None,
    sampling_variability_summary: pd.DataFrame | None,
    sensitivity_summary: pd.DataFrame | None,
    primary_metric: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
    axes = axes.ravel()

    if convergence_summary is not None and not convergence_summary.empty:
        plot_metric_line(
            axes[0],
            convergence_summary,
            x_col="slice_count",
            metric_name=primary_metric,
            color=COLOR_PALETTE["primary"],
            x_label=text("x_slices"),
            title=text("title_information_convergence"),
        )
    else:
        axes[0].set_visible(False)

    if noise_summary is not None and not noise_summary.empty:
        plot_metric_line(
            axes[1],
            noise_summary,
            x_col="noise_std",
            metric_name=primary_metric,
            color=COLOR_PALETTE["secondary"],
            x_label=text("x_noise_std"),
            title=text("title_noise_stability"),
        )
    else:
        axes[1].set_visible(False)

    if sampling_variability_summary is not None and not sampling_variability_summary.empty:
        plot_metric_line(
            axes[2],
            sampling_variability_summary,
            x_col="slice_count",
            metric_name=f"{primary_metric}_repeat_std",
            color=COLOR_PALETTE["accent"],
            x_label=text("x_slices"),
            title=text("title_sampling_overview"),
        )
    else:
        axes[2].set_visible(False)

    if sensitivity_summary is not None and not sensitivity_summary.empty:
        plot_metric_line(
            axes[3],
            sensitivity_summary,
            x_col="noise_std",
            metric_name="stability_ratio_missing",
            color=COLOR_PALETTE["primary"],
            x_label=text("x_noise_std"),
            title=text("title_input_sensitivity_overview"),
        )
    else:
        axes[3].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def select_metric_columns(frame: pd.DataFrame, *, exclude: Sequence[str]) -> List[str]:
    excluded = set(str(name) for name in exclude)
    return [str(name) for name in frame.columns if str(name) not in excluded]


def run_convergence_study(
    cases: Sequence[PreparedCase],
    *,
    runner: ModelRunner,
    slice_counts: Sequence[int],
    fixed_strategy: str,
    thickness_vox: float,
    base_seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        for slice_count in slice_counts:
            geometry_seed = None if fixed_strategy == "fibonacci" else make_seed(base_seed, 10, case_index, slice_count)
            observation = build_observation(
                case,
                num_slices=slice_count,
                strategy=fixed_strategy,
                thickness_vox=thickness_vox,
                geometry_seed=geometry_seed,
            )
            pred = runner.predict(observation.sparse, observation.mask)
            row: Dict[str, Any] = {
                "case_id": case.case_id,
                "path": case.path,
                "slice_count": int(slice_count),
                "sampling_strategy": fixed_strategy,
                "known_fraction": float(observation.mask.mean()),
            }
            row.update(evaluate_prediction(pred, target=case.volume, mask=observation.mask))
            rows.append(row)
    return pd.DataFrame(rows)


def run_noise_study(
    cases: Sequence[PreparedCase],
    *,
    runner: ModelRunner,
    noise_levels: Sequence[float],
    num_slices: int,
    fixed_strategy: str,
    thickness_vox: float,
    base_seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        geometry_seed = None if fixed_strategy == "fibonacci" else make_seed(base_seed, 20, case_index, num_slices)
        base_observation = build_observation(
            case,
            num_slices=num_slices,
            strategy=fixed_strategy,
            thickness_vox=thickness_vox,
            geometry_seed=geometry_seed,
        )
        for level_index, noise_std in enumerate(noise_levels):
            if noise_std > 0.0:
                noise_seed = make_seed(base_seed, 21, case_index, num_slices, level_index)
                observation = build_observation(
                    case,
                    num_slices=num_slices,
                    strategy=fixed_strategy,
                    thickness_vox=thickness_vox,
                    geometry_seed=geometry_seed,
                    noise_std=noise_std,
                    noise_seed=noise_seed,
                )
            else:
                observation = base_observation
            pred = runner.predict(observation.sparse, observation.mask)
            row: Dict[str, Any] = {
                "case_id": case.case_id,
                "path": case.path,
                "noise_std": float(noise_std),
                "slice_count": int(num_slices),
                "sampling_strategy": fixed_strategy,
                "known_fraction": float(observation.mask.mean()),
                "realized_input_delta_rms_known": float(
                    masked_rms(observation.sparse - base_observation.sparse, base_observation.mask)
                ),
            }
            row.update(evaluate_prediction(pred, target=case.volume, mask=observation.mask))
            rows.append(row)
    return pd.DataFrame(rows)


def run_sampling_study(
    cases: Sequence[PreparedCase],
    *,
    runner: ModelRunner,
    slice_counts: Sequence[int],
    repeats: int,
    thickness_vox: float,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if int(repeats) < 2:
        raise ValueError("--sampling_repeats must be >= 2.")

    rows: List[Dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        for slice_count in slice_counts:
            for repeat_index in range(int(repeats)):
                geometry_seed = make_seed(base_seed, 30, case_index, slice_count, repeat_index)
                observation = build_observation(
                    case,
                    num_slices=slice_count,
                    strategy="random",
                    thickness_vox=thickness_vox,
                    geometry_seed=geometry_seed,
                )
                pred = runner.predict(observation.sparse, observation.mask)
                row: Dict[str, Any] = {
                    "case_id": case.case_id,
                    "path": case.path,
                    "slice_count": int(slice_count),
                    "repeat_index": int(repeat_index),
                    "geometry_seed": int(geometry_seed),
                    "known_fraction": float(observation.mask.mean()),
                }
                row.update(evaluate_prediction(pred, target=case.volume, mask=observation.mask))
                rows.append(row)

    raw_frame = pd.DataFrame(rows)
    metric_cols = select_metric_columns(
        raw_frame,
        exclude=["case_id", "path", "slice_count", "repeat_index", "geometry_seed"],
    )
    variability_rows: List[Dict[str, Any]] = []
    for (case_id, slice_count), group_frame in raw_frame.groupby(["case_id", "slice_count"], sort=True):
        row: Dict[str, Any] = {
            "case_id": case_id,
            "slice_count": int(slice_count),
            "path": str(group_frame["path"].iloc[0]),
        }
        for metric_name in metric_cols:
            values = group_frame[metric_name].astype(float).to_numpy()
            row[f"{metric_name}_repeat_mean"] = float(np.mean(values))
            row[f"{metric_name}_repeat_std"] = float(np.std(values, ddof=1))
        variability_rows.append(row)
    return raw_frame, pd.DataFrame(variability_rows)


def run_sensitivity_study(
    cases: Sequence[PreparedCase],
    *,
    runner: ModelRunner,
    noise_levels: Sequence[float],
    num_slices: int,
    fixed_strategy: str,
    thickness_vox: float,
    base_seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        geometry_seed = None if fixed_strategy == "fibonacci" else make_seed(base_seed, 40, case_index, num_slices)
        base_observation = build_observation(
            case,
            num_slices=num_slices,
            strategy=fixed_strategy,
            thickness_vox=thickness_vox,
            geometry_seed=geometry_seed,
        )
        base_pred = runner.predict(base_observation.sparse, base_observation.mask)
        for level_index, noise_std in enumerate(noise_levels):
            noise_seed = make_seed(base_seed, 41, case_index, num_slices, level_index)
            perturbed_observation = build_observation(
                case,
                num_slices=num_slices,
                strategy=fixed_strategy,
                thickness_vox=thickness_vox,
                geometry_seed=geometry_seed,
                noise_std=noise_std,
                noise_seed=noise_seed,
            )
            perturbed_pred = runner.predict(perturbed_observation.sparse, perturbed_observation.mask)
            delta_metrics = evaluate_prediction(
                perturbed_pred,
                target=base_pred,
                mask=base_observation.mask,
            )
            input_delta = perturbed_observation.sparse - base_observation.sparse
            input_delta_rms = masked_rms(input_delta, base_observation.mask)

            row: Dict[str, Any] = {
                "case_id": case.case_id,
                "path": case.path,
                "noise_std": float(noise_std),
                "slice_count": int(num_slices),
                "sampling_strategy": fixed_strategy,
                "input_delta_rms_known": float(input_delta_rms),
            }
            for key, value in delta_metrics.items():
                row[f"output_delta_{key}"] = float(value)
            denom = max(float(input_delta_rms), 1e-8)
            row["stability_ratio_all"] = float(row["output_delta_rmse_all"] / denom)
            row["stability_ratio_missing"] = float(row["output_delta_rmse_missing"] / denom)
            rows.append(row)
    return pd.DataFrame(rows)


def resolve_model_specs(args: argparse.Namespace) -> List[ModelSpec]:
    weights = ordered_unique(args.robustness_weights)
    if not weights:
        return []
    labels = list(args.robustness_labels)
    if labels and len(labels) != len(weights):
        raise ValueError("--robustness_labels must match the length of --robustness_weights.")
    if not labels:
        labels = [os.path.splitext(os.path.basename(path))[0] for path in weights]
    return [
        ModelSpec(
            label=str(label),
            weights=str(weight),
            model_kind=str(args.model),
            init_feat=int(args.init_feat),
        )
        for label, weight in zip(labels, weights)
    ]


def run_robustness_study(
    cases: Sequence[PreparedCase],
    *,
    model_specs: Sequence[ModelSpec],
    hard_constraint: bool,
    num_slices: int,
    fixed_strategy: str,
    thickness_vox: float,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(model_specs) < 2:
        return pd.DataFrame(), pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runners = [
        (spec, ModelRunner(
            model_kind=spec.model_kind,
            weights=spec.weights,
            init_feat=spec.init_feat,
            hard_constraint=hard_constraint,
            device=device,
        ))
        for spec in model_specs
    ]

    rows: List[Dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        geometry_seed = None if fixed_strategy == "fibonacci" else make_seed(base_seed, 50, case_index, num_slices)
        observation = build_observation(
            case,
            num_slices=num_slices,
            strategy=fixed_strategy,
            thickness_vox=thickness_vox,
            geometry_seed=geometry_seed,
        )
        for spec, runner in runners:
            pred = runner.predict(observation.sparse, observation.mask)
            row: Dict[str, Any] = {
                "case_id": case.case_id,
                "path": case.path,
                "model_label": spec.label,
                "weight_path": normalize_case_path(spec.weights),
                "slice_count": int(num_slices),
                "known_fraction": float(observation.mask.mean()),
            }
            row.update(evaluate_prediction(pred, target=case.volume, mask=observation.mask))
            rows.append(row)

    raw_frame = pd.DataFrame(rows)
    metric_cols = select_metric_columns(
        raw_frame,
        exclude=["case_id", "path", "model_label", "weight_path", "slice_count"],
    )
    spread_rows: List[Dict[str, Any]] = []
    for case_id, group_frame in raw_frame.groupby("case_id", sort=True):
        row: Dict[str, Any] = {
            "case_id": case_id,
            "path": str(group_frame["path"].iloc[0]),
        }
        for metric_name in metric_cols:
            values = group_frame[metric_name].astype(float).to_numpy()
            row[f"{metric_name}_across_models_mean"] = float(np.mean(values))
            row[f"{metric_name}_across_models_std"] = float(np.std(values, ddof=1))
        spread_rows.append(row)
    return raw_frame, pd.DataFrame(spread_rows)


def combine_latex_tables(output_dir: str, table_paths: Sequence[str]) -> None:
    fragments: List[str] = []
    for path in table_paths:
        if not os.path.exists(path):
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as handle:
            fragments.append(f"% {stem}\n{handle.read().strip()}\n")
    if not fragments:
        return
    with open(os.path.join(output_dir, "thesis_tables.tex"), "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(fragments))


def ensure_metric_names_exist(metric_names: Sequence[str], available_metrics: Sequence[str], context: str) -> None:
    missing = [name for name in metric_names if name not in available_metrics]
    if missing:
        available = ", ".join(sorted(str(name) for name in available_metrics))
        raise ValueError(f"{context}: requested metrics not available: {missing}. Available metrics: {available}")


def render_outputs_from_existing(
    *,
    out_dir: str,
    studies: Sequence[str],
    headline_metrics: Sequence[str],
    primary_metric: str,
    secondary_metric: str,
) -> None:
    raw_dir = os.path.join(out_dir, "raw")
    summary_dir = os.path.join(out_dir, "summary")
    table_dir = os.path.join(out_dir, "tables")
    plot_dir = os.path.join(out_dir, "plots")
    for path in (raw_dir, summary_dir, table_dir, plot_dir):
        ensure_dir(path)

    generated_table_paths: List[str] = []
    convergence_summary: pd.DataFrame | None = None
    noise_summary: pd.DataFrame | None = None
    sampling_variability_summary: pd.DataFrame | None = None
    sensitivity_summary: pd.DataFrame | None = None

    if "convergence" in studies:
        convergence_summary = read_dataframe_csv_if_exists(os.path.join(summary_dir, "convergence_summary_long.csv"))
        if convergence_summary is not None and not convergence_summary.empty:
            ensure_metric_names_exist(
                [primary_metric, secondary_metric, *headline_metrics],
                convergence_summary["metric"].astype(str).tolist(),
                "convergence",
            )
            convergence_table = build_formatted_metric_table(
                convergence_summary,
                row_key="slice_count",
                metrics=headline_metrics,
                row_label=text("row_slices"),
            )
            write_dataframe_csv(os.path.join(summary_dir, "convergence_table.csv"), convergence_table)
            convergence_tex = os.path.join(table_dir, "convergence_table.tex")
            write_dataframe_latex(convergence_tex, convergence_table)
            generated_table_paths.append(convergence_tex)
            save_dual_metric_plot(
                convergence_summary,
                x_col="slice_count",
                x_label=text("x_slices"),
                primary_metric=primary_metric,
                secondary_metric=secondary_metric,
                title_prefix=text("title_information_convergence"),
                out_path=plot_output_path(plot_dir, "convergence_dual_metrics"),
            )

    if "noise" in studies:
        noise_summary = read_dataframe_csv_if_exists(os.path.join(summary_dir, "noise_summary_long.csv"))
        if noise_summary is not None and not noise_summary.empty:
            ensure_metric_names_exist(
                [primary_metric, secondary_metric, *headline_metrics],
                noise_summary["metric"].astype(str).tolist(),
                "noise",
            )
            noise_table = build_formatted_metric_table(
                noise_summary,
                row_key="noise_std",
                metrics=headline_metrics,
                row_label=text("row_noise_std"),
            )
            write_dataframe_csv(os.path.join(summary_dir, "noise_table.csv"), noise_table)
            noise_tex = os.path.join(table_dir, "noise_table.tex")
            write_dataframe_latex(noise_tex, noise_table)
            generated_table_paths.append(noise_tex)
            save_dual_metric_plot(
                noise_summary,
                x_col="noise_std",
                x_label=text("x_noise_std"),
                primary_metric=primary_metric,
                secondary_metric=secondary_metric,
                title_prefix=text("title_noise_stability"),
                out_path=plot_output_path(plot_dir, "noise_dual_metrics"),
            )

    if "sampling" in studies:
        sampling_summary = read_dataframe_csv_if_exists(os.path.join(summary_dir, "sampling_summary_long.csv"))
        sampling_variability_summary = read_dataframe_csv_if_exists(
            os.path.join(summary_dir, "sampling_variability_summary_long.csv")
        )
        if (
            sampling_summary is not None
            and not sampling_summary.empty
            and sampling_variability_summary is not None
            and not sampling_variability_summary.empty
        ):
            ensure_metric_names_exist(
                [primary_metric, secondary_metric, *headline_metrics],
                sampling_summary["metric"].astype(str).tolist(),
                "sampling",
            )
            sampling_table = build_sampling_table(
                sampling_summary,
                sampling_variability_summary,
                row_key="slice_count",
                metrics=headline_metrics,
                row_label=text("row_slices"),
            )
            write_dataframe_csv(os.path.join(summary_dir, "sampling_table.csv"), sampling_table)
            sampling_tex = os.path.join(table_dir, "sampling_table.tex")
            write_dataframe_latex(sampling_tex, sampling_table)
            generated_table_paths.append(sampling_tex)
            save_sampling_plot(
                sampling_summary,
                sampling_variability_summary,
                primary_metric=primary_metric,
                out_path=plot_output_path(plot_dir, "sampling_stability"),
            )

    if "sensitivity" in studies:
        sensitivity_summary = read_dataframe_csv_if_exists(os.path.join(summary_dir, "sensitivity_summary_long.csv"))
        if sensitivity_summary is not None and not sensitivity_summary.empty:
            ensure_metric_names_exist(
                ["stability_ratio_missing", "stability_ratio_all", "output_delta_rmse_missing"],
                sensitivity_summary["metric"].astype(str).tolist(),
                "sensitivity",
            )
            sensitivity_table = build_formatted_metric_table(
                sensitivity_summary,
                row_key="noise_std",
                metrics=["stability_ratio_missing", "stability_ratio_all", "output_delta_rmse_missing"],
                row_label=text("row_noise_std"),
            )
            write_dataframe_csv(os.path.join(summary_dir, "sensitivity_table.csv"), sensitivity_table)
            sensitivity_tex = os.path.join(table_dir, "sensitivity_table.tex")
            write_dataframe_latex(sensitivity_tex, sensitivity_table)
            generated_table_paths.append(sensitivity_tex)
            save_sensitivity_plot(
                sensitivity_summary,
                out_path=plot_output_path(plot_dir, "sensitivity_stability"),
            )

    if "robustness" in studies:
        robustness_frame = read_dataframe_csv_if_exists(os.path.join(raw_dir, "robustness_per_case.csv"))
        robustness_summary = read_dataframe_csv_if_exists(os.path.join(summary_dir, "robustness_summary_long.csv"))
        if robustness_frame is not None and not robustness_frame.empty and robustness_summary is not None and not robustness_summary.empty:
            ensure_metric_names_exist(
                [primary_metric, *headline_metrics],
                robustness_summary["metric"].astype(str).tolist(),
                "robustness",
            )
            robustness_table = build_formatted_metric_table(
                robustness_summary,
                row_key="model_label",
                metrics=headline_metrics,
                row_label=text("row_checkpoint"),
            )
            write_dataframe_csv(os.path.join(summary_dir, "robustness_table.csv"), robustness_table)
            robustness_tex = os.path.join(table_dir, "robustness_table.tex")
            write_dataframe_latex(robustness_tex, robustness_table)
            generated_table_paths.append(robustness_tex)
            save_robustness_plot(
                robustness_frame,
                robustness_summary,
                primary_metric=primary_metric,
                out_path=plot_output_path(plot_dir, "robustness_primary_metric"),
            )

    save_overview_plot(
        out_path=plot_output_path(plot_dir, "thesis_overview"),
        convergence_summary=convergence_summary,
        noise_summary=noise_summary,
        sampling_variability_summary=sampling_variability_summary,
        sensitivity_summary=sensitivity_summary,
        primary_metric=primary_metric,
    )
    combine_latex_tables(table_dir, generated_table_paths)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--weights", type=str, default="")
    ap.add_argument("--out", type=str, default="./runs/regularization_behavior")
    ap.add_argument("--render_only", action="store_true")
    ap.add_argument("--language", type=str, default="en", choices=["en", "de"])
    ap.add_argument("--model", type=str, default="partial", choices=["baseline", "gated", "partial"])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--init_feat", type=int, default=32)
    ap.add_argument("--thickness_vox", type=float, default=1.0)
    ap.add_argument("--normalize", type=str, default="clip01")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--split_file", type=str, default="")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_cases", type=int, default=12)
    ap.add_argument("--case_ids", type=str, nargs="*", default=[])
    ap.add_argument("--allow_all_data_eval", action="store_true")
    ap.add_argument("--no_hard_constraint", dest="hard_constraint", action="store_false")
    ap.set_defaults(hard_constraint=True)

    ap.add_argument(
        "--studies",
        type=str,
        nargs="*",
        default=list(DEFAULT_STUDIES),
        choices=["convergence", "noise", "sampling", "sensitivity", "robustness"],
    )
    ap.add_argument("--fixed_sampling_strategy", type=str, default="fibonacci", choices=["fibonacci", "random"])
    ap.add_argument("--convergence_slices", type=int, nargs="*", default=list(DEFAULT_CONVERGENCE_SLICES))
    ap.add_argument("--noise_slices", type=int, default=64)
    ap.add_argument("--noise_levels", type=float, nargs="*", default=list(DEFAULT_NOISE_LEVELS))
    ap.add_argument("--sampling_slices", type=int, nargs="*", default=list(DEFAULT_CONVERGENCE_SLICES))
    ap.add_argument("--sampling_repeats", type=int, default=8)
    ap.add_argument("--sensitivity_slices", type=int, default=64)
    ap.add_argument("--sensitivity_noise_levels", type=float, nargs="*", default=list(DEFAULT_SENSITIVITY_NOISE_LEVELS))
    ap.add_argument("--robustness_slices", type=int, default=64)
    ap.add_argument("--robustness_weights", type=str, nargs="*", default=[])
    ap.add_argument("--robustness_labels", type=str, nargs="*", default=[])

    ap.add_argument("--headline_metrics", type=str, nargs="*", default=list(DEFAULT_HEADLINE_METRICS))
    ap.add_argument("--primary_metric", type=str, default="rmse_missing")
    ap.add_argument("--secondary_metric", type=str, default="ssim_all")
    args = ap.parse_args()

    set_language(args.language)
    configure_plot_style()

    existing_config: Dict[str, Any] | None = None
    existing_config_path = os.path.join(args.out, "study_config.json")
    if args.render_only and os.path.exists(existing_config_path):
        with open(existing_config_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            existing_config = loaded

    studies = ordered_unique(args.studies)
    if args.render_only and existing_config is not None:
        if args.studies == list(DEFAULT_STUDIES) and isinstance(existing_config.get("studies"), list):
            studies = ordered_unique(existing_config["studies"])
        if args.headline_metrics == list(DEFAULT_HEADLINE_METRICS) and isinstance(existing_config.get("headline_metrics"), list):
            args.headline_metrics = list(existing_config["headline_metrics"])
        if args.primary_metric == "rmse_missing" and existing_config.get("primary_metric"):
            args.primary_metric = str(existing_config["primary_metric"])
        if args.secondary_metric == "ssim_all" and existing_config.get("secondary_metric"):
            args.secondary_metric = str(existing_config["secondary_metric"])

    convergence_slices = validate_positive_int_list(args.convergence_slices, "--convergence_slices")
    noise_levels = validate_nonnegative_float_list(args.noise_levels, "--noise_levels")
    sampling_slices = validate_positive_int_list(args.sampling_slices, "--sampling_slices")
    sensitivity_noise_levels = validate_nonnegative_float_list(
        [value for value in args.sensitivity_noise_levels if float(value) > 0.0],
        "--sensitivity_noise_levels",
    )
    headline_metrics = ordered_unique(args.headline_metrics)

    if not args.render_only:
        if not str(args.data).strip():
            raise ValueError("--data is required unless --render_only is used.")
        if not str(args.weights).strip():
            raise ValueError("--weights is required unless --render_only is used.")

    ensure_dir(args.out)
    raw_dir = os.path.join(args.out, "raw")
    summary_dir = os.path.join(args.out, "summary")
    table_dir = os.path.join(args.out, "tables")
    plot_dir = os.path.join(args.out, "plots")
    for path in (raw_dir, summary_dir, table_dir, plot_dir):
        ensure_dir(path)

    if args.render_only:
        render_outputs_from_existing(
            out_dir=args.out,
            studies=studies,
            headline_metrics=headline_metrics,
            primary_metric=args.primary_metric,
            secondary_metric=args.secondary_metric,
        )
        print("Re-rendered empirical regularization artifacts from existing data.")
        print(f"Studies: {', '.join(studies)}")
        print(f"Language: {args.language}")
        print(f"Output: {normalize_case_path(args.out)}")
        return

    selected_files, split_manifest = resolve_selected_files(args)
    if split_manifest is not None:
        write_split_manifest(split_manifest, os.path.join(args.out, "split_manifest.json"))

    cases = load_prepared_cases(
        selected_files,
        dim=int(args.dim),
        normalize=str(args.normalize),
        canonical=True,
    )

    case_rows = [{"case_id": case.case_id, "path": case.path} for case in cases]
    write_rows_csv(os.path.join(args.out, "selected_cases.csv"), case_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = ModelRunner(
        model_kind=args.model,
        weights=args.weights,
        init_feat=args.init_feat,
        hard_constraint=args.hard_constraint,
        device=device,
    )

    generated_table_paths: List[str] = []
    study_index: Dict[str, Any] = {
        "selected_cases": [case.case_id for case in cases],
        "weights": normalize_case_path(args.weights),
        "studies": studies,
    }

    convergence_summary: pd.DataFrame | None = None
    noise_summary: pd.DataFrame | None = None
    sampling_variability_summary: pd.DataFrame | None = None
    sensitivity_summary: pd.DataFrame | None = None

    if "convergence" in studies:
        convergence_frame = run_convergence_study(
            cases,
            runner=runner,
            slice_counts=convergence_slices,
            fixed_strategy=args.fixed_sampling_strategy,
            thickness_vox=float(args.thickness_vox),
            base_seed=int(args.seed),
        )
        write_dataframe_csv(os.path.join(raw_dir, "convergence_per_case.csv"), convergence_frame)

        convergence_metrics = select_metric_columns(
            convergence_frame,
            exclude=["case_id", "path", "slice_count", "sampling_strategy"],
        )
        ensure_metric_names_exist([args.primary_metric, args.secondary_metric, *headline_metrics], convergence_metrics, "convergence")
        convergence_summary = summarize_grouped_metrics(
            convergence_frame,
            group_cols=["slice_count"],
            metric_cols=convergence_metrics,
        )
        write_dataframe_csv(os.path.join(summary_dir, "convergence_summary_long.csv"), convergence_summary)

        convergence_table = build_formatted_metric_table(
            convergence_summary,
            row_key="slice_count",
            metrics=headline_metrics,
            row_label=text("row_slices"),
        )
        write_dataframe_csv(os.path.join(summary_dir, "convergence_table.csv"), convergence_table)
        convergence_tex = os.path.join(table_dir, "convergence_table.tex")
        write_dataframe_latex(convergence_tex, convergence_table)
        generated_table_paths.append(convergence_tex)

        save_dual_metric_plot(
            convergence_summary,
            x_col="slice_count",
            x_label=text("x_slices"),
            primary_metric=args.primary_metric,
            secondary_metric=args.secondary_metric,
            title_prefix=text("title_information_convergence"),
            out_path=plot_output_path(plot_dir, "convergence_dual_metrics"),
        )
        study_index["convergence"] = {
            "raw_csv": normalize_case_path(os.path.join(raw_dir, "convergence_per_case.csv")),
            "summary_csv": normalize_case_path(os.path.join(summary_dir, "convergence_summary_long.csv")),
            "table_tex": normalize_case_path(convergence_tex),
        }

    if "noise" in studies:
        noise_frame = run_noise_study(
            cases,
            runner=runner,
            noise_levels=noise_levels,
            num_slices=int(args.noise_slices),
            fixed_strategy=args.fixed_sampling_strategy,
            thickness_vox=float(args.thickness_vox),
            base_seed=int(args.seed),
        )
        write_dataframe_csv(os.path.join(raw_dir, "noise_per_case.csv"), noise_frame)

        noise_metrics = select_metric_columns(
            noise_frame,
            exclude=["case_id", "path", "noise_std", "slice_count", "sampling_strategy"],
        )
        ensure_metric_names_exist([args.primary_metric, args.secondary_metric, *headline_metrics], noise_metrics, "noise")
        noise_summary = summarize_grouped_metrics(
            noise_frame,
            group_cols=["noise_std"],
            metric_cols=noise_metrics,
        )
        write_dataframe_csv(os.path.join(summary_dir, "noise_summary_long.csv"), noise_summary)

        noise_table = build_formatted_metric_table(
            noise_summary,
            row_key="noise_std",
            metrics=headline_metrics,
            row_label=text("row_noise_std"),
        )
        write_dataframe_csv(os.path.join(summary_dir, "noise_table.csv"), noise_table)
        noise_tex = os.path.join(table_dir, "noise_table.tex")
        write_dataframe_latex(noise_tex, noise_table)
        generated_table_paths.append(noise_tex)

        save_dual_metric_plot(
            noise_summary,
            x_col="noise_std",
            x_label=text("x_noise_std"),
            primary_metric=args.primary_metric,
            secondary_metric=args.secondary_metric,
            title_prefix=text("title_noise_stability"),
            out_path=plot_output_path(plot_dir, "noise_dual_metrics"),
        )
        study_index["noise"] = {
            "raw_csv": normalize_case_path(os.path.join(raw_dir, "noise_per_case.csv")),
            "summary_csv": normalize_case_path(os.path.join(summary_dir, "noise_summary_long.csv")),
            "table_tex": normalize_case_path(noise_tex),
        }

    if "sampling" in studies:
        sampling_frame, variability_frame = run_sampling_study(
            cases,
            runner=runner,
            slice_counts=sampling_slices,
            repeats=int(args.sampling_repeats),
            thickness_vox=float(args.thickness_vox),
            base_seed=int(args.seed),
        )
        write_dataframe_csv(os.path.join(raw_dir, "sampling_per_case.csv"), sampling_frame)
        write_dataframe_csv(os.path.join(raw_dir, "sampling_within_case_variability.csv"), variability_frame)

        sampling_metrics = select_metric_columns(
            sampling_frame,
            exclude=["case_id", "path", "slice_count", "repeat_index", "geometry_seed"],
        )
        ensure_metric_names_exist([args.primary_metric, args.secondary_metric, *headline_metrics], sampling_metrics, "sampling")
        sampling_summary = summarize_grouped_metrics(
            sampling_frame,
            group_cols=["slice_count"],
            metric_cols=sampling_metrics,
        )
        variability_metrics = select_metric_columns(
            variability_frame,
            exclude=["case_id", "path", "slice_count"],
        )
        sampling_variability_summary = summarize_grouped_metrics(
            variability_frame,
            group_cols=["slice_count"],
            metric_cols=variability_metrics,
        )
        write_dataframe_csv(os.path.join(summary_dir, "sampling_summary_long.csv"), sampling_summary)
        write_dataframe_csv(
            os.path.join(summary_dir, "sampling_variability_summary_long.csv"),
            sampling_variability_summary,
        )

        sampling_table = build_sampling_table(
            sampling_summary,
            sampling_variability_summary,
            row_key="slice_count",
            metrics=headline_metrics,
            row_label=text("row_slices"),
        )
        write_dataframe_csv(os.path.join(summary_dir, "sampling_table.csv"), sampling_table)
        sampling_tex = os.path.join(table_dir, "sampling_table.tex")
        write_dataframe_latex(sampling_tex, sampling_table)
        generated_table_paths.append(sampling_tex)

        save_sampling_plot(
            sampling_summary,
            sampling_variability_summary,
            primary_metric=args.primary_metric,
            out_path=plot_output_path(plot_dir, "sampling_stability"),
        )
        study_index["sampling"] = {
            "raw_csv": normalize_case_path(os.path.join(raw_dir, "sampling_per_case.csv")),
            "variability_csv": normalize_case_path(os.path.join(raw_dir, "sampling_within_case_variability.csv")),
            "summary_csv": normalize_case_path(os.path.join(summary_dir, "sampling_summary_long.csv")),
            "table_tex": normalize_case_path(sampling_tex),
        }

    if "sensitivity" in studies:
        sensitivity_frame = run_sensitivity_study(
            cases,
            runner=runner,
            noise_levels=sensitivity_noise_levels,
            num_slices=int(args.sensitivity_slices),
            fixed_strategy=args.fixed_sampling_strategy,
            thickness_vox=float(args.thickness_vox),
            base_seed=int(args.seed),
        )
        write_dataframe_csv(os.path.join(raw_dir, "sensitivity_per_case.csv"), sensitivity_frame)

        sensitivity_metrics = [
            "input_delta_rms_known",
            "output_delta_rmse_all",
            "output_delta_rmse_missing",
            "stability_ratio_all",
            "stability_ratio_missing",
        ]
        ensure_metric_names_exist(sensitivity_metrics, sensitivity_frame.columns.tolist(), "sensitivity")
        sensitivity_summary = summarize_grouped_metrics(
            sensitivity_frame,
            group_cols=["noise_std"],
            metric_cols=sensitivity_metrics,
        )
        write_dataframe_csv(os.path.join(summary_dir, "sensitivity_summary_long.csv"), sensitivity_summary)

        sensitivity_table = build_formatted_metric_table(
            sensitivity_summary,
            row_key="noise_std",
            metrics=["stability_ratio_missing", "stability_ratio_all", "output_delta_rmse_missing"],
            row_label=text("row_noise_std"),
        )
        write_dataframe_csv(os.path.join(summary_dir, "sensitivity_table.csv"), sensitivity_table)
        sensitivity_tex = os.path.join(table_dir, "sensitivity_table.tex")
        write_dataframe_latex(sensitivity_tex, sensitivity_table)
        generated_table_paths.append(sensitivity_tex)

        save_sensitivity_plot(
            sensitivity_summary,
            out_path=plot_output_path(plot_dir, "sensitivity_stability"),
        )
        study_index["sensitivity"] = {
            "raw_csv": normalize_case_path(os.path.join(raw_dir, "sensitivity_per_case.csv")),
            "summary_csv": normalize_case_path(os.path.join(summary_dir, "sensitivity_summary_long.csv")),
            "table_tex": normalize_case_path(sensitivity_tex),
        }

    if "robustness" in studies:
        model_specs = resolve_model_specs(args)
        robustness_frame, robustness_spread_frame = run_robustness_study(
            cases,
            model_specs=model_specs,
            hard_constraint=args.hard_constraint,
            num_slices=int(args.robustness_slices),
            fixed_strategy=args.fixed_sampling_strategy,
            thickness_vox=float(args.thickness_vox),
            base_seed=int(args.seed),
        )
        if not robustness_frame.empty:
            write_dataframe_csv(os.path.join(raw_dir, "robustness_per_case.csv"), robustness_frame)
            write_dataframe_csv(os.path.join(raw_dir, "robustness_case_spread.csv"), robustness_spread_frame)

            robustness_metrics = select_metric_columns(
                robustness_frame,
                exclude=["case_id", "path", "model_label", "weight_path", "slice_count"],
            )
            ensure_metric_names_exist([args.primary_metric, *headline_metrics], robustness_metrics, "robustness")
            robustness_summary = summarize_grouped_metrics(
                robustness_frame,
                group_cols=["model_label"],
                metric_cols=robustness_metrics,
            )
            write_dataframe_csv(os.path.join(summary_dir, "robustness_summary_long.csv"), robustness_summary)

            robustness_table = build_formatted_metric_table(
                robustness_summary,
                row_key="model_label",
                metrics=headline_metrics,
                row_label=text("row_checkpoint"),
            )
            write_dataframe_csv(os.path.join(summary_dir, "robustness_table.csv"), robustness_table)
            robustness_tex = os.path.join(table_dir, "robustness_table.tex")
            write_dataframe_latex(robustness_tex, robustness_table)
            generated_table_paths.append(robustness_tex)

            save_robustness_plot(
                robustness_frame,
                robustness_summary,
                primary_metric=args.primary_metric,
                out_path=plot_output_path(plot_dir, "robustness_primary_metric"),
            )
            study_index["robustness"] = {
                "raw_csv": normalize_case_path(os.path.join(raw_dir, "robustness_per_case.csv")),
                "summary_csv": normalize_case_path(os.path.join(summary_dir, "robustness_summary_long.csv")),
                "table_tex": normalize_case_path(robustness_tex),
            }
        else:
            study_index["robustness"] = {"skipped": "Need at least two --robustness_weights."}

    save_overview_plot(
        out_path=plot_output_path(plot_dir, "thesis_overview"),
        convergence_summary=convergence_summary,
        noise_summary=noise_summary,
        sampling_variability_summary=sampling_variability_summary,
        sensitivity_summary=sensitivity_summary,
        primary_metric=args.primary_metric,
    )

    save_json(os.path.join(args.out, "study_index.json"), study_index)
    save_json(
        os.path.join(args.out, "study_config.json"),
        {
            **vars(args),
            "selected_cases": [case.case_id for case in cases],
            "resolved_weights": normalize_case_path(args.weights),
            "device": str(device),
        },
    )
    save_json(
        os.path.join(args.out, "reproducibility.json"),
        build_repro_metadata(
            vars(args),
            split_manifest_path=os.path.join(args.out, "split_manifest.json") if split_manifest is not None else None,
            extra={
                "weights": normalize_case_path(args.weights),
                "selected_cases": [case.case_id for case in cases],
            },
            cwd=ROOT,
        ),
    )
    combine_latex_tables(table_dir, generated_table_paths)

    print("Finished empirical regularization study.")
    print(f"Cases: {len(cases)}")
    print(f"Studies: {', '.join(studies)}")
    print(f"Output: {normalize_case_path(args.out)}")


if __name__ == "__main__":
    main()
