from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch

from inpainting3d.metrics import compute_metrics
from inpainting3d.splits import filter_to_known_files, get_split_files, load_split_manifest, normalize_case_path, resolve_case_paths
from inpainting3d.utils import ensure_dir, strip_dataparallel_prefix, torch_load_weights_compat
from scripts.train_inpainting import LiverInpaintingDataset, build_model, clamp_prediction


def resolve_eval_files(data_glob: str, split_file: str, split: str) -> List[str]:
    """
    Resolve the evaluation cases selected by the command-line arguments.
    
    Parameters
    ----------
    data_glob : str
        Glob pattern used to discover input case files.
    split_file : str
        Path to the split manifest file.
    split : str
        Name of the requested data split.
    
    Returns
    -------
    List[str]
        Resolved value or selection.
    """
    files = resolve_case_paths(data_glob)
    if not files:
        raise RuntimeError("Keine Dateien fuer das angegebene Daten-Glob gefunden.")
    if not split_file:
        return files
    manifest = load_split_manifest(split_file)
    selected = filter_to_known_files(get_split_files(manifest, split), files)
    if not selected:
        raise RuntimeError(f"Split '{split}' enthaelt keine Faelle.")
    return selected


def load_case_sample(
    *,
    files: List[str],
    case_id: str,
    dim: int,
    init_feat: int,
    slices_min: int,
    slices_max: int,
    slice_sampling: str,
    slice_mean: float | None,
    slice_std: float | None,
    slice_geometry: str,
    slice_axis: List[float] | None,
    slice_axis_jitter_deg: float,
    slice_fan_half_angle_deg: float,
    slice_elevation_jitter_deg: float,
    slice_sweep_jitter_deg: float,
    probe_pos_sigma_vox: float,
    probe_depth_sigma_vox: float,
    probe_tilt_sigma: float,
    thickness_vox: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Load the data sample associated with a reported evaluation case.
    
    Parameters
    ----------
    files : List[str]
        Sequence of input case files.
    case_id : str
        Case identifier.
    dim : int
        Target cubic side length of the processed volume.
    init_feat : int
        Base number of feature channels in the first stage.
    slices_min : int
        Lower bound of the sampled slice count.
    slices_max : int
        Upper bound of the sampled slice count.
    slice_sampling : str
        Rule used to sample the number of slices.
    slice_mean : float | None
        Mean of the normal slice-count distribution.
    slice_std : float | None
        Standard deviation of the normal slice-count distribution.
    slice_geometry : str
        Geometry model used for sparse slice sampling.
    slice_axis : List[float] | None
        Preferred axis used when sampling slice geometry.
    slice_axis_jitter_deg : float
        Standard deviation of dominant-axis jitter in degrees.
    slice_fan_half_angle_deg : float
        Half opening angle of the sampled fan in degrees.
    slice_elevation_jitter_deg : float
        Standard deviation of out-of-plane jitter in degrees.
    slice_sweep_jitter_deg : float
        Standard deviation of in-plane sweep jitter in degrees.
    probe_pos_sigma_vox : float
        Standard deviation of probe-position jitter in voxels.
    probe_depth_sigma_vox : float
        Standard deviation of depth jitter in voxels.
    probe_tilt_sigma : float
        Standard deviation of probe tilt perturbations.
    thickness_vox : float
        Half thickness of each sampled slice plane in voxels.
    seed : int
        Random seed used for reproducible sampling.
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]
        Loaded data structure.
    """
    dataset = LiverInpaintingDataset(
        files,
        dim=dim,
        slices_min=slices_min,
        slices_max=slices_max,
        slice_sampling=slice_sampling,
        slice_mean=slice_mean,
        slice_std=slice_std,
        slice_geometry=slice_geometry,
        slice_axis=slice_axis,
        slice_axis_jitter_deg=slice_axis_jitter_deg,
        slice_fan_half_angle_deg=slice_fan_half_angle_deg,
        slice_elevation_jitter_deg=slice_elevation_jitter_deg,
        slice_sweep_jitter_deg=slice_sweep_jitter_deg,
        probe_pos_sigma_vox=probe_pos_sigma_vox,
        probe_depth_sigma_vox=probe_depth_sigma_vox,
        probe_tilt_sigma=probe_tilt_sigma,
        thickness_vox=thickness_vox,
        is_train=False,
        seed=seed,
        return_meta=True,
    )

    for idx in range(len(dataset)):
        x, y, meta = dataset[idx]
        if str(meta["case_id"]) == case_id:
            meta["dataset_index"] = int(idx)
            meta["init_feat"] = int(init_feat)
            return x, y, meta
    raise RuntimeError(f"Fall '{case_id}' wurde im ausgewaehlten Split nicht gefunden.")


def run_prediction(
    x: torch.Tensor,
    *,
    weights: str,
    model_kind: str,
    init_feat: int,
    hard_constraint: bool,
    device: torch.device,
) -> np.ndarray:
    """
    Run a model prediction for one loaded case.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor or array.
    weights : str
        Checkpoint path or weighting coefficients used by the helper.
    model_kind : str
        Architecture identifier used to build a model.
    init_feat : int
        Base number of feature channels in the first stage.
    hard_constraint : bool
        Whether known voxels should be enforced exactly at the output.
    device : torch.device
        Torch device on which tensors should be created or evaluated.
    
    Returns
    -------
    np.ndarray
        Output produced by the model or workflow.
    """
    model = build_model(model_kind, init_feat=init_feat).to(device)
    state = torch_load_weights_compat(weights, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not (isinstance(state, dict) and any(isinstance(value, torch.Tensor) for value in state.values())):
        raise RuntimeError(f"Gewichte '{weights}' haben kein erwartetes state_dict-Format.")
    model.load_state_dict(strip_dataparallel_prefix(state), strict=True)
    model.eval()

    x_batched = x.unsqueeze(0).to(device)
    sparse = x_batched[:, 0:1, ...]
    mask = x_batched[:, 1:2, ...]
    with torch.no_grad():
        pred = clamp_prediction(model(x_batched), sparse=sparse, mask=mask, hard_constraint=hard_constraint)
    return pred[0, 0].detach().cpu().numpy().astype(np.float32)


def to_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Compute a compact metric dictionary for displayed predictions.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted reconstruction tensor or array.
    target : np.ndarray
        Reference tensor or array used as supervision.
    mask : np.ndarray
        Binary mask that marks valid or selected voxels.
    
    Returns
    -------
    dict[str, float]
        Compact metric dictionary for the provided prediction.
    """
    pred_t = torch.from_numpy(pred[None, None]).float()
    target_t = torch.from_numpy(target[None, None]).float()
    mask_t = torch.from_numpy(mask[None, None]).float()
    metrics = compute_metrics(pred_t, target_t, known_mask=mask_t, data_range=1.0)
    return {key: float(value) for key, value in metrics.items()}


def _views(vol: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Return the set of orthogonal volume views used for visualization.
    
    Parameters
    ----------
    vol : np.ndarray
        Input volume array.
    
    Returns
    -------
    List[Tuple[str, np.ndarray]]
        Orthogonal view names paired with their 2D slices.
    """
    mid_d = vol.shape[0] // 2
    mid_h = vol.shape[1] // 2
    mid_w = vol.shape[2] // 2
    return [
        ("Axial", vol[mid_d]),
        ("Coronal", vol[:, mid_h, :]),
        ("Sagittal", vol[:, :, mid_w]),
    ]


def save_multiview_predictions(
    gt: np.ndarray,
    sparse: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    out_path: str,
    *,
    label_a: str,
    label_b: str,
) -> None:
    """
    Save multi-view prediction figures.
    
    Parameters
    ----------
    gt : np.ndarray
        Ground-truth tensor or array.
    sparse : np.ndarray
        Sparse observation tensor or array.
    pred_a : np.ndarray
        Prediction generated by the first model.
    pred_b : np.ndarray
        Prediction generated by the second model.
    out_path : str
        Destination path for the written artifact.
    label_a : str
        Display label of the first model or prediction.
    label_b : str
        Display label of the second model or prediction.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    gt_views = _views(gt)
    sparse_views = _views(sparse)
    a_views = _views(pred_a)
    b_views = _views(pred_b)
    vmax = float(max(np.quantile(gt, 0.995), np.quantile(pred_a, 0.995), np.quantile(pred_b, 0.995), 1e-3))

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for row, ((name, gt_img), (_, sparse_img), (_, a_img), (_, b_img)) in enumerate(zip(gt_views, sparse_views, a_views, b_views)):
        panels = [
            (f"{name} GT", gt_img),
            (f"{name} Sparse", sparse_img),
            (f"{name} {label_a}", a_img),
            (f"{name} {label_b}", b_img),
        ]
        for col, (title, img) in enumerate(panels):
            axes[row, col].imshow(img.T, cmap="bone", origin="lower", vmin=0.0, vmax=vmax)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.normpath(out_path), dpi=220)
    plt.close(fig)


def save_multiview_errors(
    gt: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    out_path: str,
    *,
    label_a: str,
    label_b: str,
) -> None:
    """
    Save multi-view error figures.
    
    Parameters
    ----------
    gt : np.ndarray
        Ground-truth tensor or array.
    pred_a : np.ndarray
        Prediction generated by the first model.
    pred_b : np.ndarray
        Prediction generated by the second model.
    out_path : str
        Destination path for the written artifact.
    label_a : str
        Display label of the first model or prediction.
    label_b : str
        Display label of the second model or prediction.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    err_a = np.abs(gt - pred_a).astype(np.float32)
    err_b = np.abs(gt - pred_b).astype(np.float32)
    delta = err_a - err_b
    err_a_views = _views(err_a)
    err_b_views = _views(err_b)
    delta_views = _views(delta)
    vmax_err = float(max(np.quantile(err_a, 0.995), np.quantile(err_b, 0.995), 1e-6))
    vmax_delta = float(max(abs(np.quantile(delta, 0.01)), abs(np.quantile(delta, 0.99)), 1e-6))

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for row, ((name, a_img), (_, b_img), (_, d_img)) in enumerate(zip(err_a_views, err_b_views, delta_views)):
        panels = [
            (f"{name} |GT - {label_a}|", a_img, "inferno", 0.0, vmax_err),
            (f"{name} |GT - {label_b}|", b_img, "inferno", 0.0, vmax_err),
            (f"{name} Error Delta ({label_a} - {label_b})", d_img, "coolwarm", -vmax_delta, vmax_delta),
        ]
        for col, (title, img, cmap, vmin, vmax) in enumerate(panels):
            axes[row, col].imshow(img.T, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.normpath(out_path), dpi=220)
    plt.close(fig)


def save_mip_comparison(
    gt: np.ndarray,
    sparse: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    out_path: str,
    *,
    label_a: str,
    label_b: str,
) -> None:
    """
    Save maximum-intensity-projection comparison figures.
    
    Parameters
    ----------
    gt : np.ndarray
        Ground-truth tensor or array.
    sparse : np.ndarray
        Sparse observation tensor or array.
    pred_a : np.ndarray
        Prediction generated by the first model.
    pred_b : np.ndarray
        Prediction generated by the second model.
    out_path : str
        Destination path for the written artifact.
    label_a : str
        Display label of the first model or prediction.
    label_b : str
        Display label of the second model or prediction.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    projections = [
        ("Axial", lambda x: x.max(axis=0)),
        ("Coronal", lambda x: x.max(axis=1)),
        ("Sagittal", lambda x: x.max(axis=2)),
    ]
    volumes = [
        ("GT", gt),
        ("Sparse", sparse),
        (label_a, pred_a),
        (label_b, pred_b),
    ]
    vmax = float(max(np.quantile(gt, 0.995), np.quantile(pred_a, 0.995), np.quantile(pred_b, 0.995), 1e-3))

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for row, (proj_name, projector) in enumerate(projections):
        for col, (label, vol) in enumerate(volumes):
            axes[row, col].imshow(projector(vol).T, cmap="bone", origin="lower", vmin=0.0, vmax=vmax)
            axes[row, col].set_title(f"{proj_name} {label} MIP")
            axes[row, col].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.normpath(out_path), dpi=220)
    plt.close(fig)


def main() -> None:
    """
    Execute the command-line entry point for this script.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--split_file", type=str, default="")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--case_id", type=str, required=True)
    ap.add_argument("--weights_a", type=str, required=True)
    ap.add_argument("--weights_b", type=str, required=True)
    ap.add_argument("--model_a", type=str, required=True, choices=["baseline", "partial"])
    ap.add_argument("--model_b", type=str, required=True, choices=["baseline", "partial"])
    ap.add_argument("--label_a", type=str, default="Model A")
    ap.add_argument("--label_b", type=str, default="Model B")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--init_feat", type=int, default=32)
    ap.add_argument("--slices", type=int, default=None)
    ap.add_argument("--slices_min", type=int, default=4)
    ap.add_argument("--slices_max", type=int, default=32)
    ap.add_argument("--slice_sampling", type=str, default="uniform", choices=["uniform", "normal"])
    ap.add_argument("--slice_mean", type=float, default=None)
    ap.add_argument("--slice_std", type=float, default=None)
    ap.add_argument("--slice_geometry", type=str, default="random", choices=["random", "fibonacci", "ultrasound_fan", "ultrasound_probe"])
    ap.add_argument("--slice_axis", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--slice_axis_jitter_deg", type=float, default=25.0)
    ap.add_argument("--slice_fan_half_angle_deg", type=float, default=35.0)
    ap.add_argument("--slice_elevation_jitter_deg", type=float, default=6.0)
    ap.add_argument("--slice_sweep_jitter_deg", type=float, default=2.5)
    ap.add_argument("--probe_pos_sigma_vox", type=float, default=1.0)
    ap.add_argument("--probe_depth_sigma_vox", type=float, default=6.0)
    ap.add_argument("--probe_tilt_sigma", type=float, default=0.08)
    ap.add_argument("--thickness_vox", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no_hard_constraint_a", dest="hard_constraint_a", action="store_false")
    ap.add_argument("--no_hard_constraint_b", dest="hard_constraint_b", action="store_false")
    ap.set_defaults(hard_constraint_a=True, hard_constraint_b=True)
    args = ap.parse_args()

    ensure_dir(args.out)

    slices_min = int(args.slices_min)
    slices_max = int(args.slices_max)
    if args.slices is not None:
        slices_min = int(args.slices)
        slices_max = int(args.slices)

    files = resolve_eval_files(args.data, args.split_file, args.split)
    x, y, meta = load_case_sample(
        files=files,
        case_id=args.case_id,
        dim=args.dim,
        init_feat=args.init_feat,
        slices_min=slices_min,
        slices_max=slices_max,
        slice_sampling=args.slice_sampling,
        slice_mean=args.slice_mean,
        slice_std=args.slice_std,
        slice_geometry=args.slice_geometry,
        slice_axis=args.slice_axis,
        slice_axis_jitter_deg=args.slice_axis_jitter_deg,
        slice_fan_half_angle_deg=args.slice_fan_half_angle_deg,
        slice_elevation_jitter_deg=args.slice_elevation_jitter_deg,
        slice_sweep_jitter_deg=args.slice_sweep_jitter_deg,
        probe_pos_sigma_vox=args.probe_pos_sigma_vox,
        probe_depth_sigma_vox=args.probe_depth_sigma_vox,
        probe_tilt_sigma=args.probe_tilt_sigma,
        thickness_vox=args.thickness_vox,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_a = run_prediction(
        x,
        weights=args.weights_a,
        model_kind=args.model_a,
        init_feat=args.init_feat,
        hard_constraint=bool(args.hard_constraint_a),
        device=device,
    )
    pred_b = run_prediction(
        x,
        weights=args.weights_b,
        model_kind=args.model_b,
        init_feat=args.init_feat,
        hard_constraint=bool(args.hard_constraint_b),
        device=device,
    )

    gt = y[0].numpy().astype(np.float32)
    sparse = x[0].numpy().astype(np.float32)
    mask = x[1].numpy().astype(np.float32)

    save_multiview_predictions(
        gt,
        sparse,
        pred_a,
        pred_b,
        os.path.join(args.out, "comparison_views.png"),
        label_a=args.label_a,
        label_b=args.label_b,
    )
    save_multiview_errors(
        gt,
        pred_a,
        pred_b,
        os.path.join(args.out, "comparison_errors.png"),
        label_a=args.label_a,
        label_b=args.label_b,
    )
    save_mip_comparison(
        gt,
        sparse,
        pred_a,
        pred_b,
        os.path.join(args.out, "comparison_mips.png"),
        label_a=args.label_a,
        label_b=args.label_b,
    )

    metrics = {
        args.label_a: to_metrics(pred_a, gt, mask),
        args.label_b: to_metrics(pred_b, gt, mask),
    }
    payload = {
        "case": {
            "case_id": str(meta["case_id"]),
            "path": normalize_case_path(str(meta["path"])),
            "dataset_index": int(meta["dataset_index"]),
            "num_slices": int(meta["num_slices"]),
            "seed": int(meta["seed"]),
            "split": str(args.split),
        },
        "settings": {
            "dim": int(args.dim),
            "init_feat": int(args.init_feat),
            "slices_min": int(slices_min),
            "slices_max": int(slices_max),
            "slice_sampling": str(args.slice_sampling),
            "slice_mean": None if args.slice_mean is None else float(args.slice_mean),
            "slice_std": None if args.slice_std is None else float(args.slice_std),
            "slice_geometry": str(args.slice_geometry),
            "slice_axis": None if args.slice_axis is None else [float(v) for v in args.slice_axis],
            "slice_axis_jitter_deg": float(args.slice_axis_jitter_deg),
            "slice_fan_half_angle_deg": float(args.slice_fan_half_angle_deg),
            "slice_elevation_jitter_deg": float(args.slice_elevation_jitter_deg),
            "slice_sweep_jitter_deg": float(args.slice_sweep_jitter_deg),
            "probe_pos_sigma_vox": float(args.probe_pos_sigma_vox),
            "probe_depth_sigma_vox": float(args.probe_depth_sigma_vox),
            "probe_tilt_sigma": float(args.probe_tilt_sigma),
            "thickness_vox": float(args.thickness_vox),
            "seed": int(args.seed),
        },
        "models": {
            args.label_a: {
                "kind": str(args.model_a),
                "weights": normalize_case_path(args.weights_a),
                "hard_constraint": bool(args.hard_constraint_a),
            },
            args.label_b: {
                "kind": str(args.model_b),
                "weights": normalize_case_path(args.weights_b),
                "hard_constraint": bool(args.hard_constraint_b),
            },
        },
        "metrics": metrics,
    }
    with open(os.path.join(args.out, "metrics_comparison.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
