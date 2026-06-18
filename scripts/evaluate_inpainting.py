# scripts/evaluate_inpainting.py
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader

from inpainting3d.metrics import compute_metrics_per_case
from inpainting3d.repro import build_repro_metadata
from inpainting3d.splits import (
    filter_to_known_files,
    get_split_files,
    load_split_manifest,
    normalize_case_path,
    resolve_case_paths,
    write_split_manifest,
)
from inpainting3d.stats import summarize_case_metrics, summary_rows_from_nested
from inpainting3d.utils import ensure_dir, strip_dataparallel_prefix, torch_load_weights_compat
from scripts.train_inpainting import LiverInpaintingDataset, build_model, clamp_prediction


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


def write_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Write row dictionaries to a CSV file.
    
    Parameters
    ----------
    path : str
        Filesystem path to the input artifact.
    rows : List[Dict[str, Any]]
        Row dictionaries that should be written or summarized.
    
    Returns
    -------
    None
        The CSV artifact is written to disk.
    """
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_str = str(key)
            if key_str not in seen:
                seen.add(key_str)
                fieldnames.append(key_str)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_limit_cases(files: List[str], max_cases: int | None) -> List[str]:
    """
    Optionally limit the number of selected cases.
    
    Parameters
    ----------
    files : List[str]
        Sequence of input case files.
    max_cases : int | None
        Requested max cases.
    
    Returns
    -------
    List[str]
        Selected case paths after applying the optional limit.
    """
    if max_cases is None:
        return files
    limit = int(max_cases)
    if limit <= 0:
        raise ValueError("max_cases must be >= 1 when provided.")
    return files[:limit]


def resolve_eval_files(args: argparse.Namespace) -> Tuple[List[str], Dict[str, object] | None]:
    """
    Resolve the evaluation cases selected by the command-line arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed to the helper or command.
    
    Returns
    -------
    Tuple[List[str], Dict[str, object] | None]
        Resolved value or selection.
    """
    available_files = resolve_case_paths(args.data)
    if not available_files:
        raise RuntimeError("Keine Dateien gefunden.")

    if not args.split_file:
        if not args.allow_all_data_eval:
            raise RuntimeError(
                "Standalone evaluation requires a persisted split manifest by default. "
                "Pass --split_file <manifest.json> to evaluate only the fixed test split, "
                "or explicitly opt into the legacy unsafe behavior with --allow_all_data_eval."
            )
        return maybe_limit_cases(available_files, args.max_cases), None

    manifest = load_split_manifest(args.split_file)
    selected_files = filter_to_known_files(get_split_files(manifest, args.split), available_files)
    selected_files = maybe_limit_cases(selected_files, args.max_cases)
    if not selected_files:
        raise RuntimeError(f"Split '{args.split}' enthaelt keine Testfaelle.")

    manifest["selected_splits"] = {"eval": args.split}
    write_split_manifest(manifest, os.path.join(args.out, "split_manifest.json"))
    return selected_files, manifest


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
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out", type=str, default="./runs/eval")
    ap.add_argument("--model", type=str, default="partial", choices=["baseline", "partial"])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--init_feat", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--slices", type=int, default=None, help="Fixe Anzahl Slices fuer die Evaluation.")
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
    ap.add_argument("--split_file", type=str, default="")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument(
        "--allow_all_data_eval",
        action="store_true",
        help="Explicitly allow evaluation on the entire data glob when no split manifest is provided.",
    )
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--no_hard_constraint", dest="hard_constraint", action="store_false")
    ap.set_defaults(hard_constraint=True)
    args = ap.parse_args()

    ensure_dir(args.out)

    slices_min = int(args.slices_min)
    slices_max = int(args.slices_max)
    if args.slices is not None:
        slices_min = int(args.slices)
        slices_max = int(args.slices)

    eval_files, split_manifest = resolve_eval_files(args)
    ds = LiverInpaintingDataset(
        eval_files,
        dim=args.dim,
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
        is_train=False,
        seed=args.seed,
        return_meta=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    default_workers = 0 if sys.platform.startswith("win") else 2
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=default_workers,
        pin_memory=pin_memory,
    )

    model = build_model(args.model, init_feat=args.init_feat).to(device)
    state = torch_load_weights_compat(args.weights, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and any(isinstance(value, torch.Tensor) for value in state.values()):
        model.load_state_dict(strip_dataparallel_prefix(state), strict=True)
    else:
        raise RuntimeError("Gewichte haben unerwartetes Format. Erwartet state_dict.")

    rows: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = x[:, 1:2, ...]
            sparse = x[:, 0:1, ...]
            pred = clamp_prediction(model(x), sparse=sparse, mask=mask, hard_constraint=args.hard_constraint)

            case_ids = [str(item) for item in meta["case_id"]]
            paths = [str(item) for item in meta["path"]]
            batch_rows = compute_metrics_per_case(pred, y, known_mask=mask, data_range=1.0, case_ids=case_ids, paths=paths)
            for row in batch_rows:
                row["index"] = len(rows)
                row["split"] = args.split if args.split_file else "all"
                row["label"] = args.label or args.model
                rows.append(row)

    if not rows:
        raise RuntimeError("Es konnten keine Fallmetriken berechnet werden.")

    write_rows_csv(os.path.join(args.out, "metrics_per_case.csv"), rows)

    nested_summary = summarize_case_metrics(rows)
    write_rows_csv(os.path.join(args.out, "metrics_summary.csv"), summary_rows_from_nested(nested_summary))

    payload = {
        "summary": nested_summary,
        "settings": {
            "dim": int(args.dim),
            "batch": int(args.batch),
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
            "split": args.split if args.split_file else "all",
            "model": str(args.model),
            "label": args.label or args.model,
            "hard_constraint": bool(args.hard_constraint),
            "weights": normalize_case_path(args.weights),
            "num_cases": int(len(rows)),
            "max_cases": None if args.max_cases is None else int(args.max_cases),
        },
    }
    if split_manifest is not None:
        payload["split_manifest"] = {
            "path": normalize_case_path(os.path.join(args.out, "split_manifest.json")),
            "counts": split_manifest.get("counts", {}),
        }
    save_json(os.path.join(args.out, "metrics_summary.json"), payload)
    save_json(
        os.path.join(args.out, "reproducibility.json"),
        build_repro_metadata(
            vars(args),
            split_manifest_path=os.path.join(args.out, "split_manifest.json") if split_manifest is not None else None,
            extra={"weights": normalize_case_path(args.weights), "num_cases": int(len(rows))},
            cwd=ROOT,
        ),
    )

    print("Summary:")
    for metric_name, metric_stats in nested_summary.items():
        mean = float(metric_stats["mean"])
        median = float(metric_stats["median"])
        ci_low = float(metric_stats["ci95_low"])
        ci_high = float(metric_stats["ci95_high"])
        print(f"  {metric_name:16s} mean={mean:.6f} median={median:.6f} ci95=[{ci_low:.6f}, {ci_high:.6f}]")


if __name__ == "__main__":
    main()
