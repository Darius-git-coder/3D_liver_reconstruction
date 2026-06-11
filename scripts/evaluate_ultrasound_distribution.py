from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader

from inpainting3d.metrics import compute_metrics_per_case
from inpainting3d.repro import build_repro_metadata
from inpainting3d.splits import normalize_case_path
from inpainting3d.stats import summarize_case_metrics, summary_rows_from_nested
from inpainting3d.utils import ensure_dir, strip_dataparallel_prefix, torch_load_weights_compat
from scripts.evaluate_inpainting import resolve_eval_files, save_json, write_rows_csv
from scripts.train_inpainting import LiverInpaintingDataset, build_model, clamp_prediction


@dataclass
class ModelSpec:
    label: str
    kind: str
    weights: str


def default_label_from_path(path: str, fallback_model: str, index: int) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = os.path.splitext(stem)[0]
    stem = stem.strip()
    if stem:
        return stem
    return f"{fallback_model}_{index}"


def parse_model_specs(args: argparse.Namespace) -> List[ModelSpec]:
    weights = list(args.weights or [])
    models = list(args.model or [])
    labels = list(args.label or [])
    if not weights:
        raise RuntimeError("At least one --weights argument is required.")
    if len(weights) != len(models):
        raise RuntimeError("Each --weights entry requires a matching --model entry.")
    if labels and len(labels) != len(weights):
        raise RuntimeError("Optional --label entries must match the number of --weights entries.")

    specs: List[ModelSpec] = []
    for index, (weight_path, model_kind) in enumerate(zip(weights, models), start=1):
        label = labels[index - 1] if labels else default_label_from_path(weight_path, model_kind, index)
        specs.append(ModelSpec(label=str(label), kind=str(model_kind), weights=str(weight_path)))
    return specs


def resolve_eval_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    return [int(args.seed_start) + offset for offset in range(int(args.num_repeats))]


def build_dataset(
    files: Sequence[str],
    *,
    args: argparse.Namespace,
    seed: int,
) -> LiverInpaintingDataset:
    slices_min = int(args.slices_min)
    slices_max = int(args.slices_max)
    if args.slices is not None:
        slices_min = int(args.slices)
        slices_max = int(args.slices)

    return LiverInpaintingDataset(
        list(files),
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
        seed=seed,
        return_meta=True,
    )


def load_model_spec(spec: ModelSpec, *, init_feat: int, device: torch.device) -> torch.nn.Module:
    model = build_model(spec.kind, init_feat=init_feat).to(device)
    state = torch_load_weights_compat(spec.weights, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not (isinstance(state, dict) and any(isinstance(value, torch.Tensor) for value in state.values())):
        raise RuntimeError(f"Gewichte '{spec.weights}' haben kein erwartetes state_dict-Format.")
    model.load_state_dict(strip_dataparallel_prefix(state), strict=True)
    model.eval()
    return model


def summarize_grouped_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    group_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        grouped.setdefault(group, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        nested = summarize_case_metrics(group_rows)
        for row in summary_rows_from_nested(nested):
            out_row: Dict[str, Any] = dict(row)
            for key, value in zip(group_keys, group):
                out_row[key] = value
            summary_rows.append(out_row)
    return summary_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--weights", type=str, action="append", required=True, help="Repeat once per model.")
    ap.add_argument("--model", type=str, action="append", required=True, choices=["baseline", "gated", "partial"])
    ap.add_argument("--label", type=str, action="append", default=None, help="Optional label aligned with --weights.")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--init_feat", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--slices", type=int, default=None)
    ap.add_argument("--slices_min", type=int, default=4)
    ap.add_argument("--slices_max", type=int, default=32)
    ap.add_argument("--slice_sampling", type=str, default="uniform", choices=["uniform", "normal"])
    ap.add_argument("--slice_mean", type=float, default=None)
    ap.add_argument("--slice_std", type=float, default=None)
    ap.add_argument("--slice_geometry", type=str, default="ultrasound_probe", choices=["random", "fibonacci", "ultrasound_fan", "ultrasound_probe"])
    ap.add_argument("--slice_axis", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--slice_axis_jitter_deg", type=float, default=20.0)
    ap.add_argument("--slice_fan_half_angle_deg", type=float, default=30.0)
    ap.add_argument("--slice_elevation_jitter_deg", type=float, default=5.0)
    ap.add_argument("--slice_sweep_jitter_deg", type=float, default=2.0)
    ap.add_argument("--probe_pos_sigma_vox", type=float, default=1.0)
    ap.add_argument("--probe_depth_sigma_vox", type=float, default=6.0)
    ap.add_argument("--probe_tilt_sigma", type=float, default=0.08)
    ap.add_argument("--thickness_vox", type=float, default=1.0)
    ap.add_argument("--split_file", type=str, default="")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--allow_all_data_eval", action="store_true")
    ap.add_argument("--seed_start", type=int, default=1337)
    ap.add_argument("--num_repeats", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--no_hard_constraint", dest="hard_constraint", action="store_false")
    ap.set_defaults(hard_constraint=True)
    args = ap.parse_args()

    ensure_dir(args.out)
    model_specs = parse_model_specs(args)
    eval_seeds = resolve_eval_seeds(args)
    eval_files, split_manifest = resolve_eval_files(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    default_workers = 0 if sys.platform.startswith("win") else 2

    rows: List[Dict[str, Any]] = []
    for spec in model_specs:
        model = load_model_spec(spec, init_feat=args.init_feat, device=device)
        for eval_seed in eval_seeds:
            dataset = build_dataset(eval_files, args=args, seed=eval_seed)
            loader = DataLoader(
                dataset,
                batch_size=args.batch,
                shuffle=False,
                num_workers=default_workers,
                pin_memory=pin_memory,
            )

            with torch.no_grad():
                for x, y, meta in loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    mask = x[:, 1:2, ...]
                    sparse = x[:, 0:1, ...]
                    pred = clamp_prediction(model(x), sparse=sparse, mask=mask, hard_constraint=args.hard_constraint)

                    case_ids = [str(item) for item in meta["case_id"]]
                    paths = [str(item) for item in meta["path"]]
                    batch_rows = compute_metrics_per_case(
                        pred,
                        y,
                        known_mask=mask,
                        data_range=1.0,
                        case_ids=case_ids,
                        paths=paths,
                    )
                    for row in batch_rows:
                        row["index"] = len(rows)
                        row["split"] = args.split if args.split_file else "all"
                        row["label"] = spec.label
                        row["model"] = spec.kind
                        row["weight_path"] = normalize_case_path(spec.weights)
                        row["eval_seed"] = int(eval_seed)
                        row["slice_geometry"] = str(args.slice_geometry)
                        rows.append(row)

    if not rows:
        raise RuntimeError("Es konnten keine Fallmetriken berechnet werden.")

    write_rows_csv(os.path.join(args.out, "metrics_per_case.csv"), rows)

    summary_by_model = summarize_grouped_rows(rows, group_keys=["label", "model", "weight_path"])
    summary_by_model_seed = summarize_grouped_rows(rows, group_keys=["label", "model", "weight_path", "eval_seed"])
    write_rows_csv(os.path.join(args.out, "metrics_summary_by_model.csv"), summary_by_model)
    write_rows_csv(os.path.join(args.out, "metrics_summary_by_model_seed.csv"), summary_by_model_seed)

    payload = {
        "settings": {
            "dim": int(args.dim),
            "batch": int(args.batch),
            "init_feat": int(args.init_feat),
            "slices": None if args.slices is None else int(args.slices),
            "slices_min": int(args.slices_min),
            "slices_max": int(args.slices_max),
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
            "split": args.split if args.split_file else "all",
            "hard_constraint": bool(args.hard_constraint),
            "eval_seeds": [int(seed) for seed in eval_seeds],
            "num_cases": int(len(eval_files)),
            "max_cases": None if args.max_cases is None else int(args.max_cases),
        },
        "models": [
            {
                "label": spec.label,
                "model": spec.kind,
                "weights": normalize_case_path(spec.weights),
            }
            for spec in model_specs
        ],
        "summary_by_model": summary_by_model,
        "summary_by_model_seed": summary_by_model_seed,
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
            extra={
                "models": [spec.__dict__ for spec in model_specs],
                "eval_seeds": [int(seed) for seed in eval_seeds],
                "num_cases": int(len(eval_files)),
            },
            cwd=ROOT,
        ),
    )

    print("Summary by model:")
    for row in summary_by_model:
        print(
            f"  {row['label']:20s} metric={row['metric']:16s} "
            f"mean={float(row['mean']):.6f} median={float(row['median']):.6f} "
            f"ci95=[{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}]"
        )


if __name__ == "__main__":
    main()
