from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader

from inpainting3d.generative import (
    build_generative_model,
    make_diffusion_schedule,
    sample_diffusion_ddim,
    sample_flow_matching,
)
from inpainting3d.metrics import compute_metrics_per_case
from inpainting3d.repro import build_repro_metadata
from inpainting3d.splits import normalize_case_path, write_split_manifest
from inpainting3d.stats import summarize_case_metrics, summary_rows_from_nested
from inpainting3d.utils import ensure_dir, set_reproducibility, strip_dataparallel_prefix, torch_load_weights_compat
from scripts.evaluate_inpainting import resolve_eval_files
from scripts.train_inpainting import LiverInpaintingDataset


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model(weights: str, args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, Dict[str, Any]]:
    model = build_generative_model(
        init_feat=args.init_feat,
        levels=args.levels,
        time_dim=args.time_dim,
        dropout=args.dropout,
    ).to(device)
    checkpoint = torch_load_weights_compat(weights, map_location=device)
    metadata: Dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        metadata = {
            "objective": checkpoint.get("objective"),
            "epoch": checkpoint.get("epoch"),
            "best_val": checkpoint.get("best_val"),
            "config": checkpoint.get("config"),
        }
        state = checkpoint["model"]
    else:
        state = checkpoint
    if isinstance(state, dict) and any(isinstance(value, torch.Tensor) for value in state.values()):
        model.load_state_dict(strip_dataparallel_prefix(state), strict=True)
    else:
        raise RuntimeError(f"Unexpected checkpoint format: {weights}")
    model.eval()
    return model, metadata


def dataset_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    slices_min = int(args.slices if args.slices is not None else args.slices_min)
    slices_max = int(args.slices if args.slices is not None else args.slices_max)
    return {
        "dim": args.dim,
        "slices_min": slices_min,
        "slices_max": slices_max,
        "slice_sampling": args.slice_sampling,
        "slice_mean": args.slice_mean,
        "slice_std": args.slice_std,
        "slice_geometry": args.slice_geometry,
        "slice_axis": args.slice_axis,
        "slice_axis_jitter_deg": args.slice_axis_jitter_deg,
        "slice_fan_half_angle_deg": args.slice_fan_half_angle_deg,
        "slice_elevation_jitter_deg": args.slice_elevation_jitter_deg,
        "slice_sweep_jitter_deg": args.slice_sweep_jitter_deg,
        "probe_pos_sigma_vox": args.probe_pos_sigma_vox,
        "probe_depth_sigma_vox": args.probe_depth_sigma_vox,
        "probe_tilt_sigma": args.probe_tilt_sigma,
        "thickness_vox": args.thickness_vox,
        "is_train": False,
        "seed": args.seed,
        "return_meta": True,
    }


@torch.no_grad()
def generate(
    args: argparse.Namespace,
    model: torch.nn.Module,
    sparse: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if args.objective == "diffusion":
        schedule = make_diffusion_schedule(args.diffusion_steps, device=device, schedule=args.diffusion_schedule)
        return sample_diffusion_ddim(
            model,
            sparse=sparse,
            known_mask=mask,
            schedule=schedule,
            sample_steps=args.sample_steps,
            hard_constraint=args.hard_constraint,
        )
    if args.objective == "flow_matching":
        return sample_flow_matching(
            model,
            sparse=sparse,
            known_mask=mask,
            sample_steps=args.sample_steps,
            hard_constraint=args.hard_constraint,
        )
    raise ValueError(f"Unknown objective: {args.objective}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate conditional 3D diffusion or flow-matching inpainting models.")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out", type=str, default="./runs/generative_eval")
    ap.add_argument("--objective", type=str, default="diffusion", choices=["diffusion", "flow_matching"])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--init_feat", type=int, default=16)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--time_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--samples_per_case", type=int, default=1)
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--diffusion_steps", type=int, default=1000)
    ap.add_argument("--diffusion_schedule", type=str, default="cosine", choices=["cosine", "linear"])
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
    ap.add_argument("--split_file", type=str, default="")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--allow_all_data_eval", action="store_true")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--no_hard_constraint", dest="hard_constraint", action="store_false")
    ap.set_defaults(hard_constraint=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_case < 1:
        raise ValueError("--samples_per_case must be >= 1")

    ensure_dir(args.out)
    set_reproducibility(args.seed, deterministic=False)
    eval_files, split_manifest = resolve_eval_files(args)
    if split_manifest is not None:
        split_manifest["selected_splits"] = {"eval": args.split}
        write_split_manifest(split_manifest, os.path.join(args.out, "split_manifest.json"))

    ds = LiverInpaintingDataset(eval_files, **dataset_kwargs(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    workers = 0 if sys.platform.startswith("win") else 2
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=workers, pin_memory=pin_memory)

    model, checkpoint_metadata = load_model(args.weights, args, device)

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sparse = x[:, 0:1, ...]
            mask = x[:, 1:2, ...]

            predictions = []
            for _ in range(args.samples_per_case):
                predictions.append(generate(args, model, sparse, mask, device))
            pred = torch.stack(predictions, dim=0).mean(dim=0)

            case_ids = [str(item) for item in meta["case_id"]]
            paths = [str(item) for item in meta["path"]]
            batch_rows = compute_metrics_per_case(pred, y, known_mask=mask, data_range=1.0, case_ids=case_ids, paths=paths)
            for row in batch_rows:
                row["index"] = len(rows)
                row["split"] = args.split if args.split_file else "all"
                row["label"] = args.label or args.objective
                row["objective"] = args.objective
                rows.append(row)

    if not rows:
        raise RuntimeError("No metrics were computed.")

    write_rows_csv(os.path.join(args.out, "metrics_per_case.csv"), rows)
    nested_summary = summarize_case_metrics(rows)
    write_rows_csv(os.path.join(args.out, "metrics_summary.csv"), summary_rows_from_nested(nested_summary))
    save_json(
        os.path.join(args.out, "metrics_summary.json"),
        {
            "summary": nested_summary,
            "settings": {
                **vars(args),
                "weights": normalize_case_path(args.weights),
                "num_cases": len(rows),
            },
            "checkpoint": checkpoint_metadata,
        },
    )
    save_json(
        os.path.join(args.out, "reproducibility.json"),
        build_repro_metadata(
            vars(args),
            split_manifest_path=os.path.join(args.out, "split_manifest.json") if split_manifest is not None else None,
            extra={"weights": normalize_case_path(args.weights), "num_cases": len(rows)},
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
