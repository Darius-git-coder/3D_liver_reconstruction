from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, Iterable


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR_NAME = os.path.basename(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


BEST_RUN_DIR = os.path.join(
    ROOT,
    "runs",
    "ablation_final_128",
    "standard_unet3d_no_partialconv",
    "train",
)
BEST_BASELINE_WEIGHTS = os.path.join(BEST_RUN_DIR, "best_model.pt")
BEST_BASELINE_CONFIG = os.path.join(BEST_RUN_DIR, "config.json")
BEST_BASELINE_SPLIT = os.path.join(BEST_RUN_DIR, "split_manifest.json")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path!r}.")
    return payload


def append_scalar_arg(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    command.extend([name, str(value)])


def append_vector_arg(command: list[str], name: str, values: Iterable[Any] | None) -> None:
    if values is None:
        return
    command.append(name)
    command.extend(str(value) for value in values)


def ensure_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def resolve_project_output_path(path: str) -> str:
    if not path:
        return ROOT
    if os.path.isabs(path):
        return os.path.normpath(path)
    normalized = os.path.normpath(path)
    if normalized == PROJECT_DIR_NAME:
        return ROOT
    project_prefix = PROJECT_DIR_NAME + os.sep
    if normalized.startswith(project_prefix):
        normalized = normalized[len(project_prefix) :]
    return os.path.normpath(os.path.join(ROOT, normalized))


def build_default_out_root() -> str:
    return resolve_project_output_path(
        os.path.join("runs", "ultrasound_realism_compare", "thesis_ablation_best_unet128_usprobe")
    )


def build_finetune_command(args: argparse.Namespace, base_config: Dict[str, Any], out_root: str) -> list[str]:
    out_dir = args.finetune_out or os.path.join(out_root, "finetune")
    command = [
        sys.executable,
        os.path.join(ROOT, "scripts", "finetune_ultrasound_distribution.py"),
        "--data",
        str(args.data or base_config["data"]),
        "--weights",
        str(args.baseline_weights),
        "--base_config",
        str(args.base_config),
        "--split_file",
        str(args.split_file),
        "--out",
        str(out_dir),
        "--epochs",
        str(args.finetune_epochs),
        "--lr",
        str(args.finetune_lr),
        "--slice_geometry",
        str(args.slice_geometry),
        "--slice_axis_jitter_deg",
        str(args.slice_axis_jitter_deg),
        "--slice_fan_half_angle_deg",
        str(args.slice_fan_half_angle_deg),
        "--slice_elevation_jitter_deg",
        str(args.slice_elevation_jitter_deg),
        "--slice_sweep_jitter_deg",
        str(args.slice_sweep_jitter_deg),
        "--probe_pos_sigma_vox",
        str(args.probe_pos_sigma_vox),
        "--probe_depth_sigma_vox",
        str(args.probe_depth_sigma_vox),
        "--probe_tilt_sigma",
        str(args.probe_tilt_sigma),
    ]
    append_scalar_arg(command, "--wd", args.finetune_wd)
    append_scalar_arg(command, "--thickness_vox", args.thickness_vox)
    append_scalar_arg(command, "--seed", args.seed)
    append_vector_arg(command, "--slice_axis", args.slice_axis)
    return command


def build_eval_command(
    args: argparse.Namespace,
    base_config: Dict[str, Any],
    out_root: str,
    finetuned_weights: str,
) -> list[str]:
    out_dir = args.eval_out or os.path.join(out_root, "eval")
    command = [
        sys.executable,
        os.path.join(ROOT, "scripts", "evaluate_ultrasound_distribution.py"),
        "--data",
        str(args.data or base_config["data"]),
        "--split_file",
        str(args.split_file),
        "--split",
        str(args.eval_split),
        "--out",
        str(out_dir),
        "--dim",
        str(base_config["dim"]),
        "--init_feat",
        str(base_config["init_feat"]),
        "--batch",
        str(args.eval_batch),
        "--weights",
        str(args.baseline_weights),
        "--model",
        str(base_config["model"]),
        "--label",
        str(args.baseline_label),
        "--weights",
        str(finetuned_weights),
        "--model",
        str(base_config["model"]),
        "--label",
        str(args.finetune_label),
        "--slice_geometry",
        str(args.slice_geometry),
        "--slice_axis_jitter_deg",
        str(args.slice_axis_jitter_deg),
        "--slice_fan_half_angle_deg",
        str(args.slice_fan_half_angle_deg),
        "--slice_elevation_jitter_deg",
        str(args.slice_elevation_jitter_deg),
        "--slice_sweep_jitter_deg",
        str(args.slice_sweep_jitter_deg),
        "--probe_pos_sigma_vox",
        str(args.probe_pos_sigma_vox),
        "--probe_depth_sigma_vox",
        str(args.probe_depth_sigma_vox),
        "--probe_tilt_sigma",
        str(args.probe_tilt_sigma),
        "--thickness_vox",
        str(args.thickness_vox if args.thickness_vox is not None else base_config["thickness_vox"]),
        "--seed_start",
        str(args.seed_start),
        "--num_repeats",
        str(args.num_repeats),
    ]
    if args.eval_slices is not None:
        command.extend(["--slices", str(args.eval_slices)])
    else:
        command.extend(["--slices_min", str(base_config["slices_min"]), "--slices_max", str(base_config["slices_max"])])
        command.extend(["--slice_sampling", str(base_config["slice_sampling"])])
        append_scalar_arg(command, "--slice_mean", base_config.get("slice_mean"))
        append_scalar_arg(command, "--slice_std", base_config.get("slice_std"))
    append_vector_arg(command, "--slice_axis", args.slice_axis)
    return command


def build_visualization_command(
    args: argparse.Namespace,
    base_config: Dict[str, Any],
    out_root: str,
    finetuned_weights: str,
) -> list[str]:
    if not args.visualize_case_id:
        return []
    out_dir = os.path.join(out_root, "visual_compare")
    command = [
        sys.executable,
        os.path.join(ROOT, "scripts", "visualize_model_comparison.py"),
        "--data",
        str(args.data or base_config["data"]),
        "--split_file",
        str(args.split_file),
        "--split",
        str(args.eval_split),
        "--case_id",
        str(args.visualize_case_id),
        "--weights_a",
        str(args.baseline_weights),
        "--weights_b",
        str(finetuned_weights),
        "--model_a",
        str(base_config["model"]),
        "--model_b",
        str(base_config["model"]),
        "--label_a",
        str(args.baseline_label),
        "--label_b",
        str(args.finetune_label),
        "--out",
        str(out_dir),
        "--dim",
        str(base_config["dim"]),
        "--init_feat",
        str(base_config["init_feat"]),
        "--slice_geometry",
        str(args.slice_geometry),
        "--slice_axis_jitter_deg",
        str(args.slice_axis_jitter_deg),
        "--slice_fan_half_angle_deg",
        str(args.slice_fan_half_angle_deg),
        "--slice_elevation_jitter_deg",
        str(args.slice_elevation_jitter_deg),
        "--slice_sweep_jitter_deg",
        str(args.slice_sweep_jitter_deg),
        "--probe_pos_sigma_vox",
        str(args.probe_pos_sigma_vox),
        "--probe_depth_sigma_vox",
        str(args.probe_depth_sigma_vox),
        "--probe_tilt_sigma",
        str(args.probe_tilt_sigma),
        "--thickness_vox",
        str(args.thickness_vox if args.thickness_vox is not None else base_config["thickness_vox"]),
        "--seed",
        str(args.seed_start),
    ]
    if args.eval_slices is not None:
        command.extend(["--slices", str(args.eval_slices)])
    append_vector_arg(command, "--slice_axis", args.slice_axis)
    return command


def write_manifest(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", type=str, default="all", choices=["all", "finetune", "eval"])
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--baseline_weights", type=str, default=BEST_BASELINE_WEIGHTS)
    ap.add_argument("--base_config", type=str, default=BEST_BASELINE_CONFIG)
    ap.add_argument("--split_file", type=str, default=BEST_BASELINE_SPLIT)
    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--finetune_out", type=str, default="")
    ap.add_argument("--eval_out", type=str, default="")
    ap.add_argument("--finetune_epochs", type=int, default=40)
    ap.add_argument("--finetune_lr", type=float, default=5e-5)
    ap.add_argument("--finetune_wd", type=float, default=None)
    ap.add_argument("--eval_batch", type=int, default=2)
    ap.add_argument("--eval_split", type=str, default="test")
    ap.add_argument("--eval_slices", type=int, default=64)
    ap.add_argument("--seed_start", type=int, default=1337)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_repeats", type=int, default=5)
    ap.add_argument("--baseline_label", type=str, default="thesis_ablation_best_unet")
    ap.add_argument("--finetune_label", type=str, default="thesis_ablation_best_unet_usprobe_finetuned")
    ap.add_argument("--slice_geometry", type=str, default="ultrasound_probe", choices=["random", "fibonacci", "ultrasound_fan", "ultrasound_probe"])
    ap.add_argument("--slice_axis", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--slice_axis_jitter_deg", type=float, default=20.0)
    ap.add_argument("--slice_fan_half_angle_deg", type=float, default=30.0)
    ap.add_argument("--slice_elevation_jitter_deg", type=float, default=5.0)
    ap.add_argument("--slice_sweep_jitter_deg", type=float, default=2.0)
    ap.add_argument("--probe_pos_sigma_vox", type=float, default=1.0)
    ap.add_argument("--probe_depth_sigma_vox", type=float, default=6.0)
    ap.add_argument("--probe_tilt_sigma", type=float, default=0.08)
    ap.add_argument("--thickness_vox", type=float, default=None)
    ap.add_argument("--visualize_case_id", type=str, default="")
    ap.add_argument("--finetuned_weights", type=str, default="")
    args = ap.parse_args()

    ensure_exists(args.baseline_weights, "Baseline weights")
    ensure_exists(args.base_config, "Baseline config")
    ensure_exists(args.split_file, "Split manifest")
    base_config = load_json(args.base_config)

    out_root = resolve_project_output_path(args.out_root) if args.out_root else build_default_out_root()
    os.makedirs(out_root, exist_ok=True)
    finetune_out = (
        resolve_project_output_path(args.finetune_out)
        if args.finetune_out
        else os.path.join(out_root, "finetune")
    )
    eval_out = (
        resolve_project_output_path(args.eval_out)
        if args.eval_out
        else os.path.join(out_root, "eval")
    )
    finetuned_weights = args.finetuned_weights or os.path.join(finetune_out, "best_model.pt")

    manifest = {
        "baseline_run_dir": os.path.normpath(os.path.abspath(os.path.dirname(args.base_config))),
        "baseline_weights": os.path.normpath(os.path.abspath(args.baseline_weights)),
        "base_config": os.path.normpath(os.path.abspath(args.base_config)),
        "split_file": os.path.normpath(os.path.abspath(args.split_file)),
        "data": str(args.data or base_config["data"]),
        "out_root": os.path.normpath(os.path.abspath(out_root)),
        "finetune_out": os.path.normpath(os.path.abspath(finetune_out)),
        "eval_out": os.path.normpath(os.path.abspath(eval_out)),
        "expected_finetuned_weights": os.path.normpath(os.path.abspath(finetuned_weights)),
        "compare_protocol": {
            "goal": "Compare the best thesis baseline against a finetuned model under more ultrasound-realistic slice sampling.",
            "training_slice_geometry": str(args.slice_geometry),
            "evaluation_slice_geometry": str(args.slice_geometry),
            "evaluation_slice_count": None if args.eval_slices is None else int(args.eval_slices),
            "evaluation_num_repeats": int(args.num_repeats),
        },
    }
    write_manifest(os.path.join(out_root, "comparison_manifest.json"), manifest)

    if args.mode in {"all", "finetune"}:
        finetune_command = build_finetune_command(args, base_config, out_root)
        write_manifest(
            os.path.join(out_root, "finetune_command.json"),
            {"command": finetune_command},
        )
        print("Launching finetuning:")
        print(" ".join(finetune_command))
        subprocess.run(finetune_command, check=True, cwd=ROOT)

    if args.mode in {"all", "eval"}:
        ensure_exists(finetuned_weights, "Finetuned weights")
        eval_command = build_eval_command(args, base_config, out_root, finetuned_weights)
        write_manifest(
            os.path.join(out_root, "eval_command.json"),
            {"command": eval_command},
        )
        print("Launching evaluation:")
        print(" ".join(eval_command))
        subprocess.run(eval_command, check=True, cwd=ROOT)

        visualize_command = build_visualization_command(args, base_config, out_root, finetuned_weights)
        if visualize_command:
            write_manifest(
                os.path.join(out_root, "visualize_command.json"),
                {"command": visualize_command},
            )
            print("Launching visualization:")
            print(" ".join(visualize_command))
            subprocess.run(visualize_command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
