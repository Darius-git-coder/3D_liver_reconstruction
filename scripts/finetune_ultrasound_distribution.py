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


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path!r}.")
    return payload


def choose_value(cli_value: Any, config: Dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


def append_scalar_arg(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    command.extend([name, str(value)])


def append_vector_arg(command: list[str], name: str, values: Iterable[Any] | None) -> None:
    if values is None:
        return
    command.append(name)
    command.extend(str(value) for value in values)


def resolve_project_output_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    normalized = os.path.normpath(path)
    if normalized == PROJECT_DIR_NAME:
        return ROOT
    project_prefix = PROJECT_DIR_NAME + os.sep
    if normalized.startswith(project_prefix):
        normalized = normalized[len(project_prefix) :]
    return os.path.normpath(os.path.join(ROOT, normalized))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True, help="Pretrained state_dict or checkpoint used for initialization.")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--base_config", type=str, default="", help="Optional config.json from the original training run.")
    ap.add_argument("--split_file", type=str, default=None)
    ap.add_argument("--train_split", type=str, default=None)
    ap.add_argument("--val_split", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--model", type=str, default=None, choices=["baseline", "gated", "partial"])
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--init_feat", type=int, default=None)
    ap.add_argument("--slices_min", type=int, default=None)
    ap.add_argument("--slices_max", type=int, default=None)
    ap.add_argument("--slice_sampling", type=str, default=None, choices=["uniform", "normal"])
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
    ap.add_argument("--thickness_vox", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scheduler", type=str, default=None, choices=["cosine", "plateau"])
    ap.add_argument("--curriculum", dest="curriculum", action="store_true")
    ap.add_argument("--no_curriculum", dest="curriculum", action="store_false")
    ap.add_argument("--amp", dest="amp", action="store_true")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--deterministic", dest="deterministic", action="store_true")
    ap.add_argument("--no_deterministic", dest="deterministic", action="store_false")
    ap.add_argument("--hard_constraint", dest="hard_constraint", action="store_true")
    ap.add_argument("--no_hard_constraint", dest="hard_constraint", action="store_false")
    ap.set_defaults(curriculum=None, amp=None, deterministic=None, hard_constraint=None)
    args = ap.parse_args()

    base_config: Dict[str, Any] = {}
    if args.base_config:
        base_config = load_json(args.base_config)

    merged = {
        "model": choose_value(args.model, base_config, "model", "partial"),
        "dim": choose_value(args.dim, base_config, "dim", 64),
        "epochs": int(args.epochs),
        "batch": choose_value(args.batch, base_config, "batch", 2),
        "lr": float(args.lr),
        "wd": choose_value(args.wd, base_config, "wd", 5e-6),
        "init_feat": choose_value(args.init_feat, base_config, "init_feat", 32),
        "slices_min": choose_value(args.slices_min, base_config, "slices_min", 4),
        "slices_max": choose_value(args.slices_max, base_config, "slices_max", 32),
        "slice_sampling": choose_value(args.slice_sampling, base_config, "slice_sampling", "uniform"),
        "slice_mean": choose_value(args.slice_mean, base_config, "slice_mean", None),
        "slice_std": choose_value(args.slice_std, base_config, "slice_std", None),
        "slice_geometry": str(args.slice_geometry),
        "slice_axis": args.slice_axis,
        "slice_axis_jitter_deg": float(args.slice_axis_jitter_deg),
        "slice_fan_half_angle_deg": float(args.slice_fan_half_angle_deg),
        "slice_elevation_jitter_deg": float(args.slice_elevation_jitter_deg),
        "slice_sweep_jitter_deg": float(args.slice_sweep_jitter_deg),
        "probe_pos_sigma_vox": float(args.probe_pos_sigma_vox),
        "probe_depth_sigma_vox": float(args.probe_depth_sigma_vox),
        "probe_tilt_sigma": float(args.probe_tilt_sigma),
        "thickness_vox": choose_value(args.thickness_vox, base_config, "thickness_vox", 1.0),
        "seed": choose_value(args.seed, base_config, "seed", 1337),
        "scheduler": choose_value(args.scheduler, base_config, "scheduler", "cosine"),
        "curriculum": choose_value(args.curriculum, base_config, "curriculum", False),
        "amp": choose_value(args.amp, base_config, "amp", False),
        "deterministic": choose_value(args.deterministic, base_config, "deterministic", False),
        "hard_constraint": choose_value(args.hard_constraint, base_config, "hard_constraint", True),
        "train_split": choose_value(args.train_split, base_config, "train_split", "train"),
        "val_split": choose_value(args.val_split, base_config, "val_split", "val"),
        "split_file": choose_value(args.split_file, base_config, "split_file", ""),
        "hole_weight": choose_value(None, base_config, "hole_weight", 30.0),
        "valid_weight": choose_value(None, base_config, "valid_weight", 1.0),
        "hole_fg_weight": choose_value(None, base_config, "hole_fg_weight", None),
        "hole_bg_weight": choose_value(None, base_config, "hole_bg_weight", None),
        "foreground_threshold": choose_value(None, base_config, "foreground_threshold", 0.1),
        "w_grad": choose_value(None, base_config, "w_grad", 0.05),
        "w_ssim": choose_value(None, base_config, "w_ssim", 0.0),
        "w_ms_ssim": choose_value(None, base_config, "w_ms_ssim", 0.2),
        "curriculum_stage_epochs": choose_value(None, base_config, "curriculum_stage_epochs", 50),
        "sparse_noise_std": choose_value(None, base_config, "sparse_noise_std", 0.0),
        "intensity_aug_prob": choose_value(None, base_config, "intensity_aug_prob", 0.0),
        "intensity_scale_min": choose_value(None, base_config, "intensity_scale_min", 0.85),
        "intensity_scale_max": choose_value(None, base_config, "intensity_scale_max", 1.15),
        "intensity_bias_min": choose_value(None, base_config, "intensity_bias_min", -0.05),
        "intensity_bias_max": choose_value(None, base_config, "intensity_bias_max", 0.05),
        "train_ratio": choose_value(None, base_config, "train_ratio", 0.7),
        "val_ratio": choose_value(None, base_config, "val_ratio", 0.15),
        "test_ratio": choose_value(None, base_config, "test_ratio", 0.15),
        "max_train_cases": choose_value(None, base_config, "max_train_cases", None),
        "max_val_cases": choose_value(None, base_config, "max_val_cases", None),
    }

    out_dir = resolve_project_output_path(args.out)
    os.makedirs(out_dir, exist_ok=True)
    command = [
        sys.executable,
        os.path.join(ROOT, "scripts", "train_inpainting.py"),
        "--data",
        args.data,
        "--out",
        out_dir,
        "--init_weights",
        args.weights,
        "--model",
        str(merged["model"]),
        "--dim",
        str(merged["dim"]),
        "--epochs",
        str(merged["epochs"]),
        "--batch",
        str(merged["batch"]),
        "--lr",
        str(merged["lr"]),
        "--wd",
        str(merged["wd"]),
        "--init_feat",
        str(merged["init_feat"]),
        "--slices_min",
        str(merged["slices_min"]),
        "--slices_max",
        str(merged["slices_max"]),
        "--slice_sampling",
        str(merged["slice_sampling"]),
        "--slice_geometry",
        str(merged["slice_geometry"]),
        "--slice_axis_jitter_deg",
        str(merged["slice_axis_jitter_deg"]),
        "--slice_fan_half_angle_deg",
        str(merged["slice_fan_half_angle_deg"]),
        "--slice_elevation_jitter_deg",
        str(merged["slice_elevation_jitter_deg"]),
        "--slice_sweep_jitter_deg",
        str(merged["slice_sweep_jitter_deg"]),
        "--probe_pos_sigma_vox",
        str(merged["probe_pos_sigma_vox"]),
        "--probe_depth_sigma_vox",
        str(merged["probe_depth_sigma_vox"]),
        "--probe_tilt_sigma",
        str(merged["probe_tilt_sigma"]),
        "--thickness_vox",
        str(merged["thickness_vox"]),
        "--seed",
        str(merged["seed"]),
        "--scheduler",
        str(merged["scheduler"]),
        "--hole_weight",
        str(merged["hole_weight"]),
        "--valid_weight",
        str(merged["valid_weight"]),
        "--foreground_threshold",
        str(merged["foreground_threshold"]),
        "--w_grad",
        str(merged["w_grad"]),
        "--w_ssim",
        str(merged["w_ssim"]),
        "--w_ms_ssim",
        str(merged["w_ms_ssim"]),
        "--curriculum_stage_epochs",
        str(merged["curriculum_stage_epochs"]),
        "--sparse_noise_std",
        str(merged["sparse_noise_std"]),
        "--intensity_aug_prob",
        str(merged["intensity_aug_prob"]),
        "--intensity_scale_min",
        str(merged["intensity_scale_min"]),
        "--intensity_scale_max",
        str(merged["intensity_scale_max"]),
        "--intensity_bias_min",
        str(merged["intensity_bias_min"]),
        "--intensity_bias_max",
        str(merged["intensity_bias_max"]),
        "--train_split",
        str(merged["train_split"]),
        "--val_split",
        str(merged["val_split"]),
        "--train_ratio",
        str(merged["train_ratio"]),
        "--val_ratio",
        str(merged["val_ratio"]),
        "--test_ratio",
        str(merged["test_ratio"]),
    ]

    append_scalar_arg(command, "--slice_mean", merged["slice_mean"])
    append_scalar_arg(command, "--slice_std", merged["slice_std"])
    append_vector_arg(command, "--slice_axis", merged["slice_axis"])
    append_scalar_arg(command, "--split_file", merged["split_file"])
    append_scalar_arg(command, "--hole_fg_weight", merged["hole_fg_weight"])
    append_scalar_arg(command, "--hole_bg_weight", merged["hole_bg_weight"])
    append_scalar_arg(command, "--max_train_cases", merged["max_train_cases"])
    append_scalar_arg(command, "--max_val_cases", merged["max_val_cases"])

    if bool(merged["curriculum"]):
        command.append("--curriculum")
    if bool(merged["amp"]):
        command.append("--amp")
    if bool(merged["deterministic"]):
        command.append("--deterministic")
    if not bool(merged["hard_constraint"]):
        command.append("--no_hard_constraint")

    payload = {
        "command": command,
        "settings": merged,
        "weights": os.path.normpath(args.weights),
        "base_config": os.path.normpath(args.base_config) if args.base_config else "",
        "resolved_out": out_dir,
    }
    with open(os.path.join(out_dir, "finetune_ultrasound_plan.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print("Launching finetune command:")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
