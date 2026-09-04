from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inpainting3d.utils import ensure_dir


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def add_flag(command: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def add_option(command: List[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, (list, tuple)):
        command.append(flag)
        command.extend(str(item) for item in value)
        return
    command.extend([flag, str(value)])


def add_options(command: List[str], mapping: Dict[str, str], values: Dict[str, Any]) -> None:
    for key, flag in mapping.items():
        add_option(command, flag, values.get(key))


TRAIN_COMMON_MAPPING = {
    "dim": "--dim",
    "batch": "--batch",
    "seed": "--seed",
    "slices_min": "--slices_min",
    "slices_max": "--slices_max",
    "slice_sampling": "--slice_sampling",
    "slice_mean": "--slice_mean",
    "slice_std": "--slice_std",
    "slice_geometry": "--slice_geometry",
    "slice_axis": "--slice_axis",
    "slice_axis_jitter_deg": "--slice_axis_jitter_deg",
    "slice_fan_half_angle_deg": "--slice_fan_half_angle_deg",
    "slice_elevation_jitter_deg": "--slice_elevation_jitter_deg",
    "slice_sweep_jitter_deg": "--slice_sweep_jitter_deg",
    "probe_pos_sigma_vox": "--probe_pos_sigma_vox",
    "probe_depth_sigma_vox": "--probe_depth_sigma_vox",
    "probe_tilt_sigma": "--probe_tilt_sigma",
    "thickness_vox": "--thickness_vox",
    "split_file": "--split_file",
    "train_split": "--train_split",
    "val_split": "--val_split",
    "max_train_cases": "--max_train_cases",
    "max_val_cases": "--max_val_cases",
}

EVAL_COMMON_MAPPING = {
    "dim": "--dim",
    "batch": "--batch",
    "seed": "--seed",
    "slices_min": "--slices_min",
    "slices_max": "--slices_max",
    "slice_sampling": "--slice_sampling",
    "slice_mean": "--slice_mean",
    "slice_std": "--slice_std",
    "slice_geometry": "--slice_geometry",
    "slice_axis": "--slice_axis",
    "slice_axis_jitter_deg": "--slice_axis_jitter_deg",
    "slice_fan_half_angle_deg": "--slice_fan_half_angle_deg",
    "slice_elevation_jitter_deg": "--slice_elevation_jitter_deg",
    "slice_sweep_jitter_deg": "--slice_sweep_jitter_deg",
    "probe_pos_sigma_vox": "--probe_pos_sigma_vox",
    "probe_depth_sigma_vox": "--probe_depth_sigma_vox",
    "probe_tilt_sigma": "--probe_tilt_sigma",
    "thickness_vox": "--thickness_vox",
    "split_file": "--split_file",
}

RESUNET_TRAIN_MAPPING = {
    "epochs": "--epochs",
    "lr": "--lr",
    "wd": "--wd",
    "init_feat": "--init_feat",
    "hole_weight": "--hole_weight",
    "valid_weight": "--valid_weight",
    "foreground_threshold": "--foreground_threshold",
    "w_grad": "--w_grad",
    "w_ssim": "--w_ssim",
    "w_ms_ssim": "--w_ms_ssim",
    "scheduler": "--scheduler",
}

GENERATIVE_TRAIN_MAPPING = {
    "epochs": "--epochs",
    "lr": "--lr",
    "wd": "--wd",
    "init_feat": "--init_feat",
    "levels": "--levels",
    "time_dim": "--time_dim",
    "dropout": "--dropout",
    "hole_weight": "--hole_weight",
    "valid_weight": "--valid_weight",
    "scheduler": "--scheduler",
    "diffusion_steps": "--diffusion_steps",
    "diffusion_schedule": "--diffusion_schedule",
    "sample_steps": "--sample_steps",
    "sample_every": "--sample_every",
}


def merged(common: Dict[str, Any], specific: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(common)
    out.update(specific)
    return out


def run_command(command: List[str], dry_run: bool = False) -> None:
    print("\n" + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def seed_list(common: Dict[str, Any], specific: Dict[str, Any]) -> List[int]:
    raw = specific.get("seeds", common.get("seeds"))
    if raw is None:
        return [int(specific.get("seed", common.get("seed", 1337)))]
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, str):
        return [int(raw)]
    seeds = [int(seed) for seed in raw]
    if not seeds:
        raise ValueError("Seed lists must not be empty.")
    return seeds


def use_seed_dirs(config: Dict[str, Any]) -> bool:
    common = config.get("common", {})
    return any(len(seed_list(common, method_values)) > 1 for _, method_values in iter_methods(config))


def output_dirs(root_out: str, method: str, seed: int, nested_by_seed: bool) -> tuple[str, str]:
    if nested_by_seed:
        method_root = os.path.join(root_out, method, f"seed_{seed}")
    else:
        method_root = os.path.join(root_out, method)
    return os.path.join(method_root, "train"), os.path.join(method_root, "eval")


def eval_values_for_seed(values: Dict[str, Any], train_seed: int) -> Dict[str, Any]:
    eval_values = dict(values)
    eval_values["seed"] = int(values.get("eval_seed", train_seed))
    return eval_values


def build_resunet_train(python_exe: str, data: str, out_dir: str, values: Dict[str, Any]) -> List[str]:
    command = [python_exe, os.path.join("scripts", "train_inpainting.py"), "--data", data, "--out", out_dir]
    command.extend(["--model", str(values.get("model", "baseline"))])
    add_options(command, TRAIN_COMMON_MAPPING, values)
    add_options(command, RESUNET_TRAIN_MAPPING, values)
    add_flag(command, "--amp", bool(values.get("amp", False)))
    add_flag(command, "--deterministic", bool(values.get("deterministic", False)))
    if values.get("hard_constraint") is False:
        command.append("--no_hard_constraint")
    return command


def build_resunet_eval(python_exe: str, data: str, train_out: str, eval_out: str, values: Dict[str, Any]) -> List[str]:
    command = [
        python_exe,
        os.path.join("scripts", "evaluate_inpainting.py"),
        "--data",
        data,
        "--weights",
        os.path.join(train_out, "best_model.pt"),
        "--out",
        eval_out,
        "--model",
        str(values.get("model", "baseline")),
    ]
    eval_values = dict(values)
    eval_values["batch"] = values.get("eval_batch", values.get("batch", 1))
    add_options(command, EVAL_COMMON_MAPPING, eval_values)
    add_option(command, "--init_feat", values.get("init_feat"))
    add_option(command, "--split", values.get("eval_split", "test"))
    add_option(command, "--slices", values.get("eval_slices"))
    add_option(command, "--max_cases", values.get("max_eval_cases"))
    add_option(command, "--label", values.get("label", "resunet"))
    if values.get("hard_constraint") is False:
        command.append("--no_hard_constraint")
    return command


def build_generative_train(
    python_exe: str,
    objective: str,
    data: str,
    out_dir: str,
    values: Dict[str, Any],
) -> List[str]:
    command = [
        python_exe,
        os.path.join("scripts", "train_generative_inpainting.py"),
        "--data",
        data,
        "--out",
        out_dir,
        "--objective",
        objective,
    ]
    add_options(command, TRAIN_COMMON_MAPPING, values)
    add_options(command, GENERATIVE_TRAIN_MAPPING, values)
    add_flag(command, "--amp", bool(values.get("amp", False)))
    add_flag(command, "--deterministic", bool(values.get("deterministic", False)))
    if values.get("hard_constraint") is False:
        command.append("--no_hard_constraint")
    return command


def build_generative_eval(
    python_exe: str,
    objective: str,
    data: str,
    train_out: str,
    eval_out: str,
    values: Dict[str, Any],
) -> List[str]:
    command = [
        python_exe,
        os.path.join("scripts", "evaluate_generative_inpainting.py"),
        "--data",
        data,
        "--weights",
        os.path.join(train_out, "best_model.pt"),
        "--out",
        eval_out,
        "--objective",
        objective,
    ]
    eval_values = dict(values)
    eval_values["batch"] = values.get("eval_batch", values.get("batch", 1))
    add_options(command, EVAL_COMMON_MAPPING, eval_values)
    add_option(command, "--init_feat", values.get("init_feat"))
    add_option(command, "--levels", values.get("levels"))
    add_option(command, "--time_dim", values.get("time_dim"))
    add_option(command, "--dropout", values.get("dropout"))
    add_option(command, "--sample_steps", values.get("eval_sample_steps", values.get("sample_steps")))
    add_option(command, "--diffusion_steps", values.get("diffusion_steps"))
    add_option(command, "--diffusion_schedule", values.get("diffusion_schedule"))
    add_option(command, "--samples_per_case", values.get("samples_per_case", 1))
    add_option(command, "--split", values.get("eval_split", "test"))
    add_option(command, "--slices", values.get("eval_slices"))
    add_option(command, "--max_cases", values.get("max_eval_cases"))
    add_option(command, "--label", values.get("label", objective))
    if values.get("hard_constraint") is False:
        command.append("--no_hard_constraint")
    return command


def build_summary_command(
    python_exe: str,
    root_out: str,
    eval_records: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    metrics: Sequence[str] | None,
    bootstrap_repeats: int,
) -> List[str]:
    command = [
        python_exe,
        os.path.join("scripts", "summarize_method_comparison.py"),
        "--out",
        os.path.join(root_out, "scientific_summary"),
        "--baseline",
        baseline,
        "--bootstrap_repeats",
        str(int(bootstrap_repeats)),
    ]
    if metrics:
        command.append("--metrics")
        command.extend(str(metric) for metric in metrics)
    for record in eval_records:
        command.extend(
            [
                "--input",
                f"{record['method']}:{record['seed']}={record['eval_dir']}",
            ]
        )
    return command


def iter_methods(config: Dict[str, Any]) -> Iterable[tuple[str, Dict[str, Any]]]:
    methods = config.get("methods", {})
    for name in ("resunet", "diffusion", "flow_matching"):
        values = methods.get(name, {})
        if values.get("enabled", True):
            yield name, values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ResUNet vs diffusion vs flow-matching comparison suite.")
    parser.add_argument("--config", type=str, default=os.path.join("configs", "resunet_diffusion_flowmatching.example.json"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--baseline", type=str, default="resunet")
    args = parser.parse_args()

    config = load_json(args.config)
    common = config.get("common", {})
    analysis = config.get("analysis", {})
    data = str(config["data"])
    root_out = str(config.get("out_root", os.path.join("runs", "resunet_diffusion_flowmatching")))
    python_exe = str(config.get("python_exe") or sys.executable)
    ensure_dir(root_out)

    nested_by_seed = use_seed_dirs(config)
    manifest: Dict[str, Any] = {"config": args.config, "commands": [], "evaluations": []}
    for name, method_values in iter_methods(config):
        for train_seed in seed_list(common, method_values):
            values = merged(common, method_values)
            values["seed"] = int(train_seed)
            train_out, eval_out = output_dirs(root_out, name, train_seed, nested_by_seed)

            if name == "resunet":
                train_cmd = build_resunet_train(python_exe, data, train_out, values)
                eval_cmd = build_resunet_eval(
                    python_exe,
                    data,
                    train_out,
                    eval_out,
                    eval_values_for_seed(values, train_seed),
                )
            else:
                train_cmd = build_generative_train(python_exe, name, data, train_out, values)
                eval_cmd = build_generative_eval(
                    python_exe,
                    name,
                    data,
                    train_out,
                    eval_out,
                    eval_values_for_seed(values, train_seed),
                )

            if not args.skip_train:
                manifest["commands"].append({"method": name, "seed": train_seed, "stage": "train", "command": train_cmd})
                run_command(train_cmd, dry_run=args.dry_run)
            if not args.skip_eval:
                manifest["commands"].append({"method": name, "seed": train_seed, "stage": "eval", "command": eval_cmd})
                manifest["evaluations"].append(
                    {
                        "method": name,
                        "label": values.get("label", name),
                        "seed": int(train_seed),
                        "eval_seed": int(values.get("eval_seed", train_seed)),
                        "train_dir": os.path.normpath(train_out),
                        "eval_dir": os.path.normpath(eval_out),
                    }
                )
                run_command(eval_cmd, dry_run=args.dry_run)

    if (
        not args.skip_eval
        and not args.skip_summary
        and len({record["method"] for record in manifest["evaluations"]}) >= 2
        and args.baseline in {record["method"] for record in manifest["evaluations"]}
    ):
        summary_cmd = build_summary_command(
            python_exe,
            root_out,
            manifest["evaluations"],
            baseline=args.baseline,
            metrics=analysis.get("metrics"),
            bootstrap_repeats=int(analysis.get("bootstrap_repeats", 10000)),
        )
        manifest["commands"].append({"stage": "summary", "command": summary_cmd})
        run_command(summary_cmd, dry_run=args.dry_run)

    save_json(os.path.join(root_out, "suite_manifest.json"), manifest)
    print(f"\nWrote suite manifest to {os.path.join(root_out, 'suite_manifest.json')}")


if __name__ == "__main__":
    main()
