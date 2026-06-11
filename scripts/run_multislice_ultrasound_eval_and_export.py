from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path!r}.")
    return payload


def append_scalar_arg(command: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def append_vector_arg(command: List[str], flag: str, values: Iterable[Any] | None) -> None:
    if values is None:
        return
    values = list(values)
    if not values:
        return
    command.append(flag)
    command.extend(str(value) for value in values)


def build_eval_command(*, python_exe: str, out_dir: str, slices: int, repro: Dict[str, Any]) -> List[str]:
    args_payload = repro.get("args")
    extra = repro.get("extra")
    if not isinstance(args_payload, dict):
        raise RuntimeError("reproducibility.json does not contain an 'args' object.")
    if not isinstance(extra, dict) or not isinstance(extra.get("models"), list):
        raise RuntimeError("reproducibility.json does not contain model metadata in 'extra.models'.")

    script_path = os.path.join(ROOT, "scripts", "evaluate_ultrasound_distribution.py")
    command = [
        python_exe,
        script_path,
        "--data",
        str(args_payload.get("data", "")),
        "--split_file",
        str(args_payload.get("split_file", "")),
        "--split",
        str(args_payload.get("split", "test")),
        "--out",
        out_dir,
        "--dim",
        str(int(args_payload.get("dim", 64))),
        "--init_feat",
        str(int(args_payload.get("init_feat", 32))),
        "--batch",
        str(int(args_payload.get("batch", 2))),
        "--slices",
        str(int(slices)),
        "--slice_sampling",
        str(args_payload.get("slice_sampling", "uniform")),
        "--slice_geometry",
        str(args_payload.get("slice_geometry", "ultrasound_probe")),
        "--slice_axis_jitter_deg",
        str(float(args_payload.get("slice_axis_jitter_deg", 20.0))),
        "--slice_fan_half_angle_deg",
        str(float(args_payload.get("slice_fan_half_angle_deg", 30.0))),
        "--slice_elevation_jitter_deg",
        str(float(args_payload.get("slice_elevation_jitter_deg", 5.0))),
        "--slice_sweep_jitter_deg",
        str(float(args_payload.get("slice_sweep_jitter_deg", 2.0))),
        "--probe_pos_sigma_vox",
        str(float(args_payload.get("probe_pos_sigma_vox", 1.0))),
        "--probe_depth_sigma_vox",
        str(float(args_payload.get("probe_depth_sigma_vox", 6.0))),
        "--probe_tilt_sigma",
        str(float(args_payload.get("probe_tilt_sigma", 0.08))),
        "--thickness_vox",
        str(float(args_payload.get("thickness_vox", 1.0))),
    ]

    append_scalar_arg(command, "--slice_mean", args_payload.get("slice_mean"))
    append_scalar_arg(command, "--slice_std", args_payload.get("slice_std"))
    append_vector_arg(command, "--slice_axis", args_payload.get("slice_axis"))
    append_scalar_arg(command, "--max_cases", args_payload.get("max_cases"))

    if bool(args_payload.get("allow_all_data_eval", False)):
        command.append("--allow_all_data_eval")

    seeds = args_payload.get("seeds")
    if isinstance(seeds, list) and seeds:
        command.append("--seeds")
        command.extend(str(int(seed)) for seed in seeds)
    else:
        command.extend(
            [
                "--seed_start",
                str(int(args_payload.get("seed_start", 1337))),
                "--num_repeats",
                str(int(args_payload.get("num_repeats", 5))),
            ]
        )

    if not bool(args_payload.get("hard_constraint", True)):
        command.append("--no_hard_constraint")

    for model in extra["models"]:
        if not isinstance(model, dict):
            continue
        command.extend(["--weights", str(model.get("weights", ""))])
        command.extend(["--model", str(model.get("kind", model.get("model", "")))])
        command.extend(["--label", str(model.get("label", ""))])

    return command


def build_export_command(*, python_exe: str, eval_dir: str, out_dir: str) -> List[str]:
    script_path = os.path.join(ROOT, "scripts", "export_thesis_qualitative_figures.py")
    return [
        python_exe,
        script_path,
        "--eval_dir",
        eval_dir,
        "--out",
        out_dir,
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference_eval_dir", type=str, required=True)
    ap.add_argument("--slices", type=int, nargs="+", required=True)
    ap.add_argument("--eval_dir_prefix", type=str, default="eval_s")
    ap.add_argument("--export_dir_name", type=str, default="thesis_qualitative_pdfs")
    ap.add_argument("--reuse_reference_eval_for_matching_slice", action="store_true")
    args = ap.parse_args()

    reference_eval_dir = os.path.normpath(os.path.abspath(args.reference_eval_dir))
    repro = load_json(os.path.join(reference_eval_dir, "reproducibility.json"))
    args_payload = repro.get("args")
    if not isinstance(args_payload, dict):
        raise RuntimeError("Reference reproducibility.json does not contain an 'args' object.")

    reference_slice = args_payload.get("slices")
    parent_dir = os.path.dirname(reference_eval_dir)
    python_exe = sys.executable

    for slice_count in args.slices:
        slice_count = int(slice_count)
        if args.reuse_reference_eval_for_matching_slice and reference_slice is not None and int(reference_slice) == slice_count:
            eval_dir = reference_eval_dir
        else:
            eval_dir = os.path.join(parent_dir, f"{args.eval_dir_prefix}{slice_count}")
            eval_command = build_eval_command(
                python_exe=python_exe,
                out_dir=eval_dir,
                slices=slice_count,
                repro=repro,
            )
            subprocess.run(eval_command, cwd=ROOT, check=True)

        export_dir = os.path.join(eval_dir, args.export_dir_name)
        export_command = build_export_command(
            python_exe=python_exe,
            eval_dir=eval_dir,
            out_dir=export_dir,
        )
        subprocess.run(export_command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
