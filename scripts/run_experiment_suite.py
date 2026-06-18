from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from copy import deepcopy
from typing import Any, Dict, Iterable, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inpainting3d.repro import build_repro_metadata
from inpainting3d.stats import summary_rows_from_nested


def save_json(path: str, payload: dict) -> None:
    """
    Write a JSON document to disk.
    
    Parameters
    ----------
    path : str
        Filesystem path to the input artifact.
    payload : dict
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


TRAIN_BOOL_FLAGS = {
    "amp": "--amp",
    "curriculum": "--curriculum",
    "deterministic": "--deterministic",
}
TRAIN_NEGATED_BOOL_FLAGS = {
    "hard_constraint": "--no_hard_constraint",
}
EVAL_NEGATED_BOOL_FLAGS = {
    "hard_constraint": "--no_hard_constraint",
}
EVAL_INHERITED_KEYS = ("init_feat",)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load an experiment-suite configuration file.
    
    Parameters
    ----------
    path : str
        Filesystem path to the input artifact.
    
    Returns
    -------
    Dict[str, Any]
        Loaded data structure.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_dicts(base: Dict[str, Any], updates: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Merge dictionaries recursively or by overwrite semantics.
    
    Parameters
    ----------
    base : Dict[str, Any]
        Base dictionary or configuration object.
    updates : Dict[str, Any] | None
        Updated values merged into a base dictionary.
    
    Returns
    -------
    Dict[str, Any]
        Merged dicts.
    """
    merged = deepcopy(base)
    if updates:
        merged.update(updates)
    return merged


def append_cli_args(
    command: List[str],
    values: Dict[str, Any],
    *,
    positive_bool_flags: Dict[str, str] | None = None,
    negative_bool_flags: Dict[str, str] | None = None,
) -> None:
    """
    Append command-line arguments derived from a configuration dictionary.
    
    Parameters
    ----------
    command : List[str]
        Command sequence that should be executed or recorded.
    values : Dict[str, Any]
        Numeric values that should be summarized.
    positive_bool_flags : Dict[str, str] | None
        Boolean flags that are appended when their values are true. Defaults to None.
    negative_bool_flags : Dict[str, str] | None
        Boolean flags that are appended when their values are false. Defaults to None.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    positive_bool_flags = positive_bool_flags or {}
    negative_bool_flags = negative_bool_flags or {}

    for key, value in values.items():
        if value is None:
            continue
        if key in positive_bool_flags:
            if bool(value):
                command.append(positive_bool_flags[key])
            continue
        if key in negative_bool_flags:
            if not bool(value):
                command.append(negative_bool_flags[key])
            continue
        command.extend([f"--{key}", str(value)])


def run_command(command: Iterable[str], cwd: str) -> None:
    """
    Run a subprocess command for an experiment step.
    
    Parameters
    ----------
    command : Iterable[str]
        Command sequence that should be executed or recorded.
    cwd : str
        Working directory used for command execution or metadata lookup.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    subprocess.run(list(command), cwd=cwd, check=True)


def main() -> None:
    """
    Execute the command-line entry point for this script.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    project_cfg = config.get("project", {})
    train_defaults = config.get("train_defaults", {})
    eval_defaults = config.get("eval_defaults", {})
    experiments = config.get("experiments", [])
    if not experiments:
        raise RuntimeError("Die Suite-Konfiguration enthaelt keine Experimente.")

    runs_root = os.path.normpath(project_cfg.get("runs_root", "./runs/experiment_suite"))
    os.makedirs(runs_root, exist_ok=True)
    python_exe = project_cfg.get("python_exe") or sys.executable
    data_glob = project_cfg["data_glob"]
    split_file = project_cfg["split_file"]
    eval_split = project_cfg.get("eval_split", "test")

    executed: List[Dict[str, Any]] = []
    suite_rows: List[Dict[str, Any]] = []

    for experiment in experiments:
        name = experiment["name"]
        train_cfg = merge_dicts(train_defaults, experiment.get("train", {}))
        eval_cfg = merge_dicts(eval_defaults, experiment.get("eval", {}))
        model_name = str(train_cfg.get("model", experiment.get("model", "partial")))

        train_out = os.path.join(runs_root, name, "train")
        eval_out = os.path.join(runs_root, name, "eval")
        weights_path = eval_cfg.get("weights") or os.path.join(train_out, "best_model.pt")

        train_command = [
            python_exe,
            "scripts/train_inpainting.py",
            "--data",
            data_glob,
            "--out",
            train_out,
            "--split_file",
            split_file,
            "--model",
            model_name,
        ]
        append_cli_args(
            train_command,
            {key: value for key, value in train_cfg.items() if key != "model"},
            positive_bool_flags=TRAIN_BOOL_FLAGS,
            negative_bool_flags=TRAIN_NEGATED_BOOL_FLAGS,
        )

        eval_command = [
            python_exe,
            "scripts/evaluate_inpainting.py",
            "--data",
            data_glob,
            "--weights",
            weights_path,
            "--out",
            eval_out,
            "--split_file",
            split_file,
            "--split",
            eval_split,
            "--model",
            model_name,
            "--label",
            name,
        ]
        append_cli_args(
            eval_command,
            {key: value for key, value in eval_cfg.items() if key != "weights"},
            negative_bool_flags=EVAL_NEGATED_BOOL_FLAGS,
        )
        if "hard_constraint" in train_cfg and "hard_constraint" not in eval_cfg:
            append_cli_args(eval_command, {"hard_constraint": train_cfg["hard_constraint"]}, negative_bool_flags=EVAL_NEGATED_BOOL_FLAGS)
        inherited_eval_values = {
            key: train_cfg[key]
            for key in EVAL_INHERITED_KEYS
            if key in train_cfg and key not in eval_cfg
        }
        if inherited_eval_values:
            append_cli_args(eval_command, inherited_eval_values, negative_bool_flags=EVAL_NEGATED_BOOL_FLAGS)

        executed.append(
            {
                "experiment": name,
                "train_command": train_command,
                "eval_command": eval_command,
                "weights_path": os.path.normpath(weights_path),
            }
        )

        if not args.skip_train:
            run_command(train_command, cwd=ROOT)
        if not args.skip_eval:
            run_command(eval_command, cwd=ROOT)

        summary_path = os.path.join(eval_out, "metrics_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for row in summary_rows_from_nested(payload.get("summary", {})):
                suite_rows.append(
                    {
                        "experiment": name,
                        "model": model_name,
                        "weights_path": os.path.normpath(weights_path),
                        **row,
                    }
                )

    save_json(os.path.join(runs_root, "suite_commands.json"), executed)
    save_json(
        os.path.join(runs_root, "suite_reproducibility.json"),
        build_repro_metadata(vars(args), extra={"config": os.path.normpath(os.path.abspath(args.config))}, cwd=ROOT),
    )

    if suite_rows:
        write_rows_csv(os.path.join(runs_root, "suite_summary.csv"), suite_rows)


if __name__ == "__main__":
    main()
