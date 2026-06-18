from __future__ import annotations

import glob
import json
import math
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


DEFAULT_SPLIT_NAMES = ("train", "val", "test")


def normalize_case_path(path: str) -> str:
    """
    Normalize a case path into an absolute canonical form.
    
    Parameters
    ----------
    path : str
        Filesystem path to the input artifact.
    
    Returns
    -------
    str
        Normalized or derived path.
    """
    return os.path.normpath(os.path.abspath(path))


def resolve_case_paths(pattern: str) -> List[str]:
    """
    Resolve and sort case paths from a glob pattern.
    
    Parameters
    ----------
    pattern : str
        Glob pattern used to discover input files.
    
    Returns
    -------
    List[str]
        Resolved value or selection.
    """
    return sorted(normalize_case_path(path) for path in glob.glob(pattern))


def case_id_from_path(path: str) -> str:
    """
    Derive a case identifier from a volume file path.
    
    Parameters
    ----------
    path : str
        Filesystem path to the input artifact.
    
    Returns
    -------
    str
        Case identifier derived from the input path.
    """
    name = os.path.basename(path)
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    stem, _ext = os.path.splitext(name)
    return stem


def _allocate_counts(num_cases: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    """
    Allocate split counts that match the requested ratios as closely as possible.
    
    Parameters
    ----------
    num_cases : int
        Maximum number of cases to select.
    ratios : Mapping[str, float]
        Mapping from split names to target ratios.
    
    Returns
    -------
    Dict[str, int]
        Allocated case counts for each requested split.
    """
    if num_cases <= 0:
        raise ValueError("Cannot create a split without cases.")

    raw = {name: float(ratios.get(name, 0.0)) * num_cases for name in DEFAULT_SPLIT_NAMES}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = num_cases - sum(counts.values())

    order = sorted(
        DEFAULT_SPLIT_NAMES,
        key=lambda name: (raw[name] - counts[name], raw[name], name),
        reverse=True,
    )
    for idx in range(remaining):
        counts[order[idx % len(order)]] += 1

    non_zero_priority = [
        name
        for name in sorted(
            DEFAULT_SPLIT_NAMES,
            key=lambda split_name: (float(ratios.get(split_name, 0.0)), -DEFAULT_SPLIT_NAMES.index(split_name)),
            reverse=True,
        )
        if float(ratios.get(name, 0.0)) > 0.0
    ]
    required_non_empty = non_zero_priority[: min(num_cases, len(non_zero_priority))]
    for name in required_non_empty:
        if counts.get(name, 0) > 0:
            continue
        donor_candidates = sorted(
            DEFAULT_SPLIT_NAMES,
            key=lambda split_name: (counts.get(split_name, 0), float(ratios.get(split_name, 0.0))),
            reverse=True,
        )
        for donor in donor_candidates:
            if donor != name and counts.get(donor, 0) > 1:
                counts[donor] -= 1
                counts[name] = counts.get(name, 0) + 1
                break

    if sum(counts.values()) != num_cases:
        raise RuntimeError("Split allocation failed to conserve the number of cases.")
    return counts


def create_split_manifest(
    files: Sequence[str],
    seed: int = 1337,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    source_glob: str | None = None,
    project_root: str | None = None,
    git_info: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """
    Create a manifest that records deterministic train, validation, and test splits.
    
    Parameters
    ----------
    files : Sequence[str]
        Sequence of input case files.
    seed : int
        Random seed used for reproducible sampling. Defaults to 1337.
    train_ratio : float
        Relative fraction assigned to the training split. Defaults to 0.7.
    val_ratio : float
        Relative fraction assigned to the validation split. Defaults to 0.15.
    test_ratio : float
        Relative fraction assigned to the test split. Defaults to 0.15.
    source_glob : str | None
        Original glob pattern used to collect the cases. Defaults to None.
    project_root : str | None
        Project root recorded in the manifest. Defaults to None.
    git_info : Mapping[str, object] | None
        Optional git metadata stored in the manifest. Defaults to None.
    
    Returns
    -------
    Dict[str, object]
        Newly created object or artifact.
    """
    normalized_files = [normalize_case_path(path) for path in files]
    unique_files = sorted(dict.fromkeys(normalized_files))
    if len(unique_files) != len(normalized_files):
        raise ValueError("Duplicate case paths detected while creating the split manifest.")
    if not unique_files:
        raise ValueError("No files available for split creation.")

    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be positive.")
    ratios = {name: value / ratio_sum for name, value in ratios.items()}

    if len(unique_files) == 1:
        split_files = {"train": unique_files[:], "val": unique_files[:], "test": unique_files[:]}
    else:
        counts = _allocate_counts(len(unique_files), ratios)
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(unique_files))
        shuffled = [unique_files[int(idx)] for idx in order]

        split_files: Dict[str, List[str]] = {}
        start = 0
        for split_name in DEFAULT_SPLIT_NAMES:
            count = counts[split_name]
            split_files[split_name] = shuffled[start:start + count]
            start += count

    split_case_ids = {
        split_name: [case_id_from_path(path) for path in paths]
        for split_name, paths in split_files.items()
    }

    manifest: Dict[str, object] = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "source_glob": source_glob,
        "source_case_count": int(len(unique_files)),
        "project_root": normalize_case_path(project_root) if project_root else None,
        "ratios": ratios,
        "counts": {name: int(len(paths)) for name, paths in split_files.items()},
        "splits": split_files,
        "case_ids": split_case_ids,
    }
    if git_info:
        manifest["git"] = dict(git_info)
    return manifest


def load_split_manifest(path: str) -> Dict[str, object]:
    """
    Load and normalize a split manifest from disk.
    
    Parameters
    ----------
    path : str
        Filesystem path to the input artifact.
    
    Returns
    -------
    Dict[str, object]
        Loaded data structure.
    """
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    splits = manifest.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError(f"Split manifest {path} is missing a valid 'splits' mapping.")

    normalized_splits: Dict[str, List[str]] = {}
    for split_name, paths in splits.items():
        if not isinstance(paths, list):
            raise ValueError(f"Split '{split_name}' in {path} must be a list of case paths.")
        normalized_splits[split_name] = [normalize_case_path(case_path) for case_path in paths]

    manifest["splits"] = normalized_splits
    return manifest


def get_split_files(manifest: Mapping[str, object], split_name: str) -> List[str]:
    """
    Return the case paths for a named split.
    
    Parameters
    ----------
    manifest : Mapping[str, object]
        Split or evaluation manifest to serialize or inspect.
    split_name : str
        Name of the requested data split.
    
    Returns
    -------
    List[str]
        Normalized case paths for the requested split.
    """
    splits = manifest.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError("Split manifest does not contain a 'splits' mapping.")
    if split_name not in splits:
        available = ", ".join(sorted(str(name) for name in splits.keys()))
        raise KeyError(f"Split '{split_name}' not found. Available splits: {available}")
    selected = splits[split_name]
    if not isinstance(selected, list):
        raise ValueError(f"Split '{split_name}' is not a list.")
    return [normalize_case_path(path) for path in selected]


def filter_to_known_files(paths: Iterable[str], available_files: Sequence[str]) -> List[str]:
    """
    Validate that all referenced split files exist in the current file set.
    
    Parameters
    ----------
    paths : Iterable[str]
        Sequence of filesystem paths to process.
    available_files : Sequence[str]
        Files that are currently available for the dataset.
    
    Returns
    -------
    List[str]
        Selected filesystem paths.
    """
    available = {normalize_case_path(path) for path in available_files}
    resolved = [normalize_case_path(path) for path in paths]
    missing = [path for path in resolved if path not in available]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            "The split manifest references files that are not part of the current data glob: "
            f"{preview}"
        )
    return resolved


def write_split_manifest(manifest: Mapping[str, object], out_path: str) -> None:
    """
    Write a split manifest and plain-text split lists to disk.
    
    Parameters
    ----------
    manifest : Mapping[str, object]
        Split or evaluation manifest to serialize or inspect.
    out_path : str
        Destination path for the written artifact.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    stem = os.path.splitext(out_path)[0]
    splits = manifest.get("splits", {})
    if isinstance(splits, dict):
        for split_name, paths in splits.items():
            txt_path = f"{stem}_{split_name}.txt"
            with open(txt_path, "w", encoding="utf-8") as handle:
                for case_path in paths:
                    handle.write(f"{case_path}\n")
