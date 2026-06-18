from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch

from inpainting3d.data import (
    center_of_mass_threshold,
    load_nifti_volume,
    normalize_volume,
    random_normals,
    resample_to_shape,
    simulate_sparse_acquisition,
)
from inpainting3d.splits import filter_to_known_files, get_split_files, load_split_manifest, resolve_case_paths, write_split_manifest
from inpainting3d.utils import (
    ensure_dir,
    finalize_inpainting_prediction,
    strip_dataparallel_prefix,
    torch_load_weights_compat,
)
from scripts.train_inpainting import build_model


def sync_device(device: torch.device) -> None:
    """
    Synchronize the active CUDA device when needed.
    
    Parameters
    ----------
    device : torch.device
        Torch device on which tensors should be created or evaluated.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prepare_input(
    path: str,
    dim: int,
    slices: int,
    thickness_vox: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one case and build a sparse benchmark input.
    
    Parameters
    ----------
    path : str
        Filesystem path to the input artifact.
    dim : int
        Target cubic side length of the processed volume.
    slices : int
        Number of slice planes to sample.
    thickness_vox : float
        Half thickness of each sampled slice plane in voxels.
    seed : int
        Random seed used for reproducible sampling.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sparse input volume and the corresponding observed-voxel mask.
    """
    rng = np.random.default_rng(seed)
    vol, _ = load_nifti_volume(path, canonical=True, dtype=np.float32)
    vol = normalize_volume(vol, "clip01")
    vol = resample_to_shape(vol, (dim, dim, dim), order=1)
    center = center_of_mass_threshold(vol, thr=0.1)
    normals = random_normals(slices, rng)
    sparse, mask, _ = simulate_sparse_acquisition(
        vol,
        normals=normals,
        center=center,
        thickness_vox=thickness_vox,
        chunk=64,
    )
    return sparse.astype(np.float32), mask.astype(np.float32)


def measure_case(
    model: torch.nn.Module,
    device: torch.device,
    path: str,
    dim: int,
    slices: int,
    thickness_vox: float,
    seed: int,
    warmup: int,
) -> dict[str, Any]:
    """
    Measure preprocessing and inference latency for one benchmark case.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model instance used for training, inference, or visualization.
    device : torch.device
        Torch device on which tensors should be created or evaluated.
    path : str
        Filesystem path to the input artifact.
    dim : int
        Target cubic side length of the processed volume.
    slices : int
        Number of slice planes to sample.
    thickness_vox : float
        Half thickness of each sampled slice plane in voxels.
    seed : int
        Random seed used for reproducible sampling.
    warmup : int
        Number of warm-up iterations executed before timing.
    
    Returns
    -------
    dict[str, Any]
        Measured case.
    """
    t0 = time.perf_counter()
    sparse, mask = prepare_input(path, dim=dim, slices=slices, thickness_vox=thickness_vox, seed=seed)
    t1 = time.perf_counter()

    x = np.stack([sparse, mask], axis=0)[None, ...].astype(np.float32)
    xt = torch.from_numpy(x).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            pred = model(xt)
            _ = finalize_inpainting_prediction(pred, sparse=xt[:, 0:1, ...], known_mask=xt[:, 1:2, ...])
        sync_device(device)

        t2 = time.perf_counter()
        pred = model(xt)
        pred = finalize_inpainting_prediction(pred, sparse=xt[:, 0:1, ...], known_mask=xt[:, 1:2, ...])
        sync_device(device)
        t3 = time.perf_counter()

    return {
        "file": os.path.basename(path),
        "path": os.path.normpath(path),
        "preprocess_ms": (t1 - t0) * 1000.0,
        "inference_ms": (t3 - t2) * 1000.0,
        "total_ms": (t3 - t0) * 1000.0,
        "voxels_known_frac": float(mask.mean()),
        "pred_mean": float(pred.mean().item()),
    }


def resolve_benchmark_files(args: argparse.Namespace) -> list[str]:
    """
    Resolve the benchmark cases selected by the command-line arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed to the helper or command.
    
    Returns
    -------
    list[str]
        Resolved value or selection.
    """
    available_files = resolve_case_paths(args.data)
    if not available_files:
        raise RuntimeError("Keine Dateien gefunden.")

    if not args.split_file:
        return available_files

    manifest = load_split_manifest(args.split_file)
    selected_files = filter_to_known_files(get_split_files(manifest, args.split), available_files)
    if not selected_files:
        raise RuntimeError(f"Split '{args.split}' enthaelt keine Benchmark-Faelle.")
    manifest["selected_splits"] = {"benchmark": args.split}
    write_split_manifest(manifest, os.path.join(args.out, "split_manifest.json"))
    return selected_files


def main() -> None:
    """
    Execute the command-line entry point for this script.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Glob pattern for volumes.")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out", type=str, default="./runs/benchmark_inference")
    ap.add_argument("--model", type=str, default="partial", choices=["baseline", "partial"])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--init_feat", type=int, default=32)
    ap.add_argument("--slices", type=int, default=32)
    ap.add_argument("--thickness_vox", type=float, default=1.0)
    ap.add_argument("--num_cases", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--split_file", type=str, default="")
    ap.add_argument("--split", type=str, default="test")
    args = ap.parse_args()

    ensure_dir(args.out)

    files = sorted(resolve_benchmark_files(args))
    if not files:
        raise RuntimeError("Keine Dateien gefunden.")

    rng = random.Random(args.seed)
    num_cases = min(args.num_cases, len(files))
    selected = rng.sample(files, k=num_cases)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, init_feat=args.init_feat).to(device)
    state = torch_load_weights_compat(args.weights, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not (isinstance(state, dict) and any(isinstance(v, torch.Tensor) for v in state.values())):
        raise RuntimeError("Gewichte haben unerwartetes Format. Erwartet state_dict.")
    state = strip_dataparallel_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    rows: list[dict[str, Any]] = []
    for idx, path in enumerate(selected):
        row = measure_case(
            model=model,
            device=device,
            path=path,
            dim=args.dim,
            slices=args.slices,
            thickness_vox=args.thickness_vox,
            seed=args.seed + idx,
            warmup=args.warmup,
        )
        rows.append(row)
        print(
            f"{idx + 1:02d}/{num_cases:02d} {row['file']}: "
            f"pre={row['preprocess_ms']:.2f} ms "
            f"inf={row['inference_ms']:.2f} ms "
            f"total={row['total_ms']:.2f} ms"
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "benchmark_cases.csv"), index=False)

    summary = {
        "device": str(device),
        "model": args.model,
        "weights": os.path.normpath(args.weights),
        "split": args.split if args.split_file else "all",
        "num_cases": int(num_cases),
        "dim": int(args.dim),
        "init_feat": int(args.init_feat),
        "slices": int(args.slices),
        "thickness_vox": float(args.thickness_vox),
        "warmup": int(args.warmup),
        "preprocess_ms_mean": float(df["preprocess_ms"].mean()),
        "preprocess_ms_std": float(df["preprocess_ms"].std(ddof=0)),
        "inference_ms_mean": float(df["inference_ms"].mean()),
        "inference_ms_std": float(df["inference_ms"].std(ddof=0)),
        "total_ms_mean": float(df["total_ms"].mean()),
        "total_ms_std": float(df["total_ms"].std(ddof=0)),
        "selected_files": [os.path.basename(path) for path in selected],
    }
    with open(os.path.join(args.out, "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
