# scripts/invariance_test.py
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

from inpainting3d.data import load_nifti_volume, normalize_volume, resample_to_shape, center_of_mass_threshold, fibonacci_normals, simulate_sparse_acquisition
from inpainting3d.invariance import apply_rotation, apply_translation, apply_intensity, pca_align_stable
from inpainting3d.metrics import compute_metrics
from inpainting3d.models import PartialResUNet3D
from inpainting3d.utils import ensure_dir, torch_load_weights_compat, strip_dataparallel_prefix, finalize_inpainting_prediction


def run_model(model: torch.nn.Module, sparse: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Run a model on a prepared sparse input.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model instance used for training, inference, or visualization.
    sparse : np.ndarray
        Sparse observation tensor or array.
    mask : np.ndarray
        Binary mask that marks valid or selected voxels.
    device : torch.device
        Torch device on which tensors should be created or evaluated.
    
    Returns
    -------
    np.ndarray
        Output produced by the model or workflow.
    """
    x = np.stack([sparse, mask], axis=0)[None, ...].astype(np.float32)  # [1,2,D,H,W]
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        pred = finalize_inpainting_prediction(model(xt), sparse=xt[:, 0:1, ...], known_mask=xt[:, 1:2, ...]).cpu().numpy()[0, 0]
    return pred.astype(np.float32)


def main() -> None:
    """
    Execute the command-line entry point for this script.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out", type=str, default="./runs/invariance")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--slices", type=int, default=64)
    ap.add_argument("--thickness_vox", type=float, default=1.0)
    args = ap.parse_args()

    ensure_dir(args.out)

    vol, _ = load_nifti_volume(args.volume, canonical=True, dtype=np.float32)
    vol = normalize_volume(vol, "clip01")
    vol = resample_to_shape(vol, (args.dim, args.dim, args.dim), order=1)

    center = center_of_mass_threshold(vol, 0.1)
    normals = fibonacci_normals(args.slices)
    sparse, mask, _hits = simulate_sparse_acquisition(vol, normals, center=center, thickness_vox=args.thickness_vox)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartialResUNet3D().to(device)
    state = torch_load_weights_compat(args.weights, map_location=device)
    state = strip_dataparallel_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    base_pred = run_model(model, sparse, mask, device)

    # Tests: immer Bild + Maske zusammen transformieren
    tests: Dict[str, Dict[str, float]] = {}

    # Rotation
    v_rot, m_rot = apply_rotation(sparse, mask, angle_deg=30, axes=(1, 2))
    pred_rot = run_model(model, v_rot, m_rot, device)

    # Translation
    v_sh, m_sh = apply_translation(sparse, mask, shift_xyz=(4, 4, 4))
    pred_sh = run_model(model, v_sh, m_sh, device)

    # Intensity
    v_in, m_in = apply_intensity(sparse, mask, a=1.1, b=0.0)
    pred_in = run_model(model, v_in, m_in, device)

    # PCA Pipeline Test (optional): apply PCA to input, nicht zum Vergleich zurückdrehen (kanonischer Raum)
    aligned, aligned_mask, tr = pca_align_stable(sparse, mask=mask, thr=0.1)
    pred_pca = run_model(model, aligned, aligned_mask if aligned_mask is not None else mask, device)

    # Metrics (Vergleich gegen base_pred)
    # Wir brauchen torch-Tensoren
    def to_t(x): return torch.from_numpy(x[None, None]).float()
    base_t = to_t(base_pred)

    tests["rotation"] = compute_metrics(to_t(pred_rot), base_t, known_mask=to_t(mask), data_range=1.0)
    tests["translation"] = compute_metrics(to_t(pred_sh), base_t, known_mask=to_t(mask), data_range=1.0)
    tests["intensity"] = compute_metrics(to_t(pred_in), base_t, known_mask=to_t(mask), data_range=1.0)
    tests["pca_aligned"] = compute_metrics(to_t(pred_pca), base_t, known_mask=to_t(mask), data_range=1.0)

    with open(os.path.join(args.out, "invariance_results.json"), "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2)

    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
