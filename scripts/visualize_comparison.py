from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from inpainting3d.data import (
    center_of_mass_threshold,
    fibonacci_normals,
    load_nifti_volume,
    normalize_volume,
    resample_to_shape,
    simulate_sparse_acquisition,
)
from inpainting3d.metrics import compute_metrics
from inpainting3d.utils import ensure_dir, strip_dataparallel_prefix, torch_load_weights_compat, finalize_inpainting_prediction
from scripts.train_inpainting import build_model


def save_figure(fig: plt.Figure, out_path: str, dpi: int = 220) -> None:
    fig.savefig(os.path.normpath(out_path), dpi=dpi)


def kernel_regression_fill(
    sparse: np.ndarray,
    mask: np.ndarray,
    sigma: float = 2.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Simple continuous baseline: Gaussian kernel regression on known voxels."""
    weighted = gaussian_filter(sparse * mask, sigma=sigma)
    support = gaussian_filter(mask, sigma=sigma)
    pred = np.divide(weighted, support + eps, out=np.zeros_like(weighted), where=support > eps)
    pred = np.where(mask > 0.5, sparse, pred)
    return np.clip(pred, 0.0, 1.0).astype(np.float32)


def run_model(
    volume: np.ndarray,
    mask: np.ndarray,
    weights: str,
    device: torch.device,
    model_kind: str,
    init_feat: int,
) -> np.ndarray:
    model = build_model(model_kind, init_feat=init_feat).to(device)
    state = torch_load_weights_compat(weights, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    state = strip_dataparallel_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    x = np.stack([volume, mask], axis=0)[None, ...].astype(np.float32)
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        pred = finalize_inpainting_prediction(model(xt), sparse=xt[:, 0:1, ...], known_mask=xt[:, 1:2, ...]).float().cpu().numpy()[0, 0]
    return np.clip(pred, 0.0, 1.0).astype(np.float32)


def to_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pred_t = torch.from_numpy(pred[None, None]).float()
    target_t = torch.from_numpy(target[None, None]).float()
    mask_t = torch.from_numpy(mask[None, None]).float()
    return compute_metrics(pred_t, target_t, known_mask=mask_t, data_range=1.0)


def plot_volume_surface(ax, vol: np.ndarray, thr: float, title: str, color: str) -> None:
    coords = np.argwhere(vol >= thr)
    if coords.shape[0] == 0:
        ax.set_title(f"{title}\n(no voxels >= {thr:.2f})")
        ax.set_axis_off()
        return

    if coords.shape[0] > 12000:
        step = max(1, coords.shape[0] // 12000)
        coords = coords[::step]

    ax.scatter(
        coords[:, 2],
        coords[:, 1],
        coords[:, 0],
        s=2,
        c=color,
        alpha=0.22,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("W")
    ax.set_ylabel("H")
    ax.set_zlabel("D")
    ax.view_init(elev=22, azim=35)


def plot_error_cloud(ax, err: np.ndarray, title: str, cmap: str = "inferno") -> None:
    thresh = float(np.quantile(err, 0.97))
    coords = np.argwhere(err >= thresh)
    vals = err[err >= thresh]
    if coords.shape[0] == 0:
        ax.set_title(f"{title}\n(no error voxels)")
        ax.set_axis_off()
        return None

    if coords.shape[0] > 14000:
        step = max(1, coords.shape[0] // 14000)
        coords = coords[::step]
        vals = vals[::step]

    scatter = ax.scatter(
        coords[:, 2],
        coords[:, 1],
        coords[:, 0],
        c=vals,
        cmap=cmap,
        s=3,
        alpha=0.35,
        linewidths=0,
        norm=Normalize(vmin=0.0, vmax=max(float(vals.max()), 1e-6)),
    )
    ax.set_title(f"{title}\nTop 3% abs. error voxels")
    ax.set_xlabel("W")
    ax.set_ylabel("H")
    ax.set_zlabel("D")
    ax.view_init(elev=24, azim=40)
    return scatter


def save_3d_models(gt: np.ndarray, unet: np.ndarray, reg: np.ndarray, out_path: str) -> None:
    thr = max(0.15, float(np.quantile(gt, 0.75)))
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    plot_volume_surface(ax1, gt, thr, "GT 3D", "#7a0019")
    plot_volume_surface(ax2, unet, thr, "U-Net 3D", "#005f73")
    plot_volume_surface(ax3, reg, thr, "Regression 3D", "#ee9b00")
    plt.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def save_mip_comparison(
    gt: np.ndarray,
    sparse: np.ndarray,
    unet: np.ndarray,
    reg: np.ndarray,
    out_path: str,
) -> None:
    volumes = [
        ("GT", gt),
        ("Sparse Input", sparse),
        ("U-Net", unet),
        ("Regression", reg),
    ]
    projections = [
        ("Axial", lambda x: x.max(axis=0)),
        ("Coronal", lambda x: x.max(axis=1)),
        ("Sagittal", lambda x: x.max(axis=2)),
    ]

    vmax = float(
        max(
            np.quantile(gt, 0.995),
            np.quantile(sparse, 0.995),
            np.quantile(unet, 0.995),
            np.quantile(reg, 0.995),
            1e-3,
        )
    )

    panels = [
        (f"{proj_name} {vol_name} MIP", projector(vol))
        for proj_name, projector in projections
        for vol_name, vol in volumes
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    for ax, (title, img) in zip(axes.ravel(), panels):
        ax.imshow(img.T, cmap="bone", origin="lower", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.04, wspace=0.04, hspace=0.10)
    save_figure(fig, out_path)
    plt.close(fig)


def save_error_maps(
    err_unet: np.ndarray,
    err_reg: np.ndarray,
    out_path: str,
) -> None:
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sc1 = plot_error_cloud(ax1, err_unet, "U-Net Absolute Error 3D")
    sc2 = plot_error_cloud(ax2, err_reg, "Regression Absolute Error 3D")
    fig.subplots_adjust(left=0.03, right=0.92, top=0.90, bottom=0.05, wspace=0.08)
    ref_scatter = sc2 if sc2 is not None else sc1
    if ref_scatter is not None:
        cbar = fig.colorbar(ref_scatter, ax=[ax1, ax2], fraction=0.03, pad=0.04)
        cbar.set_label("Absolute Error")
    save_figure(fig, out_path)
    plt.close(fig)


def save_slice_overview(
    gt: np.ndarray,
    unet: np.ndarray,
    reg: np.ndarray,
    err_unet: np.ndarray,
    err_reg: np.ndarray,
    out_path: str,
) -> None:
    mid = gt.shape[0] // 2
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    panels = [
        ("GT Mid Slice", gt[mid]),
        ("U-Net Mid Slice", unet[mid]),
        ("Regression Mid Slice", reg[mid]),
        ("|GT - U-Net|", err_unet[mid]),
        ("|GT - Regression|", err_reg[mid]),
        ("Error Difference (Reg - U-Net)", err_reg[mid] - err_unet[mid]),
    ]
    cmaps = ["bone", "bone", "bone", "inferno", "inferno", "coolwarm"]
    for ax, (title, img), cmap in zip(axes.ravel(), panels, cmaps):
        ax.imshow(img.T, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def save_density_analysis(
    gt: np.ndarray,
    unet: np.ndarray,
    reg: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    title_prefix: str = "Missing-Region",
) -> None:
    missing = mask < 0.5
    _save_density_analysis_arrays(gt[missing], unet[missing], reg[missing], out_path, title_prefix)


def _save_density_analysis_arrays(
    gt_vals: np.ndarray,
    unet_vals: np.ndarray,
    reg_vals: np.ndarray,
    out_path: str,
    title_prefix: str,
) -> None:
    if gt_vals.size == 0:
        return

    hi = float(
        max(
            np.quantile(gt_vals, 0.995),
            np.quantile(unet_vals, 0.995),
            np.quantile(reg_vals, 0.995),
            1e-3,
        )
    )
    bins = np.linspace(0.0, hi, 80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(gt_vals, bins=bins, density=True, alpha=0.55, label="GT", color="#7a0019")
    axes[0].hist(unet_vals, bins=bins, density=True, alpha=0.55, label="U-Net", color="#005f73")
    axes[0].hist(reg_vals, bins=bins, density=True, alpha=0.55, label="Regression", color="#ee9b00")
    axes[0].set_title(f"{title_prefix} Density Histogram")
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    if gt_vals.size > 25000:
        idx = np.linspace(0, gt_vals.size - 1, 25000, dtype=int)
        gt_s = gt_vals[idx]
        unet_s = unet_vals[idx]
        reg_s = reg_vals[idx]
    else:
        gt_s = gt_vals
        unet_s = unet_vals
        reg_s = reg_vals

    lim = hi
    axes[1].scatter(gt_s, reg_s, s=5, alpha=0.12, color="#ee9b00", label="Regression")
    axes[1].scatter(gt_s, unet_s, s=5, alpha=0.12, color="#005f73", label="U-Net")
    axes[1].plot([0, lim], [0, lim], "--", color="black", linewidth=1.0, label="Ideal")
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    axes[1].set_title(f"{title_prefix} GT vs Prediction")
    axes[1].set_xlabel("GT Intensity")
    axes[1].set_ylabel("Predicted Intensity")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def save_stacked_distributions(
    gt_vals: np.ndarray,
    unet_vals: np.ndarray,
    reg_vals: np.ndarray,
    out_path: str,
    title_prefix: str,
) -> None:
    if gt_vals.size == 0:
        return

    hi = float(
        max(
            np.quantile(gt_vals, 0.995),
            np.quantile(unet_vals, 0.995),
            np.quantile(reg_vals, 0.995),
            1e-3,
        )
    )
    bins = np.linspace(0.0, hi, 80)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    panels = [
        ("GT", gt_vals, "#7a0019"),
        ("U-Net", unet_vals, "#005f73"),
        ("Regression", reg_vals, "#ee9b00"),
    ]

    for ax, (label, values, color) in zip(axes, panels):
        ax.hist(values, bins=bins, density=True, alpha=0.8, color=color)
        ax.set_title(f"{title_prefix} {label}")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Intensity")
    plt.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def save_density_analysis_without_background(
    gt: np.ndarray,
    unet: np.ndarray,
    reg: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    foreground_thr: float,
) -> None:
    missing = mask < 0.5
    foreground = gt > foreground_thr
    valid = missing & foreground
    _save_density_analysis_arrays(
        gt[valid],
        unet[valid],
        reg[valid],
        out_path,
        title_prefix="Missing-Region (No Background)",
    )


def save_stacked_distributions_without_background(
    gt: np.ndarray,
    unet: np.ndarray,
    reg: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    foreground_thr: float,
) -> None:
    missing = mask < 0.5
    foreground = gt > foreground_thr
    valid = missing & foreground
    save_stacked_distributions(
        gt[valid],
        unet[valid],
        reg[valid],
        out_path,
        title_prefix="Missing-Region (No Background)",
    )


def save_difference_mips(
    gt: np.ndarray,
    unet: np.ndarray,
    reg: np.ndarray,
    out_path: str,
) -> None:
    err_unet = np.abs(gt - unet)
    err_reg = np.abs(gt - reg)

    mip_panels = [
        ("Axial |GT - U-Net| MIP", err_unet.max(axis=0)),
        ("Coronal |GT - U-Net| MIP", err_unet.max(axis=1)),
        ("Sagittal |GT - U-Net| MIP", err_unet.max(axis=2)),
        ("Axial |GT - Regression| MIP", err_reg.max(axis=0)),
        ("Coronal |GT - Regression| MIP", err_reg.max(axis=1)),
        ("Sagittal |GT - Regression| MIP", err_reg.max(axis=2)),
    ]
    vmax = float(max(panel.max() for _, panel in mip_panels) if mip_panels else 1e-6)
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    im = None
    for ax, (title, img) in zip(axes.ravel(), mip_panels):
        im = ax.imshow(img.T, cmap="inferno", origin="lower", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.14, wspace=0.04, hspace=0.08)
    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.06,
    )
    cbar.set_label("Absolute Error")
    save_figure(fig, out_path)
    plt.close(fig)


def save_windowed_slices(
    gt: np.ndarray,
    unet: np.ndarray,
    reg: np.ndarray,
    out_path: str,
) -> None:
    mid_d = gt.shape[0] // 2
    mid_h = gt.shape[1] // 2
    mid_w = gt.shape[2] // 2
    vmax = float(max(np.quantile(gt, 0.995), np.quantile(unet, 0.995), np.quantile(reg, 0.995), 1e-3))

    slices = [
        ("Axial GT", gt[mid_d], "bone"),
        ("Axial U-Net", unet[mid_d], "bone"),
        ("Axial Regression", reg[mid_d], "bone"),
        ("Coronal GT", gt[:, mid_h, :], "bone"),
        ("Coronal U-Net", unet[:, mid_h, :], "bone"),
        ("Coronal Regression", reg[:, mid_h, :], "bone"),
        ("Sagittal GT", gt[:, :, mid_w], "bone"),
        ("Sagittal U-Net", unet[:, :, mid_w], "bone"),
        ("Sagittal Regression", reg[:, :, mid_w], "bone"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, (title, img, cmap) in zip(axes.ravel(), slices):
        ax.imshow(img.T, cmap=cmap, origin="lower", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model", type=str, default="partial", choices=["baseline", "gated", "partial"])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--init_feat", type=int, default=32)
    ap.add_argument("--slices", type=int, default=64)
    ap.add_argument("--thickness_vox", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma for regression baseline.")
    args = ap.parse_args()

    ensure_dir(args.out)

    gt, _ = load_nifti_volume(args.volume, canonical=True, dtype=np.float32)
    gt = normalize_volume(gt, "clip01")
    gt = resample_to_shape(gt, (args.dim, args.dim, args.dim), order=1)

    center = center_of_mass_threshold(gt, thr=0.1)
    normals = fibonacci_normals(args.slices)
    sparse, mask, _ = simulate_sparse_acquisition(gt, normals, center=center, thickness_vox=args.thickness_vox)

    reg = kernel_regression_fill(sparse, mask, sigma=args.sigma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = run_model(sparse, mask, args.weights, device, model_kind=args.model, init_feat=args.init_feat)

    err_unet = np.abs(gt - unet).astype(np.float32)
    err_reg = np.abs(gt - reg).astype(np.float32)

    save_3d_models(gt, unet, reg, os.path.join(args.out, "comparison_3d_models.png"))
    save_mip_comparison(gt, sparse, unet, reg, os.path.join(args.out, "comparison_mip.png"))
    save_error_maps(err_unet, err_reg, os.path.join(args.out, "comparison_error_3d.png"))
    save_slice_overview(
        gt, unet, reg, err_unet, err_reg, os.path.join(args.out, "comparison_slices.png")
    )
    save_density_analysis(
        gt, unet, reg, mask, os.path.join(args.out, "comparison_density_analysis.png")
    )
    save_density_analysis_without_background(
        gt,
        unet,
        reg,
        mask,
        os.path.join(args.out, "comparison_density_analysis_no_background.png"),
        foreground_thr=1e-3,
    )
    save_stacked_distributions_without_background(
        gt,
        unet,
        reg,
        mask,
        os.path.join(args.out, "density_stacked_nobg.png"),
        foreground_thr=1e-3,
    )
    save_difference_mips(
        gt, unet, reg, os.path.join(args.out, "comparison_difference_mip.png")
    )
    save_windowed_slices(
        gt, unet, reg, os.path.join(args.out, "comparison_windowed_slices.png")
    )

    metrics = {
        "unet": to_metrics(unet, gt, mask),
        "regression": to_metrics(reg, gt, mask),
    }
    with open(os.path.join(args.out, "metrics_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({"out": args.out, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
