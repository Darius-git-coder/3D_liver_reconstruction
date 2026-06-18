# scripts/unit_test_synthetic.py
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from inpainting3d.mask_layers import PartialConv3d
from inpainting3d.losses import MaskedInpaintingLossNormalized


def make_sphere(dim=64, radius=18, sigma=3.0) -> np.ndarray:
    """
    Create a synthetic spherical foreground volume.
    
    Parameters
    ----------
    dim : Any
        Target cubic side length of the processed volume. Defaults to 64.
    radius : Any
        Radius of the synthetic sphere in voxels. Defaults to 18.
    sigma : Any
        Standard deviation of the Gaussian kernel. Defaults to 3.0.
    
    Returns
    -------
    np.ndarray
        Synthetic sphere volume.
    """
    x = np.linspace(-1, 1, dim, dtype=np.float32)
    gx, gy, gz = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(gx**2 + gy**2 + gz**2)
    vol = np.exp(-((r - 0.0) ** 2) / (2 * (sigma / dim) ** 2))
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return vol.astype(np.float32)


def main() -> None:
    """
    Execute the command-line entry point for this script.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    torch.manual_seed(0)
    dim = 64
    gt = make_sphere(dim=dim)

    # Lochmasken: klein vs groß
    mask_small = np.ones((dim, dim, dim), np.float32)
    mask_small[24:40, 24:40, 24:40] = 0.0

    mask_big = np.ones((dim, dim, dim), np.float32)
    mask_big[16:48, 16:48, 16:48] = 0.0

    # Sparse Input: known voxels bleiben, missing werden 0
    sparse_small = gt * mask_small
    sparse_big = gt * mask_big

    # ===== Test A: PartialConv3d =====
    layer = PartialConv3d(1, 1, kernel_size=3, padding=1, bias=False)
    # kernel=ones -> einfacher Erwartungscheck
    with torch.no_grad():
        layer.conv.weight.fill_(1.0)

    x = torch.from_numpy(sparse_small[None, None]).float()
    m = torch.from_numpy(mask_small[None, None]).float()

    y, m_out = layer(x, m)

    # ===== Test B: Normalisierter Loss =====
    loss_fn = MaskedInpaintingLossNormalized(hole_weight=10.0, valid_weight=1.0)

    pred_small = torch.from_numpy(sparse_small[None, None]).float()
    pred_big = torch.from_numpy(sparse_big[None, None]).float()
    target = torch.from_numpy(gt[None, None]).float()

    m_small = torch.from_numpy(mask_small[None, None]).float()
    m_big = torch.from_numpy(mask_big[None, None]).float()

    L_small = loss_fn(pred_small, target, m_small).item()
    L_big = loss_fn(pred_big, target, m_big).item()

    print(f"Normalized loss (small hole): {L_small:.6f}")
    print(f"Normalized loss (big hole)  : {L_big:.6f}")
    print("Interpretation: Beide Losses messen Fehler pro Region, nicht pro Gesamtvolumen.")

    # ===== Visualisierung =====
    out_dir = "./runs/unit_test_synth"
    os.makedirs(out_dir, exist_ok=True)

    mid = dim // 2
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    ax[0, 0].imshow(gt[mid].T, cmap="bone", origin="lower"); ax[0, 0].set_title("GT")
    ax[0, 1].imshow(sparse_small[mid].T, cmap="bone", origin="lower"); ax[0, 1].set_title("Sparse (small hole)")
    ax[0, 2].imshow(mask_small[mid].T, cmap="gray", origin="lower"); ax[0, 2].set_title("Mask")

    ax[1, 0].imshow(y[0, 0, mid].detach().numpy().T, cmap="bone", origin="lower"); ax[1, 0].set_title("PartialConv output")
    ax[1, 1].imshow(m_out[0, 0, mid].detach().numpy().T, cmap="gray", origin="lower"); ax[1, 1].set_title("Updated mask")
    ax[1, 2].axis("off")

    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "partialconv_demo.png"), dpi=200)
    plt.close(fig)

    print(f"Plot gespeichert: {os.path.join(out_dir, 'partialconv_demo.png')}")


if __name__ == "__main__":
    main()
