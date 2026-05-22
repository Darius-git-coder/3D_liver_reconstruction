from __future__ import annotations

import argparse
import gc
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, zoom
from tqdm import tqdm

DEFAULT_THRESHOLD = 0.05
DEFAULT_TARGET_DIM = 64


def load_nifti_volume(file_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load a NIfTI volume and convert it to canonical orientation."""
    nii = nib.load(str(file_path))
    nii = nib.as_closest_canonical(nii)
    volume = nii.get_fdata().astype(np.float32)
    spacing = np.asarray(nii.header.get_zooms()[:3], dtype=np.float32)
    return volume, spacing, nii.affine, nii.header


def normalize_ct_volume(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
    min_hu: float = -100.0,
    max_hu: float = 400.0,
) -> np.ndarray:
    """Apply CT windowing and normalize intensities to [0, 1]."""
    img = np.clip(volume, min_hu, max_hu)
    img = (img - min_hu) / max(max_hu - min_hu, 1e-8)
    if mask is not None:
        img[mask == 0] = 0
    return img.astype(np.float32)


def pca_align_volume(volume: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Align the object using principal axes when enough foreground is present."""
    coords = np.argwhere(volume > threshold)
    if coords.shape[0] < 10:
        return volume.astype(np.float32)
    center = coords.mean(axis=0)
    coords_centered = coords - center
    cov = np.cov(coords_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    rotation = eigvecs[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    offset = center - rotation @ center
    aligned = affine_transform(volume, rotation, offset=offset, order=1, mode="constant", cval=0.0)
    return aligned.astype(np.float32)


def crop_and_pad_volume(
    volume: np.ndarray,
    threshold: float = 0.1,
    margin: int = 5,
    padding: int = 10,
) -> np.ndarray:
    """Crop to the foreground support and add a safety padding."""
    mask = volume > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return volume.astype(np.float32)

    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)

    z_min = max(0, int(min_c[0]) - margin)
    z_max = min(volume.shape[0], int(max_c[0]) + 1 + margin)
    y_min = max(0, int(min_c[1]) - margin)
    y_max = min(volume.shape[1], int(max_c[1]) + 1 + margin)
    x_min = max(0, int(min_c[2]) - margin)
    x_max = min(volume.shape[2], int(max_c[2]) + 1 + margin)

    cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    if padding > 0:
        cropped = np.pad(cropped, pad_width=padding, mode="constant", constant_values=0)
    cropped = cropped.astype(np.float32)
    cropped[cropped < threshold] = 0
    return cropped


def resample_isometric(img: np.ndarray, target_dim: int = 256) -> tuple[np.ndarray, dict[str, object]]:
    """Scale isometrically and embed the result into a cubic target volume."""
    target_shape = (target_dim, target_dim, target_dim)
    scale_factor = min(t / s for t, s in zip(target_shape, img.shape))
    img_rescaled = zoom(img, scale_factor, order=1)

    new_img = np.zeros(target_shape, dtype=img.dtype)
    offsets = [(t - s) // 2 for t, s in zip(target_shape, img_rescaled.shape)]
    new_img[
        offsets[0] : offsets[0] + img_rescaled.shape[0],
        offsets[1] : offsets[1] + img_rescaled.shape[1],
        offsets[2] : offsets[2] + img_rescaled.shape[2],
    ] = img_rescaled

    params = {
        "scale_factor": float(scale_factor),
        "offsets": [int(v) for v in offsets],
        "orig_crop_shape": [int(v) for v in img.shape],
    }
    return new_img.astype(np.float32), params


def process_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    target_dim: int = DEFAULT_TARGET_DIM,
    threshold: float = DEFAULT_THRESHOLD,
    margin: int = 5,
    padding: int = 10,
    pca_pad: int = 50,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {input_path}")

    progress_bar = tqdm(files, desc="Preprocessing liver CT data", unit="img")
    for file_path in progress_bar:
        progress_bar.set_postfix(file=file_path.name[:20])
        try:
            nii = nib.load(str(file_path))
            nii = nib.as_closest_canonical(nii)
            img = np.asanyarray(nii.dataobj).astype(np.float32)

            img = normalize_ct_volume(img, mask=(img > 0))
            if pca_pad > 0:
                img = np.pad(img, pad_width=pca_pad, mode="constant", constant_values=0)
            img = pca_align_volume(img, threshold=threshold)
            img = crop_and_pad_volume(img, threshold=threshold, margin=margin, padding=padding)
            final_img, params = resample_isometric(img, target_dim=target_dim)

            out_nii = nib.Nifti1Image(final_img, np.eye(4))
            nib.save(out_nii, str(output_path / file_path.name))
            np.save(output_path / file_path.name.replace(".nii.gz", "_params.npy"), params)

            del img, final_img, nii, out_nii
            gc.collect()
        except Exception as exc:
            tqdm.write(f"Error while processing {file_path.name}: {exc}")

    print(f"Finished preprocessing {len(files)} volumes into {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess liver CT volumes for sparse-to-dense reconstruction.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input .nii.gz files.")
    parser.add_argument("--output-dir", required=True, help="Directory for preprocessed output volumes.")
    parser.add_argument("--target-dim", type=int, default=DEFAULT_TARGET_DIM, help="Target cube size, e.g. 64 or 256.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Foreground threshold used for cropping and alignment.")
    parser.add_argument("--margin", type=int, default=5, help="Foreground crop margin in voxels.")
    parser.add_argument("--padding", type=int, default=10, help="Extra zero padding after cropping.")
    parser.add_argument("--pca-pad", type=int, default=50, help="Padding added before PCA alignment.")
    args = parser.parse_args()

    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_dim=args.target_dim,
        threshold=args.threshold,
        margin=args.margin,
        padding=args.padding,
        pca_pad=args.pca_pad,
    )


if __name__ == "__main__":
    main()
