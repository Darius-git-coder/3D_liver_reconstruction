# -*- coding: utf-8 -*-
"""Batch preprocessing pipeline for liver-focused CT volumes.

The script standardizes masked liver CT volumes to an isometric cube and was
used to generate the archived series

- ``pipeline_preprocessed_data_64``
- ``pipeline_preprocessed_data_128``
- ``pipeline_preprocessed_data_256``

from the same liver-focused NIfTI source directory. The pipeline is identical
across runs; only the target cube size changes.

Methodological background:
- NIfTI affine handling and canonical orientation via NIfTI / NiBabel
- CT HU windowing for soft-tissue focused normalization
- optional PCA-based pose normalization
- affine / zoom-based isometric resampling
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, zoom
from tqdm import tqdm


DEFAULT_THRESHOLD = 0.05
DEFAULT_MIN_HU = -100.0
DEFAULT_MAX_HU = 400.0
DEFAULT_MARGIN = 5
DEFAULT_PADDING = 10
DEFAULT_PRE_PAD = 50
DEFAULT_TARGET_DIM = 128
DEFAULT_OUTPUT_ROOT = Path("./data")


def load_nifti_volume(file_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a NIfTI file and convert it to canonical RAS+ orientation."""
    nii = nib.load(str(file_path))
    nii = nib.as_closest_canonical(nii)
    volume = np.asarray(nii.dataobj, dtype=np.float32)
    spacing = np.asarray(nii.header.get_zooms()[:3], dtype=np.float32)
    return volume, spacing, nii.affine


def normalize_ct_volume(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
    min_hu: float = DEFAULT_MIN_HU,
    max_hu: float = DEFAULT_MAX_HU,
) -> np.ndarray:
    """Apply HU clipping and linear scaling to ``[0, 1]``."""
    img = np.clip(volume, min_hu, max_hu)
    img = (img - min_hu) / (max_hu - min_hu)
    if mask is not None:
        img[mask == 0] = 0
    return img.astype(np.float32, copy=False)


def pca_align_volume(volume: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Align the foreground with its principal axes."""
    coords = np.argwhere(volume > threshold)
    if len(coords) < 10:
        return volume

    center = coords.mean(axis=0)
    coords_centered = coords - center
    cov = np.cov(coords_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    rotation = eigvecs[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1

    offset = center - rotation @ center
    return affine_transform(volume, rotation, offset=offset, order=1, mode="constant", cval=0.0)


def crop_and_pad_volume(
    volume: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    margin: int = DEFAULT_MARGIN,
    padding: int = DEFAULT_PADDING,
) -> np.ndarray:
    """Crop to the foreground support and add a safety border."""
    mask = volume > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return volume

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
    cropped[cropped < threshold] = 0
    return cropped


def resample_isometric(img: np.ndarray, target_dim: int) -> tuple[np.ndarray, dict[str, object]]:
    """Scale isotropically and embed the result in a centered target cube."""
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
        "target_dim": int(target_dim),
    }
    return new_img, params


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Resolve the output directory from CLI arguments."""
    if args.output_dir:
        return Path(args.output_dir)

    root = Path(args.output_root)
    return root / f"pipeline_preprocessed_data_{int(args.target_dim)}"


def process_dataset(args: argparse.Namespace) -> None:
    """Run the full preprocessing pipeline on a directory of NIfTI files."""
    input_path = Path(args.source)
    output_path = resolve_output_dir(args)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern!r} found in {input_path}.")

    manifest = {
        "source_dir": str(input_path),
        "output_dir": str(output_path),
        "pattern": args.pattern,
        "target_dim": int(args.target_dim),
        "threshold": float(args.threshold),
        "margin": int(args.margin),
        "padding": int(args.padding),
        "pre_pad": int(args.pre_pad),
        "min_hu": float(args.min_hu),
        "max_hu": float(args.max_hu),
        "num_files": int(len(files)),
        "canonical_orientation": "RAS+ via nib.as_closest_canonical",
    }
    (output_path / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    progress_bar = tqdm(files, desc="Preprocessing liver CT data", unit="img")
    for file_path in progress_bar:
        progress_bar.set_postfix(file=file_path.name[:20])

        try:
            volume, spacing, affine = load_nifti_volume(file_path)

            img = normalize_ct_volume(
                volume,
                mask=(volume > 0),
                min_hu=args.min_hu,
                max_hu=args.max_hu,
            )

            img = np.pad(img, pad_width=int(args.pre_pad), mode="constant", constant_values=0)
            img = pca_align_volume(img, threshold=args.threshold)
            img = crop_and_pad_volume(
                img,
                threshold=args.threshold,
                margin=int(args.margin),
                padding=int(args.padding),
            )
            final_img, params = resample_isometric(img, target_dim=int(args.target_dim))

            params.update(
                {
                    "canonical_spacing": [float(v) for v in spacing.tolist()],
                    "source_affine": np.asarray(affine, dtype=np.float64).tolist(),
                }
            )

            output_nii = nib.Nifti1Image(final_img, np.eye(4))
            nib.save(output_nii, str(output_path / file_path.name))
            np.save(output_path / file_path.name.replace(".nii.gz", "_params.npy"), params)

            del volume, img, final_img, output_nii
            gc.collect()

        except Exception as exc:  # pragma: no cover - diagnostic path
            tqdm.write(f"Error while processing {file_path.name}: {exc}")

    print(f"\nFinished preprocessing {len(files)} volumes into {output_path}.")


def build_argparser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=str, required=True, help="Directory containing liver-focused *.nii.gz volumes.")
    ap.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Exact output directory. If omitted, --output_root/pipeline_preprocessed_data_<dim> is used.",
    )
    ap.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory used when --output_dir is omitted.",
    )
    ap.add_argument("--pattern", type=str, default="*.nii.gz", help="Glob pattern for source files.")
    ap.add_argument("--target_dim", type=int, default=DEFAULT_TARGET_DIM, help="Target cube edge length.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Foreground threshold for PCA and cropping.")
    ap.add_argument("--min_hu", type=float, default=DEFAULT_MIN_HU, help="Lower HU clipping bound.")
    ap.add_argument("--max_hu", type=float, default=DEFAULT_MAX_HU, help="Upper HU clipping bound.")
    ap.add_argument("--margin", type=int, default=DEFAULT_MARGIN, help="Foreground crop margin in voxels.")
    ap.add_argument("--padding", type=int, default=DEFAULT_PADDING, help="Additional zero padding after cropping.")
    ap.add_argument("--pre_pad", type=int, default=DEFAULT_PRE_PAD, help="Zero padding inserted before PCA alignment.")
    return ap


if __name__ == "__main__":
    process_dataset(build_argparser().parse_args())
