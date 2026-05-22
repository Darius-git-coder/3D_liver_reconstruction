# scripts/train_inpainting.py
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from inpainting3d.data import (
    center_of_mass_threshold,
    load_nifti_volume,
    normalize_volume,
    random_normals,
    resample_to_shape,
    simulate_sparse_acquisition,
)
from inpainting3d.losses import CombinedInpaintingLoss, CombinedLossConfig
from inpainting3d.metrics import compute_metrics_per_case
from inpainting3d.models import DeepResUNet3D, GatedResUNet3D, PartialResUNet3D
from inpainting3d.repro import build_repro_metadata, get_git_info
from inpainting3d.splits import (
    case_id_from_path,
    create_split_manifest,
    filter_to_known_files,
    get_split_files,
    load_split_manifest,
    normalize_case_path,
    resolve_case_paths,
    write_split_manifest,
)
from inpainting3d.utils import (
    AmpManager,
    ensure_dir,
    finalize_inpainting_prediction,
    set_reproducibility,
    strip_dataparallel_prefix,
)


class LiverInpaintingDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        dim: int = 64,
        slices_min: int = 4,
        slices_max: int = 32,
        slice_sampling: str = "uniform",
        slice_mean: float | None = None,
        slice_std: float | None = None,
        thickness_vox: float = 1.0,
        canonical: bool = True,
        normalize: str = "clip01",
        is_train: bool = True,
        seed: int = 1337,
        curriculum: bool = False,
        curriculum_stage_epochs: int = 50,
        sparse_noise_std: float = 0.0,
        intensity_aug_prob: float = 0.0,
        intensity_scale_min: float = 0.85,
        intensity_scale_max: float = 1.15,
        intensity_bias_min: float = -0.05,
        intensity_bias_max: float = 0.05,
        return_meta: bool = False,
    ):
        self.files = list(files)
        self.dim = dim
        self.slices_min = slices_min
        self.slices_max = slices_max
        self.slice_sampling = str(slice_sampling).strip().lower()
        self.slice_mean = None if slice_mean is None else float(slice_mean)
        self.slice_std = None if slice_std is None else float(slice_std)
        self.thickness_vox = thickness_vox
        self.canonical = canonical
        self.normalize = normalize
        self.is_train = is_train
        self.seed = seed
        self.curriculum = curriculum
        self.curriculum_stage_epochs = max(1, int(curriculum_stage_epochs))
        self.sparse_noise_std = float(sparse_noise_std)
        self.intensity_aug_prob = float(intensity_aug_prob)
        self.intensity_scale_min = float(intensity_scale_min)
        self.intensity_scale_max = float(intensity_scale_max)
        self.intensity_bias_min = float(intensity_bias_min)
        self.intensity_bias_max = float(intensity_bias_max)
        self.return_meta = bool(return_meta)
        self.current_epoch = 0

    def __len__(self) -> int:
        return len(self.files)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _slice_range(self) -> Tuple[int, int]:
        if not (self.is_train and self.curriculum):
            return self.slices_min, self.slices_max
        if self.current_epoch < self.curriculum_stage_epochs:
            return 64, 128
        if self.current_epoch < 2 * self.curriculum_stage_epochs:
            return 16, 64
        return self.slices_min, self.slices_max

    def _sample_num_slices(self, rng: np.random.Generator) -> int:
        slices_min, slices_max = self._slice_range()
        if self.slice_sampling == "normal":
            mean = self.slice_mean if self.slice_mean is not None else 0.5 * (slices_min + slices_max)
            std = self.slice_std if self.slice_std is not None else max(1.0, 0.25 * (slices_max - slices_min))
            n_slices = int(np.rint(rng.normal(loc=mean, scale=max(std, 1e-6))))
            return int(np.clip(n_slices, slices_min, slices_max))
        return int(rng.integers(slices_min, slices_max + 1))

    def __getitem__(self, idx: int):
        path = self.files[idx]
        vol, _aff = load_nifti_volume(path, canonical=self.canonical, dtype=np.float32)
        vol = normalize_volume(vol, self.normalize)
        vol = resample_to_shape(vol, (self.dim, self.dim, self.dim), order=1)

        if not self.is_train:
            rng = np.random.default_rng(self.seed + idx)
        else:
            rng = np.random.default_rng()

        if self.is_train and self.intensity_aug_prob > 0.0 and rng.random() < self.intensity_aug_prob:
            scale = rng.uniform(self.intensity_scale_min, self.intensity_scale_max)
            bias = rng.uniform(self.intensity_bias_min, self.intensity_bias_max)
            vol = np.clip(scale * vol + bias, 0.0, 1.0)

        center = center_of_mass_threshold(vol, thr=0.1)
        n_slices = self._sample_num_slices(rng)
        normals = random_normals(n_slices, rng)

        sparse, mask, _hits = simulate_sparse_acquisition(
            vol,
            normals=normals,
            center=center,
            thickness_vox=self.thickness_vox,
            chunk=64,
        )
        if self.is_train and self.sparse_noise_std > 0:
            sparse = np.clip(
                sparse + rng.normal(0.0, self.sparse_noise_std, size=sparse.shape).astype(np.float32),
                0.0,
                1.0,
            )

        x = np.stack([sparse, mask], axis=0).astype(np.float32)
        y = vol[None, ...].astype(np.float32)
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        if self.return_meta:
            meta = {
                "path": normalize_case_path(path),
                "case_id": case_id_from_path(path),
                "index": int(idx),
                "num_slices": int(n_slices),
                "seed": None if self.is_train else int(self.seed + idx),
            }
            return x_t, y_t, meta
        return x_t, y_t


def save_sample_figure(x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, out_path: str) -> None:
    x0 = x[0, 0].detach().cpu().numpy()
    xm = x[0, 1].detach().cpu().numpy()
    yt = y[0, 0].detach().cpu().numpy()
    yp = pred[0, 0].detach().cpu().numpy()

    mid = x0.shape[0] // 2
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(yt[mid].T, cmap="bone", origin="lower")
    ax[0, 0].set_title("GT")
    ax[0, 1].imshow(x0[mid].T, cmap="bone", origin="lower")
    ax[0, 1].set_title("Sparse Input")
    ax[1, 0].imshow(xm[mid].T, cmap="gray", origin="lower")
    ax[1, 0].set_title("Mask (known=1)")
    ax[1, 1].imshow(yp[mid].T, cmap="bone", origin="lower")
    ax[1, 1].set_title("Prediction")
    for axis in ax.ravel():
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def clamp_prediction(
    pred: torch.Tensor,
    sparse: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    hard_constraint: bool = True,
) -> torch.Tensor:
    pred = torch.clamp(pred, 0.0, 1.0)
    if hard_constraint and sparse is not None and mask is not None:
        return finalize_inpainting_prediction(pred, sparse=sparse, known_mask=mask)
    return pred


def build_model(kind: str, init_feat: int = 32) -> torch.nn.Module:
    kind = kind.lower()
    if kind == "baseline":
        return DeepResUNet3D(in_channels=2, init_feat=init_feat)
    if kind == "gated":
        return GatedResUNet3D(in_channels=2, init_feat=init_feat)
    if kind == "partial":
        return PartialResUNet3D(init_feat=init_feat)
    raise ValueError(f"Unbekanntes model kind: {kind}")


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def maybe_limit_cases(files: List[str], max_cases: int | None, split_name: str) -> List[str]:
    if max_cases is None:
        return files
    limit = int(max_cases)
    if limit <= 0:
        raise ValueError(f"{split_name}: max_cases must be >= 1 when provided.")
    return files[:limit]


def prepare_data_splits(args: argparse.Namespace) -> Tuple[Dict[str, object], List[str], List[str], str | None]:
    files = resolve_case_paths(args.data)
    if not files:
        raise RuntimeError("Keine Dateien gefunden.")

    if args.split_file:
        manifest = load_split_manifest(args.split_file)
        split_source = normalize_case_path(args.split_file)
    else:
        manifest = create_split_manifest(
            files,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            source_glob=args.data,
            project_root=ROOT,
            git_info=get_git_info(ROOT),
        )
        split_source = None

    train_files = filter_to_known_files(get_split_files(manifest, args.train_split), files)
    val_files = filter_to_known_files(get_split_files(manifest, args.val_split), files)
    train_files = maybe_limit_cases(train_files, args.max_train_cases, "train")
    val_files = maybe_limit_cases(val_files, args.max_val_cases, "val")
    if not train_files:
        raise RuntimeError(f"Split '{args.train_split}' enthaelt keine Trainingsfaelle.")
    if not val_files:
        raise RuntimeError(f"Split '{args.val_split}' enthaelt keine Validierungsfaelle.")

    manifest["selected_splits"] = {"train": args.train_split, "val": args.val_split}
    write_split_manifest(manifest, os.path.join(args.out, "split_manifest.json"))
    return manifest, train_files, val_files, split_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Glob-Pattern fuer *.nii.gz")
    ap.add_argument("--out", type=str, default="./runs/inpainting3d")
    ap.add_argument("--model", type=str, default="partial", choices=["baseline", "gated", "partial"])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=5e-6)
    ap.add_argument("--init_feat", type=int, default=32)
    ap.add_argument("--slices_min", type=int, default=4)
    ap.add_argument("--slices_max", type=int, default=32)
    ap.add_argument("--slice_sampling", type=str, default="uniform", choices=["uniform", "normal"])
    ap.add_argument("--slice_mean", type=float, default=None)
    ap.add_argument("--slice_std", type=float, default=None)
    ap.add_argument("--hole_weight", type=float, default=30.0)
    ap.add_argument("--valid_weight", type=float, default=1.0)
    ap.add_argument("--hole_fg_weight", type=float, default=None)
    ap.add_argument("--hole_bg_weight", type=float, default=None)
    ap.add_argument("--foreground_threshold", type=float, default=0.1)
    ap.add_argument("--w_grad", type=float, default=0.05)
    ap.add_argument("--w_ssim", type=float, default=0.0)
    ap.add_argument("--w_ms_ssim", type=float, default=0.2)
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"])
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--curriculum_stage_epochs", type=int, default=50)
    ap.add_argument("--sparse_noise_std", type=float, default=0.0)
    ap.add_argument("--intensity_aug_prob", type=float, default=0.0)
    ap.add_argument("--intensity_scale_min", type=float, default=0.85)
    ap.add_argument("--intensity_scale_max", type=float, default=1.15)
    ap.add_argument("--intensity_bias_min", type=float, default=-0.05)
    ap.add_argument("--intensity_bias_max", type=float, default=0.05)
    ap.add_argument("--thickness_vox", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--split_file", type=str, default="", help="Persistierte Split-JSON mit train/val/test-Falllisten.")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--max_train_cases", type=int, default=None)
    ap.add_argument("--max_val_cases", type=int, default=None)
    ap.add_argument("--no_hard_constraint", dest="hard_constraint", action="store_false")
    ap.set_defaults(hard_constraint=True)
    args = ap.parse_args()

    ensure_dir(args.out)
    ensure_dir(os.path.join(args.out, "samples"))
    ensure_dir(os.path.join(args.out, "plots"))
    ensure_dir(os.path.join(args.out, "checkpoints"))

    set_reproducibility(args.seed, deterministic=args.deterministic)
    split_manifest, train_files, val_files, split_source = prepare_data_splits(args)

    train_ds = LiverInpaintingDataset(
        train_files,
        dim=args.dim,
        slices_min=args.slices_min,
        slices_max=args.slices_max,
        slice_sampling=args.slice_sampling,
        slice_mean=args.slice_mean,
        slice_std=args.slice_std,
        thickness_vox=args.thickness_vox,
        is_train=True,
        seed=args.seed,
        curriculum=args.curriculum,
        curriculum_stage_epochs=args.curriculum_stage_epochs,
        sparse_noise_std=args.sparse_noise_std,
        intensity_aug_prob=args.intensity_aug_prob,
        intensity_scale_min=args.intensity_scale_min,
        intensity_scale_max=args.intensity_scale_max,
        intensity_bias_min=args.intensity_bias_min,
        intensity_bias_max=args.intensity_bias_max,
        return_meta=False,
    )
    val_ds = LiverInpaintingDataset(
        val_files,
        dim=args.dim,
        slices_min=args.slices_min,
        slices_max=args.slices_max,
        slice_sampling=args.slice_sampling,
        slice_mean=args.slice_mean,
        slice_std=args.slice_std,
        thickness_vox=args.thickness_vox,
        is_train=False,
        seed=args.seed,
        return_meta=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    default_train_workers = 0 if sys.platform.startswith("win") else 4
    default_val_workers = 0 if sys.platform.startswith("win") else 2

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=default_train_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=default_val_workers,
        pin_memory=pin_memory,
    )

    model = build_model(args.model, init_feat=args.init_feat).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    loss_cfg = CombinedLossConfig(
        hole_weight=args.hole_weight,
        valid_weight=args.valid_weight,
        hole_fg_weight=args.hole_fg_weight,
        hole_bg_weight=args.hole_bg_weight,
        foreground_threshold=args.foreground_threshold,
        w_grad=args.w_grad,
        w_ssim=args.w_ssim,
        w_ms_ssim=args.w_ms_ssim,
        data_range=1.0,
    )
    criterion = CombinedInpaintingLoss(loss_cfg)
    amp = AmpManager(device=device, enabled=args.amp)

    start_epoch = 0
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": [], "val_rmse_missing": [], "val_ssim_all": []}

    if args.resume:
        history_path = os.path.join(args.out, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as handle:
                loaded_history = json.load(handle)
            if isinstance(loaded_history, dict):
                for key in history:
                    value = loaded_history.get(key)
                    if isinstance(value, list):
                        history[key] = value

        ckpt = torch.load(args.resume, map_location=device)
        state = strip_dataparallel_prefix(ckpt["model"])
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        if amp.scaler is not None and ckpt.get("scaler") is not None:
            amp.scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", best_val))

    save_json(
        os.path.join(args.out, "config.json"),
        {
            **vars(args),
            "loss_cfg": asdict(loss_cfg),
            "selected_splits": {"train": args.train_split, "val": args.val_split},
            "num_train_cases": len(train_files),
            "num_val_cases": len(val_files),
            "max_train_cases": args.max_train_cases,
            "max_val_cases": args.max_val_cases,
        },
    )
    save_json(
        os.path.join(args.out, "reproducibility.json"),
        build_repro_metadata(
            vars(args),
            split_manifest_path=os.path.join(args.out, "split_manifest.json"),
            extra={
                "num_train_cases": len(train_files),
                "num_val_cases": len(val_files),
                "selected_splits": {"train": args.train_split, "val": args.val_split},
                "split_source": split_source,
                "split_counts": split_manifest.get("counts", {}),
            },
            cwd=ROOT,
        ),
    )

    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        tr_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for x, y in loop:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast():
                mask = x[:, 1:2, ...]
                sparse = x[:, 0:1, ...]
                pred = clamp_prediction(model(x), sparse=sparse, mask=mask, hard_constraint=args.hard_constraint)
                loss = criterion(pred, y, mask)

            amp.backward_and_step(loss, optimizer)
            tr_loss += float(loss.item())
            loop.set_postfix(loss=float(loss.item()))

        tr_loss /= max(1, len(train_loader))

        model.eval()
        va_loss = 0.0
        va_metrics_rows: List[Dict[str, float | str]] = []
        with torch.no_grad():
            for batch_index, (x, y, meta) in enumerate(val_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with amp.autocast():
                    mask = x[:, 1:2, ...]
                    sparse = x[:, 0:1, ...]
                    pred = clamp_prediction(model(x), sparse=sparse, mask=mask, hard_constraint=args.hard_constraint)
                    loss = criterion(pred, y, mask)
                va_loss += float(loss.item())

                case_ids = [str(item) for item in meta["case_id"]]
                paths = [str(item) for item in meta["path"]]
                va_metrics_rows.extend(
                    compute_metrics_per_case(pred, y, known_mask=mask, data_range=1.0, case_ids=case_ids, paths=paths)
                )

                if batch_index == 0 and (epoch + 1) % 5 == 0:
                    save_sample_figure(x, y, pred, os.path.join(args.out, "samples", f"epoch_{epoch + 1:04d}.png"))

        va_loss /= max(1, len(val_loader))
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(va_loss)

        def mean_key(key: str) -> float:
            return float(np.mean([float(row[key]) for row in va_metrics_rows]))

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["val_rmse_missing"].append(mean_key("rmse_missing"))
        history["val_ssim_all"].append(mean_key("ssim_all"))

        print(
            f"Epoch {epoch + 1}: train={tr_loss:.6f} val={va_loss:.6f} "
            f"rmse_missing={history['val_rmse_missing'][-1]:.5f} "
            f"ssim_all={history['val_ssim_all'][-1]:.4f} lr={history['lr'][-1]:.2e}",
            flush=True,
        )

        ckpt_path = os.path.join(args.out, "checkpoints", "last.pt")
        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save(
            {
                "epoch": epoch,
                "model": state_dict,
                "optim": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "scaler": amp.scaler.state_dict() if amp.scaler is not None else None,
                "best_val": best_val,
            },
            ckpt_path,
        )

        if va_loss < best_val:
            best_val = va_loss
            torch.save(state_dict, os.path.join(args.out, "best_model.pt"))

        save_json(os.path.join(args.out, "history.json"), history)

        if (epoch + 1) % 5 == 0:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(history["train_loss"], label="train")
            plt.plot(history["val_loss"], label="val")
            plt.yscale("log")
            plt.grid(True, which="both", alpha=0.4)
            plt.legend()
            plt.title("Loss History (log)")
            plt.savefig(os.path.join(args.out, "plots", "loss_history.png"), dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
