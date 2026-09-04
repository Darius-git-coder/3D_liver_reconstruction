from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from inpainting3d.generative import (
    DiffusionSchedule,
    build_generative_model,
    diffusion_training_loss,
    flow_matching_training_loss,
    make_diffusion_schedule,
    sample_diffusion_ddim,
    sample_flow_matching,
)
from inpainting3d.repro import build_repro_metadata
from inpainting3d.utils import AmpManager, ensure_dir, set_reproducibility, strip_dataparallel_prefix, torch_load_weights_compat
from scripts.train_inpainting import LiverInpaintingDataset, prepare_data_splits, save_sample_figure


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def dataset_kwargs(args: argparse.Namespace, is_train: bool) -> Dict[str, Any]:
    return {
        "dim": args.dim,
        "slices_min": args.slices_min,
        "slices_max": args.slices_max,
        "slice_sampling": args.slice_sampling,
        "slice_mean": args.slice_mean,
        "slice_std": args.slice_std,
        "slice_geometry": args.slice_geometry,
        "slice_axis": args.slice_axis,
        "slice_axis_jitter_deg": args.slice_axis_jitter_deg,
        "slice_fan_half_angle_deg": args.slice_fan_half_angle_deg,
        "slice_elevation_jitter_deg": args.slice_elevation_jitter_deg,
        "slice_sweep_jitter_deg": args.slice_sweep_jitter_deg,
        "probe_pos_sigma_vox": args.probe_pos_sigma_vox,
        "probe_depth_sigma_vox": args.probe_depth_sigma_vox,
        "probe_tilt_sigma": args.probe_tilt_sigma,
        "thickness_vox": args.thickness_vox,
        "is_train": is_train,
        "seed": args.seed,
        "curriculum": bool(is_train and args.curriculum),
        "curriculum_stage_epochs": args.curriculum_stage_epochs,
        "sparse_noise_std": args.sparse_noise_std if is_train else 0.0,
        "intensity_aug_prob": args.intensity_aug_prob if is_train else 0.0,
        "intensity_scale_min": args.intensity_scale_min,
        "intensity_scale_max": args.intensity_scale_max,
        "intensity_bias_min": args.intensity_bias_min,
        "intensity_bias_max": args.intensity_bias_max,
    }


def compute_loss(
    args: argparse.Namespace,
    model: torch.nn.Module,
    y: torch.Tensor,
    sparse: torch.Tensor,
    mask: torch.Tensor,
    diffusion_schedule: DiffusionSchedule | None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if args.objective == "diffusion":
        if diffusion_schedule is None:
            raise RuntimeError("Diffusion schedule is required for diffusion training.")
        return diffusion_training_loss(
            model,
            y,
            sparse,
            mask,
            diffusion_schedule,
            hole_weight=args.hole_weight,
            valid_weight=args.valid_weight,
        )
    if args.objective == "flow_matching":
        return flow_matching_training_loss(
            model,
            y,
            sparse,
            mask,
            hole_weight=args.hole_weight,
            valid_weight=args.valid_weight,
        )
    raise ValueError(f"Unknown objective: {args.objective}")


def sample_prediction(
    args: argparse.Namespace,
    model: torch.nn.Module,
    sparse: torch.Tensor,
    mask: torch.Tensor,
    diffusion_schedule: DiffusionSchedule | None,
) -> torch.Tensor:
    if args.objective == "diffusion":
        if diffusion_schedule is None:
            raise RuntimeError("Diffusion schedule is required for diffusion sampling.")
        return sample_diffusion_ddim(
            model,
            sparse=sparse,
            known_mask=mask,
            schedule=diffusion_schedule,
            sample_steps=args.sample_steps,
            hard_constraint=args.hard_constraint,
        )
    return sample_flow_matching(
        model,
        sparse=sparse,
        known_mask=mask,
        sample_steps=args.sample_steps,
        hard_constraint=args.hard_constraint,
    )


def load_initial_state(model: torch.nn.Module, weights_path: str, device: torch.device) -> None:
    state = torch_load_weights_compat(weights_path, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and any(isinstance(value, torch.Tensor) for value in state.values()):
        model.load_state_dict(strip_dataparallel_prefix(state), strict=True)
        return
    raise RuntimeError(f"Unexpected checkpoint format: {weights_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train conditional 3D diffusion or flow-matching inpainting models.")
    ap.add_argument("--data", type=str, required=True, help="Glob pattern for dense *.nii.gz volumes.")
    ap.add_argument("--out", type=str, default="./runs/generative_inpainting")
    ap.add_argument("--objective", type=str, default="diffusion", choices=["diffusion", "flow_matching"])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--init_feat", type=int, default=16)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--time_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--diffusion_steps", type=int, default=1000)
    ap.add_argument("--diffusion_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--sample_every", type=int, default=5)
    ap.add_argument("--slices_min", type=int, default=4)
    ap.add_argument("--slices_max", type=int, default=32)
    ap.add_argument("--slice_sampling", type=str, default="uniform", choices=["uniform", "normal"])
    ap.add_argument("--slice_mean", type=float, default=None)
    ap.add_argument("--slice_std", type=float, default=None)
    ap.add_argument("--slice_geometry", type=str, default="random", choices=["random", "fibonacci", "ultrasound_fan", "ultrasound_probe"])
    ap.add_argument("--slice_axis", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--slice_axis_jitter_deg", type=float, default=25.0)
    ap.add_argument("--slice_fan_half_angle_deg", type=float, default=35.0)
    ap.add_argument("--slice_elevation_jitter_deg", type=float, default=6.0)
    ap.add_argument("--slice_sweep_jitter_deg", type=float, default=2.5)
    ap.add_argument("--probe_pos_sigma_vox", type=float, default=1.0)
    ap.add_argument("--probe_depth_sigma_vox", type=float, default=6.0)
    ap.add_argument("--probe_tilt_sigma", type=float, default=0.08)
    ap.add_argument("--thickness_vox", type=float, default=1.0)
    ap.add_argument("--hole_weight", type=float, default=4.0)
    ap.add_argument("--valid_weight", type=float, default=1.0)
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"])
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--curriculum_stage_epochs", type=int, default=50)
    ap.add_argument("--sparse_noise_std", type=float, default=0.0)
    ap.add_argument("--intensity_aug_prob", type=float, default=0.0)
    ap.add_argument("--intensity_scale_min", type=float, default=0.85)
    ap.add_argument("--intensity_scale_max", type=float, default=1.15)
    ap.add_argument("--intensity_bias_min", type=float, default=-0.05)
    ap.add_argument("--intensity_bias_max", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--init_weights", type=str, default="")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--split_file", type=str, default="", help="Persisted split JSON with train/val/test case lists.")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--max_train_cases", type=int, default=None)
    ap.add_argument("--max_val_cases", type=int, default=None)
    ap.add_argument("--no_hard_constraint", dest="hard_constraint", action="store_false")
    ap.set_defaults(hard_constraint=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.init_weights:
        raise ValueError("--resume and --init_weights cannot be used together.")

    ensure_dir(args.out)
    ensure_dir(os.path.join(args.out, "checkpoints"))
    ensure_dir(os.path.join(args.out, "samples"))
    ensure_dir(os.path.join(args.out, "plots"))

    set_reproducibility(args.seed, deterministic=args.deterministic)
    split_manifest, train_files, val_files, split_source = prepare_data_splits(args)
    _ = split_manifest

    train_ds = LiverInpaintingDataset(train_files, **dataset_kwargs(args, is_train=True))
    val_ds = LiverInpaintingDataset(val_files, **dataset_kwargs(args, is_train=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_workers = 0 if sys.platform.startswith("win") else 4
    val_workers = 0 if sys.platform.startswith("win") else 2
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=train_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=val_workers, pin_memory=pin_memory)

    model = build_generative_model(
        init_feat=args.init_feat,
        levels=args.levels,
        time_dim=args.time_dim,
        dropout=args.dropout,
    ).to(device)
    if args.init_weights:
        load_initial_state(model, args.init_weights, device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    amp = AmpManager(device=device, enabled=args.amp)
    diffusion_schedule = None
    if args.objective == "diffusion":
        diffusion_schedule = make_diffusion_schedule(args.diffusion_steps, device=device, schedule=args.diffusion_schedule)

    start_epoch = 0
    best_val = float("inf")
    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}

    if args.resume:
        checkpoint = torch_load_weights_compat(args.resume, map_location=device)
        state = strip_dataparallel_prefix(checkpoint["model"])
        target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        target_model.load_state_dict(state, strict=True)
        optimizer.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["sched"])
        if amp.scaler is not None and checkpoint.get("scaler") is not None:
            amp.scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", best_val))
        loaded_history = checkpoint.get("history", {})
        if isinstance(loaded_history, dict):
            for key in history:
                if isinstance(loaded_history.get(key), list):
                    history[key] = loaded_history[key]

    save_json(
        os.path.join(args.out, "config.json"),
        {
            **vars(args),
            "num_train_files": len(train_files),
            "num_val_files": len(val_files),
            "split_source": split_source,
            "model_parameters": sum(p.numel() for p in model.parameters()),
        },
    )
    save_json(os.path.join(args.out, "reproducibility.json"), build_repro_metadata(vars(args), cwd=ROOT))

    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"{args.objective} epoch {epoch + 1}/{args.epochs}")
        for x, y in loop:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sparse = x[:, 0:1, ...]
            mask = x[:, 1:2, ...]

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                loss, _aux = compute_loss(args, model, y, sparse, mask, diffusion_schedule)
            amp.backward_and_step(loss, optimizer)
            train_loss += float(loss.item())
            loop.set_postfix(loss=float(loss.item()))
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, (x, y) in enumerate(val_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                sparse = x[:, 0:1, ...]
                mask = x[:, 1:2, ...]
                with amp.autocast():
                    loss, _aux = compute_loss(args, model, y, sparse, mask, diffusion_schedule)
                val_loss += float(loss.item())

                should_sample = args.sample_every > 0 and (epoch + 1) % args.sample_every == 0 and index == 0
                if should_sample:
                    pred = sample_prediction(args, model, sparse, mask, diffusion_schedule)
                    save_sample_figure(x, y, pred, os.path.join(args.out, "samples", f"epoch_{epoch + 1:04d}.png"))
        val_loss /= max(1, len(val_loader))

        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        print(
            f"Epoch {epoch + 1}: train={train_loss:.6f} val={val_loss:.6f} "
            f"lr={history['lr'][-1]:.2e}",
            flush=True,
        )

        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        checkpoint = {
            "epoch": epoch,
            "model": state_dict,
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "scaler": amp.scaler.state_dict() if amp.scaler is not None else None,
            "best_val": best_val,
            "history": history,
            "objective": args.objective,
            "config": vars(args),
        }
        torch.save(checkpoint, os.path.join(args.out, "checkpoints", "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            checkpoint["best_val"] = best_val
            torch.save(checkpoint, os.path.join(args.out, "best_model.pt"))

        save_json(os.path.join(args.out, "history.json"), history)
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(history["train_loss"], label="train")
            plt.plot(history["val_loss"], label="val")
            plt.yscale("log")
            plt.grid(True, which="both", alpha=0.4)
            plt.legend()
            plt.title(f"{args.objective} loss history")
            plt.savefig(os.path.join(args.out, "plots", "loss_history.png"), dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
