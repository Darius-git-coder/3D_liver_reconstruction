# inpainting3d/utils.py
# Python 3.9+, PyTorch 1.12+ kompatibel

from __future__ import annotations
import os
import random
import warnings
import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def set_reproducibility(seed: int = 1337, deterministic: bool = False) -> None: # bool = True -> Probleme
    """
    Setzt Seeds und – optional – deterministische PyTorch-Optionen.
    Achtung: Vollständige Deterministik auf GPU kann je nach Ops nicht immer möglich sein.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN Autotuner deaktivieren, deterministische Algos bevorzugen
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(False) # True, verursacht Probleme
            except Exception as e:
                warnings.warn(f"Deterministische Algos konnten nicht erzwungen werden: {e}")
    else:
        torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def torch_load_weights_compat(path: str, map_location: torch.device) -> Any:
    """
    Lädt Checkpoints robust über PyTorch-Versionen hinweg:
    - Wenn torch.load(weights_only=...) verfügbar ist: nutze weights_only=True.
    - Sonst: Fallback auf normales torch.load.
    """
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        return torch.load(path, map_location=map_location, weights_only=True)
    # PyTorch <= 2.0 / <= 1.12: kein weights_only
    return torch.load(path, map_location=map_location)


def strip_dataparallel_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Entfernt 'module.' Prefix, wenn das Modell mit DataParallel gespeichert wurde."""
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def finalize_inpainting_prediction(
    pred: torch.Tensor,
    sparse: torch.Tensor,
    known_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Erzwingt den Inpainting-Hard-Constraint:
    bekannte Voxels bleiben exakt aus dem Input erhalten,
    nur missing Voxels werden vom Modell gefüllt.
    """
    pred = torch.clamp(pred, 0.0, 1.0)
    sparse = sparse.float()
    known_mask = known_mask.float()
    return known_mask * sparse + (1.0 - known_mask) * pred


@dataclass
class AmpManager:
    """
    AMP-Wrapper für PyTorch 1.12+:
    - CUDA: nutzt torch.cuda.amp.autocast + GradScaler
    - CPU: nullcontext + scaler=None
    """
    device: torch.device
    enabled: bool = True
    scaler: Optional[torch.cuda.amp.GradScaler] = None

    def __post_init__(self) -> None:
        if self.enabled and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def autocast(self):
        if self.enabled and self.device.type == "cuda":
            return torch.cuda.amp.autocast()
        return nullcontext()

    def backward_and_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is None:
            loss.backward()
            optimizer.step()
            return
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
