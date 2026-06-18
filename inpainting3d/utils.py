# inpainting3d/utils.py
# Public repository snapshot targets Python 3.10+.

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
    Seed Python, NumPy, and PyTorch for reproducible experiments.
    
    Parameters
    ----------
    seed : int
        Random seed used for reproducible sampling. Defaults to 1337.
    deterministic : bool
        Whether deterministic PyTorch settings should be enabled. Defaults to False.
    
    Returns
    -------
    None
        This function is executed for its side effects.
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
    """
    Create a directory if it does not already exist.
    
    Parameters
    ----------
    path : str
        Filesystem path to the input artifact.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    os.makedirs(path, exist_ok=True)


def torch_load_weights_compat(path: str, map_location: torch.device) -> Any:
    """
    Load model weights in a way that is compatible across PyTorch versions.
    
    Parameters
    ----------
    path : str
        Filesystem path to the input artifact.
    map_location : torch.device
        Device mapping used when loading checkpoints.
    
    Returns
    -------
    Any
        Loaded checkpoint payload or state dictionary.
    """
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        return torch.load(path, map_location=map_location, weights_only=True)
    # PyTorch <= 2.0 / <= 1.12: kein weights_only
    return torch.load(path, map_location=map_location)


def strip_dataparallel_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove a leading DataParallel prefix from a state dictionary.
    
    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        State dictionary that may contain DataParallel prefixes.
    
    Returns
    -------
    Dict[str, torch.Tensor]
        State dictionary with normalized parameter names.
    """
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def finalize_inpainting_prediction(
    pred: torch.Tensor,
    sparse: torch.Tensor,
    known_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Reinsert known voxels into a predicted inpainting volume.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted reconstruction tensor or array.
    sparse : torch.Tensor
        Sparse observation tensor or array.
    known_mask : torch.Tensor
        Binary mask that marks voxels observed in the sparse input.
    
    Returns
    -------
    torch.Tensor
        Prediction tensor after reinserting the known voxels.
    """
    pred = torch.clamp(pred, 0.0, 1.0)
    sparse = sparse.float()
    known_mask = known_mask.float()
    return known_mask * sparse + (1.0 - known_mask) * pred


@dataclass
class AmpManager:
    """
    Manage automatic mixed precision for training and inference.
    
    Attributes
    ----------
    device : torch.device
        Stored value for device.
    enabled : bool
        Stored value for enabled.
    scaler : Optional[torch.cuda.amp.GradScaler]
        Stored value for scaler.
    """
    device: torch.device
    enabled: bool = True
    scaler: Optional[torch.cuda.amp.GradScaler] = None

    def __post_init__(self) -> None:
        """
        Initialize AMP state for the selected device.
        
        Returns
        -------
        None
            This method updates the instance in place.
        """
        if self.enabled and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def autocast(self):
        """
        Return the autocast context for the active device.
        
        Returns
        -------
        Any
            Autocast context manager for the active device.
        """
        if self.enabled and self.device.type == "cuda":
            return torch.cuda.amp.autocast()
        return nullcontext()

    def backward_and_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        """
        Backpropagate a loss tensor and update the optimizer.
        
        Parameters
        ----------
        loss : torch.Tensor
            Loss tensor used for backpropagation.
        optimizer : torch.optim.Optimizer
            Optimizer that should be updated after backpropagation.
        
        Returns
        -------
        None
            This function is executed for its side effects.
        """
        if self.scaler is None:
            loss.backward()
            optimizer.step()
            return
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
