from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


def _run_git_command(args: Sequence[str], cwd: str) -> str | None:
    """
    Run a git command and return its stripped standard output.
    
    Parameters
    ----------
    args : Sequence[str]
        Arguments passed to the helper or command.
    cwd : str
        Working directory used for command execution or metadata lookup.
    
    Returns
    -------
    str | None
        Standard output of the git command, or None when execution fails.
    """
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def get_git_info(start_dir: str) -> Dict[str, Any]:
    """
    Collect git metadata for the current working tree.
    
    Parameters
    ----------
    start_dir : str
        Directory from which repository metadata lookup starts.
    
    Returns
    -------
    Dict[str, Any]
        Collected metadata.
    """
    root = _run_git_command(["git", "rev-parse", "--show-toplevel"], cwd=start_dir)
    if not root:
        return {
            "available": False,
            "root": None,
            "commit": None,
            "short_commit": None,
            "is_dirty": None,
        }

    commit = _run_git_command(["git", "rev-parse", "HEAD"], cwd=root)
    short_commit = _run_git_command(["git", "rev-parse", "--short", "HEAD"], cwd=root)
    status = _run_git_command(["git", "status", "--porcelain"], cwd=root)
    return {
        "available": True,
        "root": os.path.normpath(root),
        "commit": commit,
        "short_commit": short_commit,
        "is_dirty": bool(status),
    }


def build_repro_metadata(
    args: Mapping[str, Any],
    *,
    command: Sequence[str] | None = None,
    split_manifest_path: str | None = None,
    extra: Optional[Mapping[str, Any]] = None,
    cwd: str | None = None,
) -> Dict[str, Any]:
    """
    Assemble experiment metadata for reproducibility tracking.
    
    Parameters
    ----------
    args : Mapping[str, Any]
        Arguments passed to the helper or command.
    command : Sequence[str] | None
        Command sequence that should be executed or recorded. Defaults to None.
    split_manifest_path : str | None
        Filesystem path to a split manifest. Defaults to None.
    extra : Optional[Mapping[str, Any]]
        Additional metadata merged into the serialized output. Defaults to None.
    cwd : str | None
        Working directory used for command execution or metadata lookup. Defaults to None.
    
    Returns
    -------
    Dict[str, Any]
        Constructed object ready for downstream use.
    """
    workdir = os.path.normpath(cwd or os.getcwd())
    payload: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": workdir,
        "command": list(command) if command is not None else list(sys.argv),
        "args": dict(args),
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "git": get_git_info(workdir),
        "split_manifest_path": (
            os.path.normpath(os.path.abspath(split_manifest_path))
            if split_manifest_path
            else None
        ),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload
