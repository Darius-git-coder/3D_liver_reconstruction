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
