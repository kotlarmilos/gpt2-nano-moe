from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any

import torch


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def configure_deterministic_execution(device: torch.device) -> None:
    torch.use_deterministic_algorithms(True)
    if device.type != "cuda":
        return
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "CUDA publication runs require CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "or CUBLAS_WORKSPACE_CONFIG=:16:8"
        )
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def environment_metadata(device: torch.device) -> dict[str, Any]:
    commit: str | None = None
    git_dirty: bool | None = None
    git_status_sha256: str | None = None
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode == 0:
            encoded_status = status.stdout.encode()
            git_dirty = bool(encoded_status)
            git_status_sha256 = hashlib.sha256(encoded_status).hexdigest()
    except OSError:
        pass
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "git_commit": commit,
        "git_dirty": git_dirty,
        "git_status_sha256": git_status_sha256,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_sdp_backends": (
            {
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
            }
            if device.type == "cuda"
            else None
        ),
        "pid": os.getpid(),
        "command": sys.argv,
    }
