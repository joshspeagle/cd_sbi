"""Capture environment metadata for a run."""
import platform as _platform
import subprocess
import sys

import torch


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def capture_env() -> dict:
    return {
        "git_sha": _git_sha(),
        "dirty_tree": _git_dirty(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "sbi_version": _package_version("sbi"),
        "nflows_version": _package_version("nflows"),
        "numpy_version": _package_version("numpy"),
        "scipy_version": _package_version("scipy"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": _platform.platform(),
        "num_threads": torch.get_num_threads(),
    }
