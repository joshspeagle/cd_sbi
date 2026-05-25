"""Shared pytest fixtures."""
import os
import pytest
import torch

# Force deterministic CUDA when present; harmless on CPU
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture
def seed():
    """Default seed for tests that need one."""
    return 12345


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Throwaway run directory."""
    d = tmp_path / "run"
    d.mkdir()
    return d
