"""Deterministic seeding for torch / numpy / python; named per-purpose RNG streams."""
import hashlib
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SeededRNGs:
    train: np.random.Generator
    eval: np.random.Generator
    init: np.random.Generator


def _derive(master: int, label: str) -> int:
    """Deterministic 64-bit substream seed from (master, label)."""
    h = hashlib.sha256(f"{master}:{label}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63 - 1)


def seed_everything(seed: int) -> SeededRNGs:
    """Seed all global RNGs and return named numpy.Generator streams.

    `sbi`-based runners pass explicit per-call seeds derived from the
    same master; they do not rely on global `sbi` state.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return SeededRNGs(
        train=np.random.default_rng(_derive(seed, "train")),
        eval=np.random.default_rng(_derive(seed, "eval")),
        init=np.random.default_rng(_derive(seed, "init")),
    )
