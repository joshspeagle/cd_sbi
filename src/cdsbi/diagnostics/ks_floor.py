"""KS-test noise floor: 1.628 / sqrt(N/k) for the per-bin case."""
from __future__ import annotations

import math


def ks_noise_floor(N: int, n_bins: int = 1) -> float:
    """Noise floor for a KS test at the 99% level: 1.628/√(N/k)."""
    return 1.628 / math.sqrt(N / n_bins)
