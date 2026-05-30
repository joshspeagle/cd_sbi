"""SufficiencyRecovery: does a trained learned summary retain the sufficient info?

Spearman (rank, monotone-invariant) of each oracle-summary coordinate against the
trained learned features. Monotone-invariant so a curved-but-equivalent feature
(e.g. ∝ s² rather than log s²) still scores ~1. No-ops unless the procedure
exposes encode_fn (a learned summary) and the simulator exposes oracle_summary.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr, ConstantInputWarning

from cdsbi.diagnostics.base import DiagnosticResult

_ORACLE_NAMES = ["log_s2", "xbar"]


class SufficiencyRecovery:
    name = "sufficiency_recovery"

    def __init__(self, n_eval: int = 4000, pass_threshold: float = 0.9, seed: int = 123):
        self.n_eval = n_eval
        self.pass_threshold = pass_threshold
        self.seed = seed

    def _noop(self, reason: str) -> DiagnosticResult:
        return DiagnosticResult(self.name, value=pd.DataFrame(), passed=True,
                                noise_floor=0.0, n_samples=0, meta={"reason": reason})

    def __call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult:
        proc = getattr(trained, "procedure", None)
        encode_fn = getattr(proc, "encode_fn", None)
        if encode_fn is None:
            return self._noop("procedure has no encode_fn (no learned summary)")
        if not hasattr(simulator, "oracle_summary"):
            return self._noop("simulator has no oracle_summary")
        rng = np.random.default_rng(self.seed)
        _, x = simulator.sample(self.n_eval, rng)
        with torch.no_grad():
            feats = encode_fn(x).detach().cpu().numpy()                  # (n, d_out)
            oracle = simulator.oracle_summary(x).detach().cpu().numpy()  # (n, d_oracle)
        row = {}
        spearmans = []
        # A collapsed learned feature is constant → spearmanr/pearsonr return NaN.
        # Use np.nanmax (NOT max(), which is order-dependent with NaN) so a real
        # recovery on the OTHER feature index isn't masked, and suppress the warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            for k in range(oracle.shape[1]):
                name = _ORACLE_NAMES[k] if k < len(_ORACLE_NAMES) else f"coord{k}"
                sp = np.nanmax([abs(spearmanr(oracle[:, k], feats[:, j]).statistic)
                                for j in range(feats.shape[1])])
                pe = np.nanmax([abs(pearsonr(oracle[:, k], feats[:, j])[0])
                                for j in range(feats.shape[1])])
                row[f"spearman_{name}"] = float(sp)
                row[f"pearson_{name}"] = float(pe)
                spearmans.append(sp)
        row["sufficiency_min_spearman"] = float(np.nanmin(spearmans))
        df = pd.DataFrame([row])
        passed = bool(row["sufficiency_min_spearman"] > self.pass_threshold)
        return DiagnosticResult(self.name, value=df, passed=passed,
                                noise_floor=0.0, n_samples=self.n_eval,
                                meta={"pass_threshold": self.pass_threshold})
