"""BartlettSummaryConditioner: oracle X → (log D11, log D22, D21, X̄1, X̄2)."""
import numpy as np
import torch


def test_encode_matches_simulator_oracle_summary():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.conditioners.bartlett_summary import BartlettSummaryConditioner
    sim = NormalBivariateUnknownCov()
    cond = BartlettSummaryConditioner(n_iid=sim.n_iid)
    x = sim.sample_x_given_theta((0.0, 0.0, 0.3, 1.0, -1.0), 64, np.random.default_rng(0))
    feats, log_det = cond.encode(x)
    assert feats.shape == (64, 5) and log_det.shape == (64,)
    assert torch.allclose(feats, sim.oracle_summary(x), atol=1e-5)
    assert cond.n_params() == 0
