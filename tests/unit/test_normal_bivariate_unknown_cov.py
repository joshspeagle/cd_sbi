"""NormalBivariateUnknownCov: sampling, shapes, prior, signs."""
import numpy as np
import torch


def test_sample_shapes_and_dims():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    assert sim.d_theta == 5 and sim.d_x == 20
    theta, x = sim.sample(64, np.random.default_rng(0))
    assert theta.shape == (64, 5) and x.shape == (64, 20)


def test_sample_x_given_theta_is_bivariate_normal():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    theta0 = (0.2, -0.1, 0.5, 1.0, -2.0)
    x = sim.sample_x_given_theta(theta0, 20000, np.random.default_rng(1))
    assert x.shape == (20000, 20)
    obs = x.reshape(20000, sim.n_iid, 2).reshape(-1, 2).numpy()
    L = np.array([[np.exp(0.2), 0.0], [0.5, np.exp(-0.1)]])
    Sig = L @ L.T
    assert np.allclose(obs.mean(0), [1.0, -2.0], atol=0.05)
    assert np.allclose(np.cov(obs.T), Sig, atol=0.08)


def test_signs():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    assert tuple(sim.theta_signs) == (1.0, 1.0, -1.0, 1.0, 1.0)
    assert tuple(sim.feat_signs) == (-1.0, -1.0, 1.0, -1.0, -1.0)


def test_r_star_is_standard_normal_at_truth():
    import numpy as np
    from scipy.stats import kstest
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    rng = np.random.default_rng(3)
    theta0 = np.array([0.3, -0.2, 0.6, 0.5, -1.0])
    x = sim.sample_x_given_theta(theta0, 8000, rng)
    theta = torch.tensor(theta0, dtype=torch.float32).expand(8000, 5)
    r = sim.r_star(theta, x).numpy()
    assert r.shape == (8000, 5)
    for j in range(5):
        assert abs(r[:, j].mean()) < 0.06, f"coord {j} mean {r[:, j].mean():.3f}"
        assert abs(r[:, j].std() - 1.0) < 0.06, f"coord {j} std {r[:, j].std():.3f}"
        assert kstest(r[:, j], "norm").statistic < 0.04, f"coord {j} KS"
    corr = np.corrcoef(r.T)
    assert np.abs(corr - np.eye(5))[~np.eye(5, dtype=bool)].max() < 0.05


def test_oracle_summary_shape_and_data_entropy():
    import numpy as np, math
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    x = sim.sample_x_given_theta((0.0, 0.0, 0.0, 0.0, 0.0), 32, np.random.default_rng(0))
    feats = sim.oracle_summary(x)
    assert feats.shape == (32, 5)
    H = sim.data_entropy_lower_bound()
    mid = 0.5 * (sim.log_chol_range[0] + sim.log_chol_range[1])
    expected = sim.n_iid * (0.5 * sim.p * (1 + math.log(2 * math.pi)) + 2 * mid)
    assert abs(H - expected) < 1e-6


def test_entropy_lower_bound_is_finite_and_stable():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    H = sim.entropy_lower_bound(n_mc=40000, seed=0)
    assert 1.5 < H < 2.5, f"entropy floor {H:.3f} outside expected ~2.0"
    H2 = sim.entropy_lower_bound(n_mc=40000, seed=1)
    assert abs(H - H2) < 0.05
