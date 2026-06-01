import torch
from cdsbi.conditioners.moment_regression import MomentRegressionConditioner


def test_encode_shape_and_logdet_zero():
    cond = MomentRegressionConditioner(n_iid=10, d_theta=2)
    x = torch.randn(7, 10)
    feats, log_det = cond.encode(x)
    assert feats.shape == (7, 2)
    assert torch.allclose(log_det, torch.zeros(7))


def test_regression_target_theta_is_identity():
    cond = MomentRegressionConditioner(n_iid=10, d_theta=2, target="theta")
    theta = torch.tensor([[0.3, -1.0], [0.5, 2.0]])
    assert torch.equal(cond.regression_target(theta), theta)


def test_regression_target_theta_sq_is_coordinatewise_square():
    cond = MomentRegressionConditioner(n_iid=10, d_theta=1, target="theta_sq")
    theta = torch.tensor([[2.0], [-3.0]])
    assert torch.equal(cond.regression_target(theta), torch.tensor([[4.0], [9.0]]))
