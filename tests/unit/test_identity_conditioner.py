import torch
from cdsbi.conditioners.identity import Identity


def test_identity_returns_x_unchanged():
    cond = Identity()
    x = torch.randn(5, 1)
    ctx, log_det_contrib = cond.encode(x)
    assert torch.equal(ctx, x)
    assert torch.equal(log_det_contrib, torch.zeros(5))


def test_identity_n_params():
    cond = Identity()
    assert cond.n_params() == 0
