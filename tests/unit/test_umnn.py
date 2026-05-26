import torch
from cdsbi.flows.umnn import UMNNBlock


def test_umnn_strictly_monotone_on_grid(seed):
    torch.manual_seed(seed)
    block = UMNNBlock(context_dim=0, hidden=32)
    z = torch.linspace(-3, 3, 200).unsqueeze(-1)
    g = block(z, context=None)
    # strictly increasing
    diffs = g[1:] - g[:-1]
    assert (diffs > 0).all()


def test_umnn_jacobian_factor_positive(seed):
    torch.manual_seed(seed)
    block = UMNNBlock(context_dim=0, hidden=32)
    z = torch.linspace(-3, 3, 200).unsqueeze(-1)
    j = block.jacobian_factor(z, context=None)
    assert (j > 0).all()


def test_umnn_with_context_shape(seed):
    torch.manual_seed(seed)
    block = UMNNBlock(context_dim=3, hidden=16)
    z = torch.randn(10, 1)
    ctx = torch.randn(10, 3)
    g = block(z, context=ctx)
    assert g.shape == (10, 1)


def test_umnn_n_params_matches_torch(seed):
    torch.manual_seed(seed)
    block = UMNNBlock(context_dim=0, hidden=32)
    counted = block.n_params()
    actual = sum(p.numel() for p in block.parameters())
    assert counted == actual
