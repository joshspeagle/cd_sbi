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


def test_bias_trainable_false_freezes_context_bias():
    """bias_trainable=False must freeze the linear-bias term when context_dim > 0,
    matching the scalar-context branch (which freezes bias_param.requires_grad).
    Otherwise the redundant-bias degeneracy from v0.2 is back."""
    from cdsbi.flows.umnn import UMNNBlock
    # context_dim > 0 branch
    block = UMNNBlock(context_dim=4, hidden=8, bias_trainable=False)
    assert block.bias_net is not None
    assert block.bias_net.bias.requires_grad is False
    # Weights of the bias_net stay trainable (only bias is frozen).
    assert block.bias_net.weight.requires_grad is True
    # Default bias_trainable=True still gives trainable bias.
    block_train = UMNNBlock(context_dim=4, hidden=8, bias_trainable=True)
    assert block_train.bias_net.bias.requires_grad is True


def test_bias_trainable_scalar_branch_unchanged():
    """Sanity: the context_dim==0 path still works as before."""
    from cdsbi.flows.umnn import UMNNBlock
    block_frozen = UMNNBlock(context_dim=0, hidden=8, bias_trainable=False)
    assert block_frozen.bias_param.requires_grad is False
    block_trainable = UMNNBlock(context_dim=0, hidden=8, bias_trainable=True)
    assert block_trainable.bias_param.requires_grad is True
