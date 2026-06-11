"""CPF plan step 3 — isotonic (rearrangement) monotone-in-level head."""
import torch

from cdsbi.methods.cpf.monotone_alpha import MonotoneQuantileHead, rearrange_quantiles


def test_crossing_rows_become_monotone():
    q = torch.tensor([[3.0, 1.0, 2.0], [0.5, 0.4, 0.6], [1.0, 1.0, 0.9]])
    out = rearrange_quantiles(q)
    assert bool((out.diff(dim=-1) >= 0).all())


def test_idempotent():
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(64, 5, generator=gen)
    once = rearrange_quantiles(q)
    assert torch.equal(rearrange_quantiles(once), once)


def test_multiset_preserved_per_row():
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(32, 7, generator=gen)
    out = rearrange_quantiles(q)
    assert torch.allclose(torch.sort(q, dim=-1).values, out)


def test_already_monotone_unchanged():
    q = torch.tensor([[0.1, 0.2, 0.3], [-1.0, 0.0, 5.0]])
    assert torch.equal(rearrange_quantiles(q), q)


def test_wrapped_multiquantile_mlp_is_noncrossing():
    """The real per-level head (independent linear heads, no cross-level
    coupling — the documented crossing source) wrapped -> monotone output."""
    from cdsbi.methods.lf2i import MultiQuantileMLP

    torch.manual_seed(3)
    base = MultiQuantileMLP(input_dim=2, hidden=16, depth=2, n_quantiles=4)
    head = MonotoneQuantileHead(base)
    theta = torch.randn(128, 2)
    out = head(theta)
    assert out.shape == (128, 4)
    assert bool((out.diff(dim=-1) >= 0).all())
