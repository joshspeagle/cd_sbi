"""MultiQuantileMLP: parameter count and forward shape."""
from __future__ import annotations

import torch

from cdsbi.methods.lf2i import MultiQuantileMLP, multi_pinball_loss


def test_multi_quantile_param_count_formula():
    # H² + 3H (trunk: input_dim=1, depth=2) + n_q * (H + 1) (heads)
    for H, n_q in [(8, 3), (12, 4), (16, 5)]:
        net = MultiQuantileMLP(input_dim=1, hidden=H, depth=2, n_quantiles=n_q)
        actual = sum(p.numel() for p in net.parameters())
        expected = H * H + 3 * H + n_q * (H + 1)
        assert actual == expected, f"H={H} n_q={n_q}: actual={actual}, expected={expected}"


def test_multi_quantile_forward_shape():
    net = MultiQuantileMLP(input_dim=1, hidden=8, depth=2, n_quantiles=4)
    out = net(torch.randn(32, 1))
    assert out.shape == (32, 4)


def test_multi_quantile_param_count_independent_of_alpha_grid_len_for_lf2i():
    """LF2IRunner.n_params should grow linearly in n_q only through the
    output heads (n_q * (H+1)), not through duplicated trunks."""
    from cdsbi.methods.lf2i import LF2IRunner

    class _DummyFlow:
        def n_params(self):
            return 1000

    runner = LF2IRunner(stat_flow=_DummyFlow(), quantile_hidden=12, quantile_depth=2)
    n3 = runner.n_params(alpha_grid_len=3)["total"]
    n4 = runner.n_params(alpha_grid_len=4)["total"]
    n10 = runner.n_params(alpha_grid_len=10)["total"]
    # Per-α head contributes only (H+1) = 13 params per additional quantile.
    assert n4 - n3 == 13
    assert n10 - n4 == 6 * 13


def test_multi_pinball_loss_recovers_per_alpha_quantile(seed):
    """Train an MLP to convergence on a known quantile regression problem;
    verify the trained head returns ≈ the population α-quantile."""
    from cdsbi.reproducibility.seeding import seed_everything
    seed_everything(seed)

    # y | x ~ x + N(0, 1); the α-quantile is x + Φ⁻¹(α).
    from scipy.stats import norm
    alphas = [0.1, 0.5, 0.9]
    net = MultiQuantileMLP(input_dim=1, hidden=16, depth=2, n_quantiles=3)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    n = 4000
    x = torch.linspace(-2, 2, n).unsqueeze(-1)
    eps = torch.randn(n)
    y = x.squeeze(-1) + eps
    for _ in range(2000):
        preds = net(x)
        loss = multi_pinball_loss(preds, y, alphas)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # At x=0, true quantiles are Φ⁻¹(α).
    with torch.no_grad():
        pred_at_0 = net(torch.zeros(1, 1)).squeeze(0)
    for k, a in enumerate(alphas):
        truth = float(norm.ppf(a))
        assert abs(pred_at_0[k].item() - truth) < 0.20, (
            f"alpha={a}: pred={pred_at_0[k].item():.3f}, truth={truth:.3f}"
        )
