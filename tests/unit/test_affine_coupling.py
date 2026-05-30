"""AffineCouplingBijection: invertible ℝ^d→ℝ^d with tractable log|det|."""
import torch


def _bij(d=4, seed=0):
    torch.manual_seed(seed)
    from cdsbi.flows.affine_coupling import AffineCouplingBijection
    return AffineCouplingBijection(d=d, hidden=16, n_layers=4, depth=2)


def test_shapes():
    bij = _bij()
    x = torch.randn(8, 4)
    z, log_det = bij(x)
    assert z.shape == (8, 4) and log_det.shape == (8,)


def test_logdet_matches_autograd():
    import torch.nn as nn
    torch.manual_seed(1)
    bij = _bij()
    # the ctor zero-inits coupling output layers (near-identity start) — perturb them
    # so this test exercises NON-TRIVIAL scales (else it's a vacuous identity check).
    for net in list(bij.scale_nets) + list(bij.shift_nets):
        nn.init.normal_(net[-1].weight, std=0.5)
        nn.init.normal_(net[-1].bias, std=0.3)
    x = torch.randn(1, 4, requires_grad=True)
    z, log_det = bij(x)
    assert float(log_det.abs()) > 1e-3, "test must exercise non-identity scales"
    J = torch.zeros(4, 4)
    for i in range(4):
        (g,) = torch.autograd.grad(z[0, i], x, retain_graph=True)
        J[i] = g[0]
    assert torch.allclose(log_det[0], torch.log(torch.abs(torch.det(J))), atol=1e-4)


def test_nonsingular_jacobian():
    bij = _bij()
    x = torch.randn(16, 4)
    z, ld = bij(x)
    assert torch.all(torch.isfinite(ld)) and torch.all(torch.exp(ld) > 0)
