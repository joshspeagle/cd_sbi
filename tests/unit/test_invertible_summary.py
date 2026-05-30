"""InvertibleSummaryConditioner: bijection X→(S,A); encode→(S, log|∂z/∂X|)."""
import torch


def _cond(n_iid=10, d_theta=2):
    torch.manual_seed(0)
    from cdsbi.conditioners.invertible_summary import InvertibleSummaryConditioner
    return InvertibleSummaryConditioner(n_iid=n_iid, d_theta=d_theta, hidden=16, n_layers=4)


def test_encode_returns_S_and_bijection_logdet():
    cond = _cond()
    x = torch.randn(8, 10)
    S, log_det = cond.encode(x)
    assert S.shape == (8, 2) and log_det.shape == (8,)


def test_transform_returns_full_latent():
    cond = _cond()
    x = torch.randn(8, 10)
    z, log_det = cond.transform(x)
    assert z.shape == (8, 10) and log_det.shape == (8,)
    S, ld2 = cond.encode(x)
    assert torch.allclose(z[:, :2], S) and torch.allclose(log_det, ld2)


def test_is_module_with_params():
    cond = _cond()
    assert isinstance(cond, torch.nn.Module) and cond.n_params() > 0
