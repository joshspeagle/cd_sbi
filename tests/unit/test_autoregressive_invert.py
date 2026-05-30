"""autoregressive_invert: forward∘invert ≈ identity for SingleIndexMonotoneFlow."""
import torch


def test_invert_recovers_theta():
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.flows.invert import autoregressive_invert
    torch.manual_seed(0)
    flow = SingleIndexMonotoneFlow(d=5, theta_signs=[1, 1, -1, 1, 1],
                                   feat_signs=[-1, -1, 1, -1, -1], hidden=16).eval()
    theta = torch.empty(64, 5).uniform_(-2, 2)
    feat = torch.randn(64, 5)
    with torch.no_grad():
        r, _ = flow.forward(theta, context=feat)
        theta_rec = autoregressive_invert(flow, r, feat, lo=-6.0, hi=6.0, iters=40)
    assert torch.allclose(theta_rec, theta, atol=1e-2), \
        f"max err {(theta_rec - theta).abs().max():.4f}"
