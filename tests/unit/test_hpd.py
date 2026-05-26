import torch
from cdsbi.confidence_set.hpd import hpd_1d


def test_hpd_1d_standard_normal():
    torch.manual_seed(0)
    samples = torch.randn(20_000)
    cs = hpd_1d(samples, alpha=0.95)
    lo, hi = cs.boundary_repr.tolist()
    assert abs(lo + 1.96) < 0.05
    assert abs(hi - 1.96) < 0.05


def test_hpd_1d_contains_works():
    torch.manual_seed(0)
    samples = torch.randn(10_000)
    cs = hpd_1d(samples, alpha=0.90)
    assert cs.contains(0.0)
    assert not cs.contains(5.0)
