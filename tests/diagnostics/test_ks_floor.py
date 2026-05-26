import math
from cdsbi.diagnostics.ks_floor import ks_noise_floor


def test_ks_noise_floor_marginal_N5000():
    expected = 1.628 / math.sqrt(5000)
    assert abs(ks_noise_floor(N=5000, n_bins=1) - expected) < 1e-12


def test_ks_noise_floor_per_bin_N5000_k5():
    expected = 1.628 / math.sqrt(5000 / 5)
    assert abs(ks_noise_floor(N=5000, n_bins=5) - expected) < 1e-12


def test_ks_noise_floor_per_bin_N6000_k7():
    expected = 1.628 / math.sqrt(6000 / 7)
    assert abs(ks_noise_floor(N=6000, n_bins=7) - expected) < 1e-12
