import math
from cdsbi.confidence_set.root_find import bisect_1d


def test_bisect_1d_finds_root():
    root = bisect_1d(lambda x: x - 3.0, 0.0, 10.0, tol=1e-6)
    assert abs(root - 3.0) < 1e-5


def test_bisect_1d_nonlinear():
    root = bisect_1d(lambda x: x**3 - 8.0, 0.0, 10.0, tol=1e-6)
    assert abs(root - 2.0) < 1e-5


def test_bisect_1d_no_sign_change_returns_midpoint():
    """If f doesn't change sign on [a, b], return midpoint with a warning (graceful)."""
    root = bisect_1d(lambda x: x**2 + 1.0, -1.0, 1.0, tol=1e-6)
    # Should not raise; should return something in [-1, 1]
    assert -1.0 <= root <= 1.0
