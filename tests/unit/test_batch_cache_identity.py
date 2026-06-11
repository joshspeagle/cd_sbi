"""_BatchCache identity safety (hardening item 2 follow-up).

The cache keys embed id(x_obs_batch), but CPython recycles ids when objects
die — and the chunked eval engine frees each x-slice before creating the next,
so later chunks could silently receive an earlier chunk's cached values (or
crash on a shape mismatch, which is how this surfaced: a 464-row final chunk
receiving a 512-row cached ll_max). The fix requires a weakref-validated
``anchor``: a hit needs the keyed batch to be the SAME LIVE OBJECT.
"""
import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import _BatchCache, LikelihoodBasedProcedure
from cdsbi.diagnostics.engine import evaluate_coverage


def test_same_live_anchor_hits():
    cache = _BatchCache()
    x = torch.zeros(4, 2)
    calls = []
    v1 = cache.get_or_compute(("k", 1), lambda: calls.append(1) or "V1", anchor=x)
    v2 = cache.get_or_compute(("k", 1), lambda: calls.append(2) or "V2", anchor=x)
    assert v1 == v2 == "V1" and calls == [1]  # second call was a cache hit


def test_different_object_same_key_misses():
    """The recycled-id scenario, made deterministic: two DIFFERENT tensors
    presented under the SAME key must NOT share a cache entry."""
    cache = _BatchCache()
    a = torch.zeros(4, 2)
    b = torch.ones(3, 2)  # different object (and shape, as in the real crash)
    va = cache.get_or_compute(("k", 1), lambda: a.shape[0], anchor=a)
    vb = cache.get_or_compute(("k", 1), lambda: b.shape[0], anchor=b)
    assert va == 4 and vb == 3  # b recomputed; no stale 4-row value


def test_dead_anchor_misses():
    cache = _BatchCache()
    a = torch.zeros(4, 2)
    cache.get_or_compute(("k", 1), lambda: "old", anchor=a)
    del a  # anchor dies; weakref invalidates the entry
    b = torch.zeros(5, 2)
    v = cache.get_or_compute(("k", 1), lambda: "new", anchor=b)
    assert v == "new"


def test_no_anchor_backward_compatible():
    cache = _BatchCache()
    v1 = cache.get_or_compute(("k",), lambda: "A")
    v2 = cache.get_or_compute(("k",), lambda: "B")
    assert v1 == v2 == "A"


def test_chunked_engine_repro_nle():
    """The exact crash scenario: NLE-style procedure evaluated through the
    chunked engine with n_per_theta NOT divisible by the chunk size (2000 =
    512+512+512+464). Pre-fix this crashed (512-row cached ll_max applied to
    the 464-row chunk) or silently corrupted same-size chunks. Must now match
    the closed-form Wilks answer."""

    class _Sim:
        d_theta = 2
        theta_range = (-7.0, 7.0)

        def sample_x_given_theta(self, theta_0, n, rng):
            t = torch.tensor(theta_0, dtype=torch.float32)
            noise = torch.from_numpy(rng.standard_normal((n, 2))).to(torch.float32)
            return t + noise

    def ll(theta, x):  # exact Gaussian: Wilks stat = ||theta0 - x||^2
        return -0.5 * ((theta - x) ** 2).sum(dim=-1)

    proc = LikelihoodBasedProcedure(ll, 2, theta_range=(-7.0, 7.0))
    sim = _Sim()
    out = evaluate_coverage(
        proc, sim, [(0.0, 0.0)], [0.5, 0.9], n_per_theta=2000, chunk_size=512, seed=3,
    )
    df = out["coverage"]
    # Exact statistic + exact threshold: empirical coverage within MC error.
    for _, row in df.iterrows():
        se = (row["nominal"] * (1 - row["nominal"]) / 2000) ** 0.5
        assert abs(row["empirical"] - row["nominal"]) < 4 * se, (
            f"coverage {row['empirical']:.3f} vs nominal {row['nominal']:.2f}"
        )
