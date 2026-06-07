"""CPF three-split firewall RNG streams (plan 2026-06-07 step 1).

The Conformal Pivot Flow firewall needs three disjoint simulation splits:
``flow`` (train the pivot), ``cal`` (fit F-hat + conformal scores), ``audit``
(independent coverage). These must be deterministic from the master seed,
pairwise distinct, and disjoint by construction.
"""
import numpy as np

from cdsbi.reproducibility.seeding import SeededRNGs, _derive, seed_everything

_CPF_STREAMS = ("flow", "cal", "audit")
_ALL_STREAMS = ("train", "eval", "init", *_CPF_STREAMS)


def test_cpf_streams_exist_and_are_generators():
    rngs = seed_everything(0)
    for name in _CPF_STREAMS:
        assert hasattr(rngs, name), f"SeededRNGs is missing the {name!r} stream"
        assert isinstance(getattr(rngs, name), np.random.Generator)


def test_cpf_streams_deterministic_from_master():
    a = seed_everything(123).flow.random(1000)
    b = seed_everything(123).flow.random(1000)
    np.testing.assert_array_equal(a, b)
    # A different master seed yields a different stream.
    c = seed_everything(124).flow.random(1000)
    assert not np.array_equal(a, c)


def test_cpf_streams_pairwise_distinct():
    rngs = seed_everything(0)
    draws = {name: getattr(rngs, name).random(1000) for name in _ALL_STREAMS}
    for i, ni in enumerate(_ALL_STREAMS):
        for nj in _ALL_STREAMS[i + 1 :]:
            assert not np.array_equal(draws[ni], draws[nj]), (
                f"streams {ni!r} and {nj!r} produce identical draws"
            )


def test_cpf_stream_seeds_match_derive():
    # The new streams follow the existing _derive(seed, label) convention.
    seed = 7
    for label in _CPF_STREAMS:
        expected = np.random.default_rng(_derive(seed, label)).random(100)
        got = getattr(seed_everything(seed), label).random(100)
        np.testing.assert_array_equal(got, expected)
