import numpy as np
import torch
from cdsbi.reproducibility.seeding import seed_everything


def test_seed_everything_returns_named_streams(seed):
    rngs = seed_everything(seed)
    assert hasattr(rngs, "train")
    assert hasattr(rngs, "eval")
    assert hasattr(rngs, "init")
    # Streams are distinct
    assert rngs.train.bit_generator.state != rngs.eval.bit_generator.state


def test_seed_everything_is_deterministic(seed):
    seed_everything(seed)
    a = torch.randn(5)
    b = np.random.randn(5)
    seed_everything(seed)
    c = torch.randn(5)
    d = np.random.randn(5)
    assert torch.equal(a, c)
    np.testing.assert_array_equal(b, d)


def test_eval_stream_differs_from_train_stream(seed):
    rngs = seed_everything(seed)
    train_draw = rngs.train.standard_normal(3)
    rngs2 = seed_everything(seed)
    eval_draw = rngs2.eval.standard_normal(3)
    # Different streams ⇒ different draws even from the same master seed
    assert not np.allclose(train_draw, eval_draw)
