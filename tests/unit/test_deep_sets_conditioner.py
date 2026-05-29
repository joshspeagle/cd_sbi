"""DeepSetsConditioner: permutation-invariant learned summary, device-1 (log_det=0)."""
import torch


def _cond(n_iid=10, hidden=32):
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    return DeepSetsConditioner(n_iid=n_iid, d_out=2, hidden=hidden)


def test_encode_shapes_and_zero_logdet():
    cond = _cond()
    x = torch.randn(16, 10)
    feats, log_det = cond.encode(x)
    assert feats.shape == (16, 2)
    assert log_det.shape == (16,)
    assert torch.allclose(log_det, torch.zeros(16))


def test_permutation_invariant_over_iid_axis():
    torch.manual_seed(0)
    cond = _cond().eval()
    x = torch.randn(8, 10)
    perm = torch.randperm(10)
    f1, _ = cond.encode(x)
    f2, _ = cond.encode(x[:, perm])
    assert torch.allclose(f1, f2, atol=1e-5)


def test_has_trainable_params_and_is_module():
    cond = _cond()
    assert isinstance(cond, torch.nn.Module)
    assert cond.n_params() > 0


def test_running_standardization_eval_uses_buffers():
    cond = _cond()
    x = torch.randn(64, 10)
    cond.train(); cond.encode(x)
    cond.eval()
    f1, _ = cond.encode(x); f2, _ = cond.encode(x)
    assert torch.allclose(f1, f2)
