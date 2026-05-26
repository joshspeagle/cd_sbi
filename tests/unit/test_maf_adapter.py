import torch
from cdsbi.flows.maf_adapter import MAFAdapter


def test_maf_adapter_no_guarantees():
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    assert flow.monotonicity_guarantees == frozenset()


def test_maf_adapter_log_prob_shape(seed):
    torch.manual_seed(seed)
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    x = torch.randn(10, 1)
    context = torch.randn(10, 1)
    log_p = flow.log_prob(x, context)
    assert log_p.shape == (10,)


def test_maf_adapter_n_params_matches_torch(seed):
    torch.manual_seed(seed)
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    assert flow.n_params() == sum(p.numel() for p in flow.parameters())
