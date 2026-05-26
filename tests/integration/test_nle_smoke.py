from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.nle import NLERunner
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_nle_smoke(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    runner = NLERunner(flow=flow)
    trained = runner.fit(simulator=sim, config={"n_train": 200, "n_epochs": 5}, seed=seed)
    assert trained.procedure is not None
