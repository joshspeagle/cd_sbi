"""Checkpoint loader: model.pt -> {arch_metadata, final_loss, loss_history_tail}."""
from __future__ import annotations

import pytest

from tests.figures_fixtures import make_run_dir


def test_load_checkpoint_returns_arch_metadata_and_final_loss(tmp_path):
    from cdsbi.analysis.figures.data_io.checkpoints import load_checkpoint
    rd = make_run_dir(
        tmp_path, method="cd_sbi", budget_name="medium", seed=0,
        coverage_error_max=0.025, final_loss=1.43,
        loss_history_tail=[1.5, 1.45, 1.43],
    )
    ck = load_checkpoint(str(rd))
    assert ck.final_loss == pytest.approx(1.43)
    assert ck.loss_history_tail == [1.5, 1.45, 1.43]
    assert isinstance(ck.arch_metadata, dict)


def test_load_checkpoint_missing_file_raises(tmp_path):
    from cdsbi.analysis.figures.data_io.checkpoints import load_checkpoint
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path / "does_not_exist"))


def test_loss_history_tail_none_when_absent(tmp_path):
    from cdsbi.analysis.figures.data_io.checkpoints import load_checkpoint
    import torch
    rd = tmp_path / "run"
    rd.mkdir()
    torch.save({"arch_metadata": {}, "final_loss": 0.9}, rd / "model.pt")
    ck = load_checkpoint(str(rd))
    assert ck.loss_history_tail is None
    assert ck.final_loss == pytest.approx(0.9)
