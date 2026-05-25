import json
import pytest
from cdsbi.reproducibility.run_dir import RunDir, RunStatus


def test_set_status_writes_atomic(tmp_run_dir):
    rd = RunDir(tmp_run_dir)
    rd.set_status(RunStatus.RUNNING)
    assert (tmp_run_dir / "STATUS").read_text().strip() == "RUNNING"
    rd.set_status(RunStatus.OK)
    assert (tmp_run_dir / "STATUS").read_text().strip() == "OK"


def test_write_atomic_no_partial_on_crash(tmp_run_dir):
    rd = RunDir(tmp_run_dir)
    rd.write_atomic(tmp_run_dir / "config.yaml", "key: value\n")
    assert (tmp_run_dir / "config.yaml").read_text() == "key: value\n"
    # No leftover .tmp file
    assert not (tmp_run_dir / "config.yaml.tmp").exists()


def test_write_atomic_overwrites_existing(tmp_run_dir):
    rd = RunDir(tmp_run_dir)
    p = tmp_run_dir / "x.json"
    rd.write_atomic(p, json.dumps({"a": 1}))
    rd.write_atomic(p, json.dumps({"a": 2}))
    assert json.loads(p.read_text()) == {"a": 2}
