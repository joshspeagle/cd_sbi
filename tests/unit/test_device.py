import torch
import pytest

from cdsbi.device import get_device


def test_get_device_auto():
    d = get_device("auto")
    assert d.type in {"cpu", "cuda"}
    if torch.cuda.is_available():
        assert d.type == "cuda"
    else:
        assert d.type == "cpu"


def test_get_device_cpu_force():
    assert get_device("cpu").type == "cpu"


def test_get_device_invalid():
    with pytest.raises(ValueError):
        get_device("tpu")
