from cdsbi.reproducibility.env import capture_env


def test_capture_env_has_required_keys():
    env = capture_env()
    assert "git_sha" in env
    assert "dirty_tree" in env
    assert "python_version" in env
    assert "torch_version" in env
    assert "device" in env
    assert "platform" in env
    assert isinstance(env["dirty_tree"], bool)
