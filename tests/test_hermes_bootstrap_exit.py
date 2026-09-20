from types import SimpleNamespace

import pytest

from benchmarks.native_hosts.hermes_native_bootstrap_v7 import exit_origins


def fixture_modules():
    cfg = {"hermes_checkout": "/isolated", "phase_root": "/phase"}
    modules = {
        name: SimpleNamespace(__file__="/isolated/" + name.replace(".", "/") + ".py")
        for name in (
            "hermes_cli.main",
            "hermes_cli.cli_agent_setup_mixin",
            "agent.conversation_loop",
            "agent.memory_manager",
        )
    }
    modules["mnemos.adapters.hermes"] = SimpleNamespace(
        __file__="/phase/wheel/mnemos/adapters/hermes.py"
    )
    return cfg, modules


def test_exit_binds_native_modules_and_verified_wheel():
    cfg, modules = fixture_modules()
    row = exit_origins(cfg, modules)
    assert row["operation"] == "bootstrap_exit"
    assert len(row["origins"]) == 4
    assert row["adapter_origin"] == modules["mnemos.adapters.hermes"].__file__


@pytest.mark.parametrize("fault", ["missing", "live-host", "prefix-collision", "wrong-wheel"])
def test_exit_rejects_unbound_imports(fault):
    cfg, modules = fixture_modules()
    if fault == "missing":
        del modules["agent.memory_manager"]
    elif fault == "live-host":
        modules["agent.memory_manager"].__file__ = "/live/agent/memory_manager.py"
    elif fault == "prefix-collision":
        modules["agent.memory_manager"].__file__ = "/isolated-other/agent/memory_manager.py"
    else:
        modules["mnemos.adapters.hermes"].__file__ = "/live/mnemos/adapters/hermes.py"
    with pytest.raises(ValueError, match="escaped"):
        exit_origins(cfg, modules)
