"""The MCP and hook wrappers must use the selected Python environment."""

from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def load_wrapper():
    spec = spec_from_file_location("plugin_python_tests", ".claude-plugin/run_mnemos_mcp.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", ["_resolve_runtime_python", "_ready_runtime_python"])
def test_both_entry_points_resolve_commands_from_path(tmp_path, method):
    folder = tmp_path / "selected runtime"
    folder.mkdir()
    executable = folder / ("python.exe" if os.name == "nt" else "python")
    executable.touch()
    executable.chmod(0o700)
    env = {"MNEMOS_PLUGIN_PYTHON": "python", "PATH": str(folder)}

    selected = getattr(load_wrapper(), method)(tmp_path, env)

    assert selected is not None and Path(selected) == executable


@pytest.mark.parametrize("method", ["_resolve_runtime_python", "_ready_runtime_python"])
@pytest.mark.parametrize("relative", [False, True])
def test_explicit_runtime_paths_are_preserved(tmp_path, method, relative):
    value = "./runtime with spaces/python" if relative else str(tmp_path / "runtime python")

    assert getattr(load_wrapper(), method)(tmp_path, {"MNEMOS_PLUGIN_PYTHON": value}) == value


@pytest.mark.parametrize("method", ["_resolve_runtime_python", "_ready_runtime_python"])
def test_unlisted_current_directory_does_not_supply_the_interpreter(tmp_path, monkeypatch, method):
    current = tmp_path / "current"
    runtime = tmp_path / "runtime"
    current.mkdir()
    runtime.mkdir()
    name = "python.exe" if os.name == "nt" else "python"
    for folder in [current, runtime]:
        executable = folder / name
        executable.touch()
        executable.chmod(0o700)
    monkeypatch.chdir(current)
    # Locate the wrapper independently of the changed working directory.
    wrapper_path = Path(__file__).resolve().parents[1] / ".claude-plugin/run_mnemos_mcp.py"
    spec = spec_from_file_location("plugin_current_directory_tests", wrapper_path)
    assert spec is not None and spec.loader is not None
    wrapper = module_from_spec(spec)
    spec.loader.exec_module(wrapper)

    selected = getattr(wrapper, method)(
        tmp_path, {"MNEMOS_PLUGIN_PYTHON": name, "PATH": str(runtime)}
    )

    assert selected is not None and Path(selected) == runtime / name


def process_env(tmp_path, command):
    env = {key: value for key, value in os.environ.items() if key in {"SYSTEMROOT", "WINDIR"}}
    env.update(
        CLAUDE_PLUGIN_ROOT=str(tmp_path / "uninstalled-plugin"),
        MNEMOS_PLUGIN_PYTHON=command,
        MNEMOS_HOST="manual",
        PATH=str(Path(sys.executable).parent),
        PYTHONPATH=str(tmp_path),
        PYTHONIOENCODING="utf-8",
    )
    return env


@pytest.mark.parametrize("entry", ["mcp", "hook"])
def test_wrapper_child_keeps_its_environment_and_protocol_streams(tmp_path, entry):
    package = tmp_path / "mnemos"
    (package / "adapters").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "adapters/__init__.py").write_text("", encoding="utf-8")
    child = package / ("mcp_server.py" if entry == "mcp" else "adapters/claude_code.py")
    child.write_text(
        "import json, os, sys\n"
        "print(json.dumps({'prefix': sys.prefix, 'payload': sys.stdin.read(), "
        "'host': os.environ['MNEMOS_HOST']}))\n"
        "raise SystemExit(23)\n",
        encoding="utf-8",
    )
    wrapper = Path(f".claude-plugin/run_mnemos_{entry}.py").resolve()
    command = [sys.executable, str(wrapper)] + (["Stop"] if entry == "hook" else [])
    completed = subprocess.run(
        command,
        env=process_env(tmp_path, "python"),
        cwd=tmp_path,
        input="synthetic input",
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 23, completed.stderr
    payload = json.loads(completed.stdout)
    assert Path(payload["prefix"]).resolve() == Path(sys.prefix).resolve()
    assert payload["payload"] == "synthetic input" and payload["host"] == "claude-code"
    assert completed.stderr == ""
    assert not (tmp_path / "uninstalled-plugin").exists()


@pytest.mark.parametrize("entry", ["mcp", "hook"])
def test_unavailable_explicit_command_does_not_install_or_block(tmp_path, entry):
    wrapper = Path(f".claude-plugin/run_mnemos_{entry}.py").resolve()
    command = [sys.executable, str(wrapper)] + (["Stop"] if entry == "hook" else [])
    env = process_env(tmp_path, "mnemos-python-not-on-path")
    env["PATH"] = str(tmp_path)
    completed = subprocess.run(
        command,
        env=env,
        cwd=tmp_path,
        input="synthetic input",
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == (0 if entry == "hook" else 1)
    assert completed.stdout == "" and "Traceback" not in completed.stderr
    expected = "reason=runtime_not_ready" if entry == "hook" else "MNEMOS_PLUGIN_PYTHON"
    assert expected in completed.stderr
    assert not (tmp_path / "uninstalled-plugin").exists()
