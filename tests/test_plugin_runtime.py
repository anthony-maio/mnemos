"""Focused tests for the Claude plugin runtime bootstrap."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from importlib.util import module_from_spec, spec_from_file_location


def _load_wrapper():
    wrapper_path = Path(".claude-plugin/run_mnemos_mcp.py")
    spec = spec_from_file_location("mnemos_plugin_runtime_tests", wrapper_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_create_venv(venv_dir: Path) -> None:
    python_path = _load_wrapper()._venv_python(venv_dir)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_bytes(b"fake python")


def _plugin_root(tmp_path: Path) -> Path:
    root = tmp_path / "plugin"
    (root / "mnemos").mkdir(parents=True)
    (root / "agents").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (root / "mnemos" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "agents" / "agent.md").write_text("agent\n", encoding="utf-8")
    return root


def test_source_digest_tracks_package_and_plugin_agent_source(tmp_path: Path) -> None:
    wrapper = _load_wrapper()
    root = _plugin_root(tmp_path)
    initial = wrapper._source_digest(root)

    cache = root / "mnemos" / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-313.pyc").write_bytes(b"cache")
    assert wrapper._source_digest(root) == initial

    (root / "agents" / "agent.md").write_text("changed\n", encoding="utf-8")
    assert wrapper._source_digest(root) != initial


def test_missing_interpreter_never_trusts_matching_install_stamp(
    monkeypatch, tmp_path: Path
) -> None:
    wrapper = _load_wrapper()
    root = _plugin_root(tmp_path)
    venv_dir = root / ".claude-plugin" / ".venv"
    venv_dir.mkdir(parents=True)
    expected = wrapper._expected_install_stamp(root, {})
    (venv_dir / ".mnemos-install-stamp").write_text(expected, encoding="utf-8")
    calls: list[str] = []

    def create(path: Path) -> None:
        _fake_create_venv(path)

    def install(python_path: Path, plugin: Path, env: dict[str, str]) -> None:
        calls.append(str(python_path))

    monkeypatch.setattr(wrapper, "_create_venv", create)
    monkeypatch.setattr(wrapper, "_install_plugin", install)

    result = wrapper._ensure_runtime(venv_dir, root, {})
    assert result.exists()
    assert calls == [str(result)]


def test_source_change_invalidates_install_stamp(monkeypatch, tmp_path: Path) -> None:
    wrapper = _load_wrapper()
    root = _plugin_root(tmp_path)
    venv_dir = root / ".claude-plugin" / ".venv"
    calls: list[str] = []

    monkeypatch.setattr(wrapper, "_create_venv", _fake_create_venv)
    monkeypatch.setattr(
        wrapper,
        "_install_plugin",
        lambda python_path, plugin, env: calls.append(str(python_path)),
    )

    wrapper._ensure_runtime(venv_dir, root, {})
    (root / "mnemos" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    wrapper._ensure_runtime(venv_dir, root, {})
    assert len(calls) == 2


def test_runtime_lock_serializes_installation_across_threads(monkeypatch, tmp_path: Path) -> None:
    wrapper = _load_wrapper()
    root = _plugin_root(tmp_path)
    venv_dir = root / ".claude-plugin" / ".venv"
    install_count = 0
    count_lock = threading.Lock()

    monkeypatch.setattr(wrapper, "_create_venv", _fake_create_venv)

    def install(python_path: Path, plugin: Path, env: dict[str, str]) -> None:
        nonlocal install_count
        with count_lock:
            install_count += 1
        time.sleep(0.15)

    monkeypatch.setattr(wrapper, "_install_plugin", install)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: wrapper._ensure_runtime(venv_dir, root, {}), range(2)))

    assert results == [results[0], results[0]]
    assert install_count == 1


def test_hook_wrapper_is_a_real_stdio_subprocess(tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_modules"
    module_path = fake_root / "mnemos" / "adapters"
    module_path.mkdir(parents=True)
    (fake_root / "mnemos" / "__init__.py").write_text("", encoding="utf-8")
    (fake_root / "mnemos" / "adapters" / "__init__.py").write_text("", encoding="utf-8")
    (module_path / "claude_code.py").write_text(
        "import sys\n"
        "payload = sys.stdin.read()\n"
        "sys.stdout.write('received:' + payload)\n"
        "raise SystemExit(23)\n",
        encoding="utf-8",
    )
    hook = Path(".claude-plugin/run_mnemos_hook.py").resolve()
    env = dict(os.environ)
    env.update(
        {
            "CLAUDE_PLUGIN_ROOT": str(Path(".").resolve()),
            "MNEMOS_PLUGIN_PYTHON": sys.executable,
            "PYTHONPATH": str(fake_root),
        }
    )
    completed = subprocess.run(
        [sys.executable, str(hook), "Stop"],
        input='{"hook_event_name":"Stop"}',
        text=True,
        capture_output=True,
        env=env,
        cwd=fake_root,
        check=False,
    )
    assert completed.returncode == 23
    assert completed.stdout == 'received:{"hook_event_name":"Stop"}'
    assert completed.stderr == ""


def _stub_runtime(tmp_path: Path, exit_code: int) -> Path:
    stub = tmp_path / "stub"
    (stub / "mnemos" / "adapters").mkdir(parents=True)
    (stub / "mnemos" / "__init__.py").write_text("", encoding="utf-8")
    (stub / "mnemos" / "adapters" / "__init__.py").write_text("", encoding="utf-8")
    (stub / "mnemos" / "adapters" / "claude_code.py").write_text(
        f"import sys\nsys.stdin.read()\nraise SystemExit({exit_code})\n", encoding="utf-8"
    )
    return stub


def _run_hook_command(
    event: str, plugin_root: Path, env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    """Run the exact hooks.json command the way a host shell would."""
    hooks = json.loads(Path("hooks/hooks.json").read_text(encoding="utf-8"))["hooks"]
    command = hooks[event][0]["hooks"][0]["command"]
    assert command.startswith("python ")
    # Claude Code substitutes the plugin root into the command text.
    command = f'"{sys.executable}"' + command[len("python") :]
    command = command.replace("${CLAUDE_PLUGIN_ROOT}", str(plugin_root))
    return subprocess.run(
        command,
        shell=True,
        input=json.dumps({"hook_event_name": event}),
        text=True,
        capture_output=True,
        env={**os.environ, "CLAUDE_PLUGIN_ROOT": str(plugin_root), **env},
        cwd=cwd,
        check=False,
    )


def test_a_runtime_exit_of_two_never_reaches_claude_as_a_block(tmp_path: Path) -> None:
    stub = _stub_runtime(tmp_path, 2)
    completed = _run_hook_command(
        "Stop",
        Path(".").resolve(),
        {"MNEMOS_PLUGIN_PYTHON": sys.executable, "PYTHONPATH": str(stub)},
        cwd=stub,
    )
    assert completed.returncode == 1, completed.stderr


def test_a_removed_plugin_directory_does_not_block_stop_or_prompts(tmp_path: Path) -> None:
    # A plugin update can delete the version directory a running session uses.
    for event in ("Stop", "UserPromptSubmit"):
        completed = _run_hook_command(
            event,
            tmp_path / "removed-version",
            {"MNEMOS_PLUGIN_PYTHON": sys.executable},
            cwd=tmp_path,
        )
        assert completed.returncode == 1, completed.stderr


def test_hooks_skip_until_the_runtime_is_installed(monkeypatch, tmp_path: Path, capsys) -> None:
    runtime = _load_wrapper()
    sys.modules["run_mnemos_mcp"] = runtime
    try:
        spec = spec_from_file_location(
            "mnemos_plugin_hook_runtime_tests", Path(".claude-plugin/run_mnemos_hook.py")
        )
        assert spec is not None and spec.loader is not None
        hook = module_from_spec(spec)
        spec.loader.exec_module(hook)
    finally:
        del sys.modules["run_mnemos_mcp"]
    root = _plugin_root(tmp_path)

    def must_not_install(*args: object, **kwargs: object) -> None:
        raise AssertionError("a hook must not build or install the runtime")

    monkeypatch.delenv("MNEMOS_PLUGIN_PYTHON", raising=False)
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(root))
    monkeypatch.setattr(runtime, "_create_venv", must_not_install)
    monkeypatch.setattr(runtime, "_install_plugin", must_not_install)
    monkeypatch.setattr(hook.subprocess, "run", must_not_install)
    monkeypatch.setattr(hook.sys, "argv", ["run_mnemos_hook.py", "UserPromptSubmit"])

    assert hook.main() == 0
    assert "reason=runtime_not_ready" in capsys.readouterr().err

    # Once the MCP server has installed and stamped the runtime, hooks use it.
    venv_dir = root / ".claude-plugin" / ".venv"
    _fake_create_venv(venv_dir)
    (venv_dir / ".mnemos-install-stamp").write_text(
        runtime._expected_install_stamp(root, {}), encoding="utf-8"
    )
    assert runtime._ready_runtime_python(root, {}) == str(runtime._venv_python(venv_dir))
    (root / "mnemos" / "module.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert runtime._ready_runtime_python(root, {}) is None


def test_a_failed_install_ends_the_mcp_launcher_with_a_message(monkeypatch, capsys) -> None:
    wrapper = _load_wrapper()

    def offline(root: Path, env: dict[str, str]) -> str:
        raise subprocess.CalledProcessError(1, ["pip", "install"])

    monkeypatch.setattr(wrapper, "_resolve_runtime_python", offline)

    assert wrapper.main() == 1
    err = capsys.readouterr().err
    assert "CalledProcessError" in err and "Traceback" not in err


def test_plugin_mcp_child_receives_claude_host(monkeypatch) -> None:
    wrapper = _load_wrapper()
    captured: dict[str, object] = {}
    monkeypatch.setenv("MNEMOS_HOST", "manual")
    monkeypatch.setattr(wrapper, "_resolve_runtime_python", lambda root, env: sys.executable)

    def fake_run(args, *, env, check):
        captured["args"] = args
        captured["env"] = env
        captured["check"] = check
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    assert wrapper.main() == 0
    assert captured["args"] == [sys.executable, "-m", "mnemos.mcp_server"]
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["MNEMOS_HOST"] == "claude-code"
    assert captured["check"] is False
