"""Focused tests for the Claude plugin runtime bootstrap."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
