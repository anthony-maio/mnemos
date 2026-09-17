"""Real CLI subprocesses using an isolated, authenticated service."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mnemos.service import MnemosService
from tests.test_memory_service import service


def test_cli_memory_lifecycle_uses_one_service(tmp_path):
    memory, context = service(tmp_path)
    registry_path = tmp_path / "registry.json"
    memory.registry.save(registry_path)
    runtime = MnemosService(dispatcher=memory, descriptor_path=tmp_path / "runtime.json")
    runtime.start()
    env = {
        **os.environ,
        "MNEMOS_RUNTIME_DESCRIPTOR": str(runtime.descriptor_path),
        "MNEMOS_RUNTIME_DIR": str(tmp_path),
        "MNEMOS_SCOPE_REGISTRY": str(registry_path),
        "MNEMOS_USER_PROFILE_ID": context.user_profile_id,
        "MNEMOS_PROJECT_ROOT": str(tmp_path),
        "MNEMOS_SESSION_ID": "cli-lifecycle",
    }

    def cli(*args):
        result = subprocess.run(
            [sys.executable, "-m", "mnemos.cli", *args],
            env=env,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    try:
        assert cli("store", "Use SQLite for the local cache")["status"] == "committed"
        cli("consolidate")
        facts = cli("list")["items"]
        assert len(facts) == 1
        fact = facts[0]
        assert cli("inspect", fact["id"])["evidence"]
        cli(
            "correct",
            fact["id"],
            "Use SQLite with WAL for the local cache",
            "--expected-version",
            "1",
        )
        cli("consolidate")
        corrected = cli("list")["items"]
        assert len(corrected) == 1
        assert corrected[0]["id"] == fact["id"]
        assert corrected[0]["version"] == 2
        assert cli("forget", fact["id"], "--expected-version", "2")["forgotten"] is True
        assert cli("list")["items"] == []
    finally:
        runtime.stop()


def test_cli_stops_isolated_service_and_releases_offline_lock(tmp_path):
    from mnemos.service import _ProcessLock, read_runtime_descriptor

    descriptor = tmp_path / "runtime.json"
    env = {
        **os.environ,
        "MNEMOS_RUNTIME_DESCRIPTOR": str(descriptor),
        "MNEMOS_RUNTIME_DIR": str(tmp_path),
        "MNEMOS_CONFIG_PATH": str(tmp_path / "settings.toml"),
        "MNEMOS_SCOPE_REGISTRY": str(tmp_path / "scopes.json"),
        "MNEMOS_LLM_PROVIDER": "ollama",
        "MNEMOS_LLM_MODEL": "unused-shutdown-fixture",
        "MNEMOS_OLLAMA_URL": "http://127.0.0.1:1",
        "MNEMOS_EMBEDDING_PROVIDER": "simple",
        "MNEMOS_STORE_TYPE": "sqlite",
        "MNEMOS_SQLITE_PATH": str(tmp_path / "memory.db"),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "mnemos.service"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + 15
        while read_runtime_descriptor(descriptor) is None and time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(process.communicate()[1])
            time.sleep(0.05)
        assert read_runtime_descriptor(descriptor) is not None
        result = subprocess.run(
            [sys.executable, "-m", "mnemos.cli", "service-stop"],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["status"] == "stopped"
        assert process.wait(timeout=5) == 0
        lock = _ProcessLock(descriptor.with_suffix(".lock"))
        assert lock.acquire()
        lock.release()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_cli_degraded_result_has_failure_exit_status(monkeypatch, capsys):
    import pytest
    from mnemos.cli import main

    async def failed_command(args):
        print(json.dumps({"status": "degraded", "committed": False}))
        return {"status": "degraded", "committed": False}

    monkeypatch.setattr("mnemos.managed_cli.run_command", failed_command)
    monkeypatch.setattr(sys, "argv", ["mnemos", "stats"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["committed"] is False


def test_hook_dry_run_does_not_invoke_adapter(monkeypatch, capsys):
    from mnemos.cli import main

    def unexpected(*args, **kwargs):
        raise AssertionError("A dry run must not construct a service client")

    monkeypatch.setattr("mnemos.adapters.claude_code.ClaudeCodeAdapter", unexpected)
    monkeypatch.setattr(sys, "argv", ["mnemos", "autostore-hook", "UserPromptSubmit", "--dry-run"])
    main()
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run"


def test_legacy_copy_rejects_managed_store_without_losing_evidence(tmp_path, monkeypatch, capsys):
    import pytest
    from mnemos.cli import main
    from mnemos.utils.storage import SQLiteStore

    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    store = SQLiteStore(str(source))
    store.close()
    monkeypatch.setenv("MNEMOS_RUNTIME_DESCRIPTOR", str(tmp_path / "runtime.json"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mnemos",
            "repair-db",
            "--source-sqlite-path",
            str(source),
            "--target-sqlite-path",
            str(target),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "backup/restore" in capsys.readouterr().err
    assert not target.exists()
