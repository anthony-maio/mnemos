"""The experimental native profile must preserve the guarded fallback boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mnemos import hook_client
from tests.test_native_hook_launcher import BINARY, http, isolated, reply, save_descriptor, server

pytestmark = pytest.mark.skipif(not BINARY, reason="Experimental native launcher is not configured")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(root, env, *, host="codex"):
    launcher = root / "guarded.py"
    launcher.write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "assert sys.argv[1:3]==['hook',os.environ['MNEMOS_HOST']]\n"
        "assert 'MNEMOS_LLM_PROVIDER' not in os.environ\n"
        "assert 'MNEMOS_API_KEY' not in os.environ\n"
        "Path(__file__).with_suffix('.input').write_bytes(sys.stdin.buffer.read())\n"
        "print(json.dumps({'guarded_fallback':True,'event':sys.argv[3]}))\n"
    )
    values = {"MNEMOS_HOST": host, "MNEMOS_USER_PROFILE_ID": "synthetic-profile"}
    value = {
        "schema_version": 1,
        "host": host,
        "python": sys.executable,
        "fallback_launcher": str(launcher),
        "fallback_launcher_sha256": digest(launcher),
        "worker_descriptor": env[hook_client.WORKER_DESCRIPTOR_ENV],
        "environment": values,
    }
    path = root / "native-profile.json"
    path.write_text(json.dumps(value))
    effective = {key: value for key, value in env.items() if not key.upper().startswith("MNEMOS_")}
    effective.update(values)
    effective[hook_client.WORKER_DESCRIPTOR_ENV] = value["worker_descriptor"]
    return path, value, effective


def invoke_profile(root, env, path, *, expected=None, host="codex", raw=None):
    return subprocess.run(
        [
            BINARY,
            host,
            "Stop",
            "--profile",
            str(path),
            "--profile-sha256",
            expected or digest(path),
        ],
        cwd=root,
        env=env,
        input=raw or b'{"last_assistant_message":"synthetic original input"}',
        capture_output=True,
        timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


@pytest.mark.parametrize("host", ["codex", "claude-code"])
def test_profile_resets_inherited_selectors_before_worker_dispatch(isolated, host):
    root, env = isolated
    path, _, effective = profile(root, env, host=host)
    env.update(MNEMOS_LLM_PROVIDER="remote", MNEMOS_API_KEY="synthetic-secret")
    with server(lambda request: http(reply(request))) as (port, requests):
        save_descriptor(effective, port, host=host)
        result = invoke_profile(root, env, path, host=host)
    assert result.returncode == 0 and not result.stderr
    assert json.loads(result.stdout) == {"continue": True}
    assert len(requests) == 1
    assert requests[0][0]["payload"]["environment_sha256"] == hook_client.environment_fingerprint(
        effective
    )
    assert not (root / "guarded.input").exists()
    assert not Path(env["FALLBACK_MARKER"]).exists()


@pytest.mark.parametrize("host", ["codex", "claude-code"])
def test_profile_missing_worker_uses_guarded_launcher_with_original_input(isolated, host):
    root, env = isolated
    path, _, _ = profile(root, env, host=host)
    env.update(MNEMOS_LLM_PROVIDER="remote", MNEMOS_API_KEY="synthetic-secret")
    raw = b'{"last_assistant_message":"caf\\u00e9 original input"}'
    result = invoke_profile(root, env, path, host=host, raw=raw)
    assert result.returncode == 0 and not result.stderr
    assert json.loads(result.stdout) == {"guarded_fallback": True, "event": "Stop"}
    assert (root / "guarded.input").read_bytes() == raw
    assert not Path(env["FALLBACK_MARKER"]).exists()


def test_bound_worker_decline_uses_guarded_launcher(isolated):
    root, env = isolated
    path, _, effective = profile(root, env)
    with server(lambda request: http(reply(request, status="declined", dispatched=False))) as (
        port,
        requests,
    ):
        save_descriptor(effective, port)
        result = invoke_profile(root, env, path)
    assert len(requests) == 1 and result.returncode == 0
    assert json.loads(result.stdout)["guarded_fallback"] is True
    assert not Path(env["FALLBACK_MARKER"]).exists()


def test_lost_acknowledgement_never_runs_either_fallback(isolated):
    root, env = isolated
    path, _, effective = profile(root, env)
    with server(lambda request: None) as (port, requests):
        save_descriptor(effective, port)
        result = invoke_profile(root, env, path)
    assert len(requests) == 1 and result.returncode == 0
    assert json.loads(result.stdout) == {"continue": True}
    assert b"status=error" in result.stderr
    assert not (root / "guarded.input").exists()
    assert not Path(env["FALLBACK_MARKER"]).exists()


@pytest.mark.parametrize(
    "change",
    [
        "profile_hash",
        "launcher_hash",
        "launcher_changed",
        "host",
        "schema",
        "python_relative",
        "launcher_relative",
        "descriptor_relative",
        "foreign_env",
        "descriptor_env",
        "null_env",
        "nul_env",
    ],
)
def test_invalid_profile_dispatches_nothing_and_never_runs_fallback(isolated, change):
    root, env = isolated
    path, value, effective = profile(root, env)
    expected = None
    if change == "profile_hash":
        expected = "0" * 64
    elif change == "launcher_hash":
        value["fallback_launcher_sha256"] = "0" * 64
    elif change == "launcher_changed":
        Path(value["fallback_launcher"]).write_text("raise RuntimeError('must not execute')")
    elif change == "host":
        value["host"] = "claude-code"
    elif change == "schema":
        value["schema_version"] = True
    elif change == "python_relative":
        value["python"] = "python.exe"
    elif change == "launcher_relative":
        value["fallback_launcher"] = "guarded.py"
    elif change == "descriptor_relative":
        value["worker_descriptor"] = "worker.json"
    elif change == "foreign_env":
        value["environment"]["PATH"] = "must not replace"
    elif change == "descriptor_env":
        value["environment"][hook_client.WORKER_DESCRIPTOR_ENV] = "foreign"
    elif change == "null_env":
        value["environment"]["MNEMOS_HOST"] = None
    elif change == "nul_env":
        value["environment"]["MNEMOS_HOST"] = "codex\0foreign"
    path.write_text(json.dumps(value))
    with server(lambda request: http(reply(request))) as (port, requests):
        save_descriptor(effective, port)
        result = invoke_profile(root, env, path, expected=expected)
    assert result.returncode == 0 and not requests
    assert json.loads(result.stdout) == {"continue": True}
    assert b"status=error" in result.stderr
    assert not (root / "guarded.input").exists()
    assert not Path(env["FALLBACK_MARKER"]).exists()


def test_launcher_change_during_decline_cannot_execute_replacement(isolated):
    root, env = isolated
    path, value, effective = profile(root, env)

    def decline(request):
        Path(value["fallback_launcher"]).write_text("raise RuntimeError('must not execute')")
        return http(reply(request, status="declined", dispatched=False))

    with server(decline) as (port, requests):
        save_descriptor(effective, port)
        result = invoke_profile(root, env, path)
    assert result.returncode == 0 and len(requests) == 1 and not result.stdout
    assert b"status=error" in result.stderr and b"must not execute" not in result.stderr
    assert not Path(env["FALLBACK_MARKER"]).exists()
