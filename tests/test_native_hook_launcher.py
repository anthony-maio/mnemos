"""Opt-in subprocess checks for the development-only compiled hook launcher."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

import pytest

from mnemos import hook_client
from mnemos.client import Outbox
from mnemos.hook_worker import HookWorker
from tests.test_hook_worker import configured, payload  # noqa: F401

BINARY = os.environ.get("MNEMOS_NATIVE_HOOK_EXE")
pytestmark = pytest.mark.skipif(not BINARY, reason="Experimental native launcher is not configured")


def invoke(root, env, value, *, host="codex", event="Stop"):
    raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
    return subprocess.run(
        [BINARY, host, event, "--python", sys.executable],
        cwd=root,
        env=env,
        input=raw,
        capture_output=True,
        timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


@pytest.fixture
def isolated(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    package = tmp_path / "fallback" / "mnemos"
    (package / "adapters").mkdir(parents=True)
    for path in (package / "__init__.py", package / "adapters/__init__.py"):
        path.write_text("")
    stub = (
        "import json,os,sys\nfrom pathlib import Path\n"
        "Path(os.environ['FALLBACK_MARKER']).write_bytes(sys.stdin.buffer.read())\n"
        "print(json.dumps({'fallback':True,'event':sys.argv[1]}))\n"
    )
    for host in ("codex", "claude_code"):
        (package / "adapters" / f"{host}.py").write_text(stub)
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("MNEMOS_")}
    env.update(
        PYTHONPATH=str(package.parent),
        FALLBACK_MARKER=str(tmp_path / "fallback-input.json"),
        MNEMOS_HOOK_WORKER_DESCRIPTOR=str(tmp_path / "worker.json"),
        NATIVE_TEST_UNICODE="caf\u00e9 \U0001f600\ncontrol\tvalue",
    )
    # Match the Windows child process's os.environ normalization.
    if sys.platform == "win32":
        env = {key.upper(): value for key, value in env.items()}
    return root, env


def save_descriptor(env, port, *, host="codex", changes=None):
    value = dict(
        hook_protocol=1,
        adapter_host=host,
        host="127.0.0.1",
        port=port,
        token="private-synthetic-token",
        environment_sha256=hook_client.environment_fingerprint(env),
    )
    value.update(changes or {})
    Path(env[hook_client.WORKER_DESCRIPTOR_ENV]).write_text(json.dumps(value))


def reply(request, **changes):
    envelope = request["payload"]
    value = {
        key: envelope[key] for key in ("hook_protocol", "request_id", "host", "environment_sha256")
    }
    value.update(status="handled", dispatched=True, adapter_status="ok", output={"continue": True})
    value.update(changes)
    return json.dumps(value, ensure_ascii=False).encode()


@contextmanager
def server(callback):
    requests = []
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                value = json.loads(raw)
                requests.append((value, dict(self.headers)))
                assert self.path == "/v1/dispatch"
                assert self.headers["Authorization"] == "Bearer private-synthetic-token"
                response = callback(value)
                if response is not None:
                    self.connection.sendall(response)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                errors.append(type(exc).__name__)
            finally:
                self.close_connection = True

        def log_message(self, *args):
            pass

    listener = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=listener.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield listener.server_port, requests
    finally:
        listener.shutdown()
        listener.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive() and not errors


def http(body, *, status="200 OK", headers=None, length=None):
    header = f"HTTP/1.1 {status}\r\nContent-Length: {len(body) if length is None else length}\r\n"
    return (header + (headers or "") + "\r\n").encode() + body


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"PATH": "first", "Path": "second", "MNEMOS_HOOK_WORKER_DESCRIPTOR": "excluded"},
        {'quote"': "slash\\\n\t\r\b\f\x00\x1f\x7f", "\u00e9": "\U0001f600"},
        {"z": "last", "\ue000": "bmp", "\U00010000": "supplementary"},
    ],
)
def test_environment_fingerprint_matches_python(values):
    result = subprocess.run(
        [BINARY, "--fingerprint"], input=json.dumps(values).encode(), capture_output=True, timeout=5
    )
    assert result.returncode == 0 and not result.stderr
    assert result.stdout.decode().strip() == hook_client.environment_fingerprint(values)


@pytest.mark.parametrize("host", ["codex", "claude-code"])
def test_bound_response_preserves_input_and_host_output(isolated, host):
    root, env = isolated
    value = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "caf\u00e9 \U0001f600",
        "number": 2**90 + 1,
    }
    expected = {"context": "source \u00e9", "continue": True}
    with server(lambda request: http(reply(request, output=expected))) as (port, requests):
        save_descriptor(env, port, host=host)
        result = invoke(root, env, value, host=host)
    assert result.returncode == 0 and not result.stderr
    assert json.loads(result.stdout) == expected
    assert len(requests) == 1
    assert requests[0][0]["payload"]["payload"] == value
    assert requests[0][0]["payload"]["event"] == "UserPromptSubmit"
    assert Path(requests[0][0]["payload"]["cwd"]).resolve() == root.resolve()
    assert not Path(env["FALLBACK_MARKER"]).exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"host": "192.0.2.1"},
        {"hook_protocol": True},
        {"hook_protocol": 1.0},
        {"adapter_host": "claude-code"},
        {"port": True},
        {"port": 0},
        {"port": 65536},
        {"environment_sha256": "foreign"},
        {"token": "bad\r\nheader"},
        {"token": ""},
    ],
)
def test_invalid_descriptor_falls_back_without_sending(isolated, changes):
    root, env = isolated
    value = {"last_assistant_message": "private synthetic content"}
    with server(lambda request: http(reply(request))) as (port, requests):
        save_descriptor(env, port, changes=changes)
        result = invoke(root, env, value)
    assert result.returncode == 0 and not requests
    assert json.loads(result.stdout) == {"fallback": True, "event": "Stop"}
    assert json.loads(Path(env["FALLBACK_MARKER"]).read_bytes()) == value


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": "foreign"},
        {"host": "claude-code"},
        {"environment_sha256": "foreign"},
        {"hook_protocol": True},
        {"hook_protocol": 1.0},
        {"status": "failed"},
        {"dispatched": False},
        {"dispatched": 1},
        {"output": []},
        {"adapter_status": 1},
    ],
)
def test_unbound_or_invalid_reply_never_falls_back(isolated, changes):
    root, env = isolated
    with server(lambda request: http(reply(request, **changes))) as (port, requests):
        save_descriptor(env, port)
        result = invoke(root, env, {"last_assistant_message": "must not print this"})
    assert len(requests) == 1 and result.returncode == 0
    assert json.loads(result.stdout) == {"continue": True}
    assert b"status=error" in result.stderr and b"must not print" not in result.stderr
    assert not Path(env["FALLBACK_MARKER"]).exists()


@pytest.mark.parametrize(
    "failure",
    [
        "lost",
        "redirect",
        "auth",
        "server",
        "truncated",
        "large",
        "missing_length",
        "duplicate_length",
        "negative_length",
        "chunked",
        "invalid_json",
        "large_header",
    ],
)
def test_wire_failures_do_not_retry_or_fallback(isolated, failure):
    root, env = isolated

    def response(request):
        body = reply(request)
        return {
            "lost": None,
            "redirect": http(body, status="302 Found", headers="Location: http://127.0.0.1:1/\r\n"),
            "auth": http(body, status="401 Unauthorized", headers="WWW-Authenticate: Basic\r\n"),
            "server": http(body, status="503 Unavailable"),
            "truncated": http(body[:-1], length=len(body)),
            "large": http(b"", length=1_048_577),
            "missing_length": b"HTTP/1.1 200 OK\r\n\r\n" + body,
            "duplicate_length": http(body, headers=f"Content-Length: {len(body)}\r\n"),
            "negative_length": http(body, length=-1),
            "chunked": http(body, headers="Transfer-Encoding: chunked\r\n"),
            "invalid_json": http(b"not json"),
            "large_header": http(body, headers="X-Long: " + "x" * 16384 + "\r\n"),
        }[failure]

    with server(response) as (port, requests):
        save_descriptor(env, port)
        result = invoke(root, env, {})
    assert result.returncode == 0 and len(requests) == 1
    assert json.loads(result.stdout) == {"continue": True}
    assert b"status=error" in result.stderr
    assert not Path(env["FALLBACK_MARKER"]).exists()


def test_explicit_bound_decline_allows_original_adapter(isolated):
    root, env = isolated
    with server(lambda request: http(reply(request, status="declined", dispatched=False))) as (
        port,
        requests,
    ):
        save_descriptor(env, port)
        result = invoke(root, env, {"prompt": "preserve input"})
    assert len(requests) == 1 and result.returncode == 0
    assert json.loads(result.stdout)["fallback"] is True
    assert json.loads(Path(env["FALLBACK_MARKER"]).read_bytes()) == {"prompt": "preserve input"}


def test_connection_refused_allows_original_adapter(isolated):
    root, env = isolated
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    save_descriptor(env, port)
    result = invoke(root, env, {})
    assert result.returncode == 0 and json.loads(result.stdout)["fallback"] is True


@pytest.mark.parametrize("raw", [None, b"not json", b"x" * 65_537])
def test_missing_or_invalid_descriptor_allows_original_adapter(isolated, raw):
    root, env = isolated
    if raw is not None:
        Path(env[hook_client.WORKER_DESCRIPTOR_ENV]).write_bytes(raw)
    result = invoke(root, env, {})
    assert result.returncode == 0 and json.loads(result.stdout)["fallback"] is True


def test_oversized_frame_falls_back_with_complete_original_input(isolated):
    root, env = isolated
    value = {"prompt": "x" * 1_048_576}
    with server(lambda request: http(reply(request))) as (port, requests):
        save_descriptor(env, port)
        result = invoke(root, env, value)
    assert result.returncode == 0 and not requests
    assert json.loads(result.stdout)["fallback"] is True
    assert json.loads(Path(env["FALLBACK_MARKER"]).read_bytes()) == value


@pytest.mark.parametrize("raw", [b"not json private text", b"\xffprivate text"])
def test_invalid_input_is_sanitized_without_dispatch(isolated, raw):
    root, env = isolated
    result = invoke(root, env, raw)
    assert result.returncode == 0 and json.loads(result.stdout) == {"continue": True}
    assert b"status=error" in result.stderr and b"private text" not in result.stderr
    assert not Path(env["FALLBACK_MARKER"]).exists()


@pytest.mark.parametrize("raw", [b"[]", b"null", b"123"])
def test_nonobject_input_dispatches_nothing(isolated, raw):
    root, env = isolated
    result = invoke(root, env, raw)
    assert result.returncode == 0 and not result.stdout and not result.stderr
    assert not Path(env["FALLBACK_MARKER"]).exists()


def test_real_worker_capture_survives_restart(configured, monkeypatch):
    root = configured
    monkeypatch.setenv(hook_client.WORKER_DESCRIPTOR_ENV, str(root / "worker.json"))
    worker = HookWorker("codex", descriptor_path=root / "worker.json", cwd=root)
    worker.start()
    try:
        result = invoke(root, dict(os.environ), payload(root))
        assert result.returncode == 0 and json.loads(result.stdout) == {"continue": True}
        assert b"status=degraded" in result.stderr
        queued = Outbox(root / "outbox.json").pending()
        assert len(queued) == 1
        event_id = queued[0]["payload"]["event_id"]
    finally:
        worker.stop()
    assert not (root / "worker.json").exists()
    worker = HookWorker("codex", descriptor_path=root / "worker.json", cwd=root)
    worker.start()
    try:
        # The repeated native event has the same identity and stays one queued
        # capture. The service is intentionally unavailable in this check.
        result = invoke(root, dict(os.environ), payload(root))
        assert result.returncode == 0
        queued = Outbox(root / "outbox.json").pending()
        assert len(queued) == 1 and queued[0]["payload"]["event_id"] == event_id
    finally:
        worker.stop()
