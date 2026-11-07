"""The native hook transport keeps one bounded, authenticated request."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading

import pytest

from mnemos import _winhttp_hook as native
from mnemos import hook_client

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows native transport")


@contextmanager
def responding(reply):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append({"path": self.path, "headers": dict(self.headers), "body": data})
            status, headers, body = reply(data)
            self.close_connection = True
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            if "Content-Length" not in headers:
                self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture
def native_supported():
    with responding(lambda data: (200, {}, b"{}")) as (port, _):
        try:
            assert native.post(port, "synthetic-token", b"{}", timeout=2, limit=1024) == b"{}"
        except native.WinHTTPUnavailable:
            pytest.skip("Required WinHTTP controls are unavailable on this Windows version")


def test_native_roundtrip_preserves_utf8_and_ignores_proxy(native_supported, monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    wire = json.dumps({"content": "caf\u00e9 \u65e5\u672c"}, ensure_ascii=False).encode()
    with responding(lambda data: (200, {}, data)) as (port, requests):
        assert native.post(port, "synthetic-token", wire, timeout=2, limit=1024) == wire
    assert len(requests) == 1 and requests[0]["body"] == wire
    assert requests[0]["path"] == "/v1/dispatch"
    assert requests[0]["headers"]["Authorization"] == "Bearer synthetic-token"


def test_failed_initial_connection_allows_fallback(native_supported):
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        port = unavailable.getsockname()[1]
        assert native.post(port, "synthetic-token", b"{}", timeout=0.5, limit=1024) is None


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 407, 408, 503])
def test_responses_never_redirect_authenticate_or_retry(native_supported, status):
    with responding(lambda data: (200, {}, b"unexpected")) as (destination, redirected):
        headers = {
            "Location": f"http://127.0.0.1:{destination}/unexpected",
            "WWW-Authenticate": "Negotiate",
            "Proxy-Authenticate": "Negotiate",
            "Set-Cookie": "synthetic=value",
        }
        with responding(lambda data: (status, headers, b"failure")) as (port, requests):
            with pytest.raises(ValueError, match="HTTP status"):
                native.post(port, "synthetic-token", b"{}", timeout=2, limit=1024)
        assert len(requests) == 1 and redirected == []


def test_short_reply_after_request_does_not_become_safe_fallback(native_supported):
    with responding(lambda data: (200, {"Content-Length": "100"}, b"{}")) as (port, requests):
        with pytest.raises(ValueError, match="Incomplete"):
            native.post(port, "synthetic-token", b"{}", timeout=2, limit=1024)
    assert len(requests) == 1


@pytest.mark.parametrize("size", [31, 32, 33])
def test_response_bound(native_supported, size):
    with responding(lambda data: (200, {}, b"x" * size)) as (port, requests):
        if size <= 32:
            assert native.post(port, "synthetic-token", b"{}", timeout=2, limit=32) == b"x" * size
        else:
            with pytest.raises(ValueError, match="exceeds"):
                native.post(port, "synthetic-token", b"{}", timeout=2, limit=32)
    assert len(requests) == 1


def test_native_command_does_not_import_http_parser(native_supported, tmp_path):
    def reply(data):
        request = json.loads(data)["payload"]
        response = {
            key: request[key]
            for key in ("hook_protocol", "request_id", "host", "environment_sha256")
        }
        response.update(
            status="handled", dispatched=True, adapter_status="ok", output={"continue": True}
        )
        return 200, {}, json.dumps(response).encode()

    with responding(reply) as (port, requests):
        descriptor = tmp_path / "worker.json"
        descriptor.write_text(
            json.dumps(
                {
                    "host": "127.0.0.1",
                    "port": port,
                    "token": "synthetic-token",
                    "hook_protocol": 1,
                    "adapter_host": "codex",
                    "environment_sha256": hook_client.environment_fingerprint(),
                }
            )
        )
        script = """
import importlib.abc,sys
class BlockParser(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='http.client' or fullname=='email' or fullname.startswith('email.'):
            raise AssertionError('Native command imported '+fullname)
sys.path.insert(0,sys.argv[1]);sys.meta_path.insert(0,BlockParser())
from mnemos.hook_client import request_hook
result=request_hook('codex','Stop',{},cwd=sys.argv[2],descriptor_path=sys.argv[3])
assert result['output']=={'continue':True}
assert 'mnemos._winhttp_hook' in sys.modules and 'http.client' not in sys.modules
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                script,
                str(Path(__file__).resolve().parents[1]),
                str(tmp_path),
                str(descriptor),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        assert completed.returncode == 0 and not completed.stderr, completed.stderr
    assert len(requests) == 1


class FaultAPI:
    def __init__(self, failure):
        self.failure = failure
        self.closed = []
        self.sent = 0
        self.options = []

    def WinHttpOpen(self, *args):
        return 0 if self.failure == "open" else 101

    def WinHttpSetTimeouts(self, *args):
        return self.failure != "timeouts"

    def WinHttpSetOption(self, handle, option, pointer, size):
        self.options.append((handle, option, ctypes.string_at(pointer, size)))
        return option != self.failure

    def WinHttpConnect(self, *args):
        return 0 if self.failure == "connect" else 102

    def WinHttpOpenRequest(self, *args):
        return 0 if self.failure == "request" else 103

    def WinHttpSendRequest(self, *args):
        self.sent += 1
        return self.failure != "send"

    def WinHttpReceiveResponse(self, *args):
        return False

    def WinHttpCloseHandle(self, handle):
        self.closed.append(handle)
        return True


@pytest.mark.parametrize(
    "failure,closed",
    [
        ("open", []),
        ("timeouts", [101]),
        (4, [101]),
        (162, [101]),
        (195, [101]),
        ("connect", [101]),
        ("request", [102, 101]),
        (63, [103, 102, 101]),
        (91, [103, 102, 101]),
    ],
)
def test_preparation_failure_never_sends_and_closes_handles(monkeypatch, failure, closed):
    api = FaultAPI(failure)
    monkeypatch.setattr(native, "_load_api", lambda: api)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 123)
    with pytest.raises(native.WinHTTPUnavailable):
        native.post(12345, "synthetic-token", b"{}", timeout=2, limit=1024)
    assert api.sent == 0 and api.closed == closed


@pytest.mark.parametrize(
    "failure,error,unconnected",
    [
        ("send", 12029, True),
        ("send", 12030, False),
        ("receive", 12029, False),
        ("receive", 12032, False),
    ],
)
def test_only_initial_cannot_connect_allows_retry(monkeypatch, failure, error, unconnected):
    api = FaultAPI(failure)
    monkeypatch.setattr(native, "_load_api", lambda: api)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: error)
    if unconnected:
        assert native.post(12345, "synthetic-token", b"{}", timeout=2, limit=1024) is None
    else:
        with pytest.raises(OSError):
            native.post(12345, "synthetic-token", b"{}", timeout=2, limit=1024)
    assert api.sent == 1 and api.closed == [103, 102, 101]
    options = {number: data for _, number, data in api.options}
    assert options[4] == b"\x01\0\0\0" and options[162] == b"\0" * 8
    assert options[195] == b"\x01\0\0\0" and options[63] == b"\x0f\0\0\0"


@pytest.mark.parametrize("error", [native.WinHTTPUnavailable("setup"), OSError("after send")])
def test_alternate_transport_only_after_native_setup_failure(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    calls = []
    monkeypatch.setattr(native, "post", fail)
    monkeypatch.setattr(hook_client, "_post_http", lambda *args: calls.append(args) or b"{}")
    if isinstance(error, native.WinHTTPUnavailable):
        assert hook_client._post(12345, "synthetic-token", b"{}", 2) == b"{}"
        assert len(calls) == 1
    else:
        with pytest.raises(OSError):
            hook_client._post(12345, "synthetic-token", b"{}", 2)
        assert calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"token": ""},
        {"token": "safe\r\nInjected: value"},
        {"token": "synthetic-\u00e9"},
        {"port": True},
        {"port": 65536},
        {"timeout": float("nan")},
        {"timeout": 0},
        {"limit": 0},
        {"limit": 2**31},
        {"wire": b"x" * 1025},
    ],
)
def test_invalid_input_is_rejected_before_loading_native_api(monkeypatch, change):
    def forbidden():
        pytest.fail("Invalid input must not initialize a transport")

    monkeypatch.setattr(native, "_load_api", forbidden)
    arguments = {
        "port": 12345,
        "token": "synthetic-token",
        "wire": b"{}",
        "timeout": 2,
        "limit": 1024,
    }
    arguments.update(change)
    with pytest.raises(ValueError, match="Invalid loopback"):
        native.post(**arguments)
