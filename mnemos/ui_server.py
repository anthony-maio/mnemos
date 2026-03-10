"""
mnemos/ui_server.py — Local browser UI for Mnemos onboarding and health.
"""

from __future__ import annotations

import json
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

from .control_plane import ControlPlaneService


@dataclass(slots=True)
class UiResponse:
    status: int
    content_type: str
    body: bytes


def _ui_assets_dir() -> Path:
    return Path(__file__).resolve().parent / "ui"


class MnemosUiRouter:
    def __init__(self, service: ControlPlaneService) -> None:
        self.service = service
        self.assets_dir = _ui_assets_dir()

    def _json_response(self, payload: Any, *, status: int = HTTPStatus.OK) -> UiResponse:
        return UiResponse(
            status=int(status),
            content_type="application/json; charset=utf-8",
            body=json.dumps(payload, indent=2).encode("utf-8"),
        )

    def _text_response(
        self,
        text: str,
        *,
        status: int = HTTPStatus.OK,
        content_type: str = "text/plain; charset=utf-8",
    ) -> UiResponse:
        return UiResponse(status=int(status), content_type=content_type, body=text.encode("utf-8"))

    def _asset_response(self, name: str, content_type: str) -> UiResponse:
        path = self.assets_dir / name
        if not path.exists():
            return self._text_response("Not found", status=HTTPStatus.NOT_FOUND)
        return UiResponse(
            status=HTTPStatus.OK,
            content_type=content_type,
            body=path.read_bytes(),
        )

    def handle(self, method: str, raw_path: str, body: bytes | None) -> UiResponse:
        parsed = urlparse(raw_path)
        path = parsed.path

        if method == "GET" and path == "/":
            return self._asset_response("index.html", "text/html; charset=utf-8")
        if method == "GET" and path == "/app.js":
            return self._asset_response("app.js", "application/javascript; charset=utf-8")
        if method == "GET" and path == "/styles.css":
            return self._asset_response("styles.css", "text/css; charset=utf-8")

        if method == "GET" and path == "/api/settings":
            return self._json_response(self.service.get_settings_view())
        if method == "GET" and path == "/api/health":
            return self._json_response(self.service.health_report())
        if method == "GET" and path == "/api/memory":
            return self._json_response(self.service.get_memory_snapshot())

        payload: dict[str, Any] = {}
        if body:
            payload = json.loads(body.decode("utf-8"))

        if method == "POST" and path == "/api/settings/global":
            return self._json_response(self.service.save_settings(payload, scope="global"))
        if method == "POST" and path == "/api/settings/project":
            return self._json_response(self.service.save_settings(payload, scope="project"))
        if method == "POST" and path == "/api/import":
            return self._json_response(self.service.import_existing_setup())
        if method == "POST" and path == "/api/smoke":
            return self._json_response(self.service.run_smoke_tests())

        integration_prefix = "/api/integrations/"
        if method == "POST" and path.startswith(integration_prefix):
            suffix = path[len(integration_prefix) :]
            parts = [part for part in suffix.split("/") if part]
            if len(parts) == 2 and parts[1] in {"preview", "apply"}:
                host_name = parts[0]
                if host_name not in {"claude-code", "cursor", "codex"}:
                    return self._text_response("Unknown host", status=HTTPStatus.NOT_FOUND)
                host = cast(Literal["claude-code", "cursor", "codex"], host_name)
                if parts[1] == "preview":
                    return self._json_response(self.service.preview_integration(host))
                return self._json_response(self.service.apply_integration(host))

        return self._text_response("Not found", status=HTTPStatus.NOT_FOUND)


class _MnemosUiHandler(BaseHTTPRequestHandler):
    router: MnemosUiRouter

    def do_GET(self) -> None:  # noqa: N802
        response = self.router.handle("GET", self.path, None)
        self._write_response(response)

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else None
        response = self.router.handle("POST", self.path, body)
        self._write_response(response)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        _ = format, args

    def _write_response(self, response: UiResponse) -> None:
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        self.wfile.write(response.body)


def run_ui_server(
    *,
    service: ControlPlaneService,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    router = MnemosUiRouter(service)

    class Handler(_MnemosUiHandler):
        pass

    Handler.router = router

    server = ThreadingHTTPServer((host, port), Handler)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    if open_browser:
        webbrowser.open(url)
    print(f"Mnemos UI running at {url}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
