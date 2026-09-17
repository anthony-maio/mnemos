"""
mnemos/ui_server.py — Local browser UI for Mnemos onboarding and health.
"""

from __future__ import annotations

import json
import secrets
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal, Mapping, cast
from urllib.parse import parse_qs, urlparse

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
        self.csrf_token = secrets.token_urlsafe(24)

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

    def _csrf_allowed(self, headers: Mapping[str, str] | None) -> bool:
        # Direct router calls are an in-process test/API surface. HTTP requests
        # must prove same-origin and carry the per-server mutation token.
        if headers is None:
            return True
        origin = headers.get("Origin")
        host = headers.get("Host")
        if origin and urlparse(origin).netloc != host:
            return False
        return headers.get("X-Mnemos-CSRF") == self.csrf_token

    def handle(
        self,
        method: str,
        raw_path: str,
        body: bytes | None,
        headers: Mapping[str, str] | None = None,
    ) -> UiResponse:
        parsed = urlparse(raw_path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if headers is not None:
            host = urlparse("//" + headers.get("Host", "")).hostname
            if host not in {"127.0.0.1", "localhost", "::1"}:
                return self._json_response(
                    {"error": "Loopback Host header required."}, status=HTTPStatus.FORBIDDEN
                )

        if method == "GET" and path == "/":
            return self._asset_response("index.html", "text/html; charset=utf-8")
        if method == "GET" and path == "/app.js":
            return self._asset_response("app.js", "application/javascript; charset=utf-8")
        if method == "GET" and path == "/styles.css":
            return self._asset_response("styles.css", "text/css; charset=utf-8")
        if method == "GET" and path == "/api/csrf":
            return self._json_response({"csrf_token": self.csrf_token})

        if method == "GET" and path == "/api/dashboard":
            return self._json_response(self.service.get_dashboard_view())
        if method == "GET" and path == "/api/settings":
            return self._json_response(self.service.get_settings_view())
        if method == "GET" and path == "/api/health":
            return self._json_response(self.service.health_report())
        if method == "GET" and path == "/api/memory":
            return self._json_response(self.service.get_memory_snapshot())
        if method == "GET" and path.startswith("/api/memory/"):
            chunk_id = path.removeprefix("/api/memory/")
            try:
                return self._json_response(
                    self.service.get_memory_detail(
                        chunk_id,
                        query=query.get("query", [None])[0],
                        current_scope=query.get("current_scope", ["project"])[0],
                        scope_id=query.get("scope_id", [None])[0],
                        allowed_scopes=tuple(
                            scope.strip()
                            for scope in query.get("allowed_scopes", [""])[0].split(",")
                            if scope.strip()
                        )
                        or None,
                    )
                )
            except KeyError:
                return self._json_response(
                    {"error": f"Memory chunk {chunk_id!r} not found."},
                    status=HTTPStatus.NOT_FOUND,
                )
            except ValueError as exc:
                return self._json_response(
                    {"error": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE
                )

        if method == "GET" and path == "/api/jobs":
            return self._json_response(self.service.get_memory_snapshot(limit=0).get("jobs", {}))
        if method == "GET" and path == "/api/jobs/exhausted":
            try:
                return self._json_response(self.service.get_jobs())
            except (ValueError, KeyError) as exc:
                return self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        if method == "GET" and path == "/api/legacy-review":
            try:
                limit = int(query.get("limit", [50])[0] or 50)
                after_id = query.get("after_id", [""])[0] or ""
                return self._json_response(
                    self.service.get_legacy_review(limit=limit, after_id=after_id)
                )
            except (ValueError, KeyError) as exc:
                return self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        if method == "GET" and path == "/api/routing-review":
            try:
                unconfigured = query.get("unconfigured", ["0"])[0] in {"1", "true", "yes"}
                limit = int(query.get("limit", [256])[0] or 256)
                return self._json_response(
                    self.service.get_routing_review(unconfigured=unconfigured, limit=limit)
                )
            except (ValueError, KeyError, OSError) as exc:
                return self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        if method == "POST" and not self._csrf_allowed(headers):
            return self._json_response(
                {"error": "CSRF validation failed."}, status=HTTPStatus.FORBIDDEN
            )

        payload: dict[str, Any] = {}
        if body:
            try:
                parsed_payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._json_response(
                    {"error": "Request body must be valid JSON."}, status=HTTPStatus.BAD_REQUEST
                )
            if not isinstance(parsed_payload, dict):
                return self._json_response(
                    {"error": "Request body must be a JSON object."}, status=HTTPStatus.BAD_REQUEST
                )
            payload = parsed_payload

        if method == "POST" and path == "/api/settings/global":
            return self._json_response(self.service.save_settings(payload, scope="global"))
        if method == "POST" and path == "/api/project":
            try:
                if not isinstance(payload.get("enabled"), bool):
                    raise ValueError("enabled must be a boolean")
                return self._json_response(
                    self.service.configure_project(str(payload["profile"]), payload["enabled"])
                )
            except (KeyError, ValueError) as exc:
                return self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        if method == "POST" and path == "/api/settings/project":
            return self._json_response(self.service.save_settings(payload, scope="project"))
        if method == "POST" and path == "/api/import":
            return self._json_response(self.service.import_existing_setup())
        if method == "POST" and path == "/api/consolidate":
            return self._json_response(self.service.run_consolidation())
        if method == "POST" and path == "/api/jobs/retry":
            try:
                return self._json_response(self.service.retry_job(str(payload["job_id"])))
            except (ValueError, KeyError) as exc:
                return self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        if method == "POST" and path == "/api/routing-review/ack":
            try:
                record_id = str(payload["record_id"])
                unconfigured = payload.get("unconfigured") is True
                return self._json_response(
                    self.service.ack_routing_review(record_id, unconfigured=unconfigured)
                )
            except (KeyError, ValueError, OSError) as exc:
                return self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        if method == "POST" and path == "/api/smoke":
            return self._json_response(self.service.run_smoke_tests())
        if method == "POST" and path == "/api/memory/remember":
            try:
                return self._json_response(
                    self.service.remember_memory(
                        str(payload["content"]), scope=str(payload.get("scope", "project"))
                    )
                )
            except (KeyError, ValueError) as exc:
                return self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        if method == "POST" and path.startswith("/api/memory/"):
            suffix = path.removeprefix("/api/memory/")
            parts = [part for part in suffix.split("/") if part]
            if len(parts) == 2 and parts[1] in {
                "correct",
                "forget",
                "forget-preview",
                "forget-related",
            }:
                try:
                    fact_id = parts[0]
                    if parts[1] == "correct":
                        version = int(payload["expected_version"])
                        result = self.service.correct_memory(
                            fact_id, str(payload["content"]), version
                        )
                    elif parts[1] == "forget":
                        # Preserve the historical UI contract: clients may
                        # submit the optimistic version as a JSON string.
                        version = int(payload["expected_version"])
                        result = self.service.forget_memory(fact_id, version)
                    elif parts[1] == "forget-preview":
                        result = self.service.preview_forget_memory(
                            fact_id,
                            payload["expected_version"],
                            payload.get("limit", 100),
                        )
                    else:
                        result = self.service.forget_related_memory(
                            fact_id,
                            payload["expected_version"],
                            payload["preview_digest"],
                            payload.get("preview_limit", 100),
                            payload.get("selected_fact_ids"),
                            payload.get("selected_source_event_ids"),
                        )
                    return self._json_response(result)
                except (KeyError, TypeError, ValueError) as exc:
                    return self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        integration_prefix = "/api/integrations/"
        if method == "POST" and path.startswith(integration_prefix):
            suffix = path[len(integration_prefix) :]
            parts = [part for part in suffix.split("/") if part]
            if len(parts) == 2 and parts[1] in {"preview", "apply"}:
                host_name = parts[0]
                if host_name not in {"claude-code", "cursor", "codex", "hermes"}:
                    return self._text_response("Unknown host", status=HTTPStatus.NOT_FOUND)
                host = cast(Literal["claude-code", "cursor", "codex", "hermes"], host_name)
                if parts[1] == "preview":
                    return self._json_response(self.service.preview_integration(host))
                return self._json_response(self.service.apply_integration(host))

        return self._text_response("Not found", status=HTTPStatus.NOT_FOUND)


class _MnemosUiHandler(BaseHTTPRequestHandler):
    router: MnemosUiRouter

    def do_GET(self) -> None:  # noqa: N802
        response = self.router.handle("GET", self.path, None, dict(self.headers))
        self._write_response(response)

    def do_POST(self) -> None:  # noqa: N802
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._write_response(
                self.router._json_response({"error": "Invalid content length."}, status=400)
            )
            return
        if content_length < 0 or content_length > 1_048_576:
            self._write_response(
                self.router._json_response({"error": "Request body too large."}, status=413)
            )
            return
        body = self.rfile.read(content_length) if content_length > 0 else None
        response = self.router.handle("POST", self.path, body, dict(self.headers))
        self._write_response(response)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        _ = format, args

    def _write_response(self, response: UiResponse) -> None:
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; style-src 'self' 'unsafe-inline'",
        )
        self.end_headers()
        self.wfile.write(response.body)


def run_ui_server(
    *,
    service: ControlPlaneService,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Mnemos UI must bind to a loopback address")
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
