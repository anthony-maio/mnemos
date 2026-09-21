"""One MnemosClient shared by threads that each run their own event loop."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from mnemos.client import MnemosClient


def test_a_second_loop_does_not_break_a_request_in_flight_on_the_first(tmp_path: Path) -> None:
    first_arrived = threading.Event()
    second_arrived = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body["payload"]["query"] == "first":
                first_arrived.set()
                # Hold the first reply until the other thread has dispatched.
                second_arrived.wait(timeout=5)
            else:
                second_arrived.set()
            data = json.dumps({"schema_version": 1, "status": "ok", "facts": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    descriptor = tmp_path / "runtime.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "host": "127.0.0.1",
                "port": server.server_address[1],
                "pid": os.getpid(),
                "token": "test",
            }
        ),
        encoding="utf-8",
    )
    client = MnemosClient(
        descriptor_path=descriptor,
        outbox_path=tmp_path / "outbox.json",
        auto_start=False,
        request_timeout=3.0,
    )
    results: dict[str, object] = {}

    def call(query: str) -> None:
        # Hermes runs every service call under its own asyncio.run.
        try:
            results[query] = asyncio.run(client.retrieve({"query": query}))
        except Exception as exc:
            results[query] = exc

    try:
        first = threading.Thread(target=call, args=("first",))
        first.start()
        assert first_arrived.wait(timeout=5)
        second = threading.Thread(target=call, args=("second",))
        second.start()
        second.join(timeout=10)
        first.join(timeout=10)
    finally:
        server.shutdown()
        server.server_close()

    for query in ("first", "second"):
        result = results.get(query)
        assert isinstance(result, dict) and result.get("status") == "ok", (query, result)
