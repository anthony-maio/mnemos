"""Temporary local UI fixture for the Playwright smoke checkpoint.

This runner owns no persistent configuration. It creates a temporary service,
registry, database, and UI listener, then prints the loopback URL for a browser
driver. Stop it with Ctrl+C after the browser evidence is captured.
"""

from __future__ import annotations

import asyncio
import argparse
import os
from datetime import datetime, timedelta, timezone
import signal
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from mnemos.client import MnemosClient
from mnemos.config import MnemosConfig
from mnemos.control_plane import ControlPlaneService
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.routing_review import enqueue_routing_failure
from mnemos.types import MemoryChunk
from mnemos.service import MnemosService
from mnemos.ui_server import MnemosUiRouter, _MnemosUiHandler
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore

HTML_FIXTURE = '<img src=x onerror="window.__mnemos_xss = true"> Keep this escaped.'


def _build_fixture(
    root: Path, *, exhausted_job: bool = False, review_queues: bool = False
) -> tuple[MnemosService, ControlPlaneService, ScopeRegistry]:
    project = root / "project"
    project.mkdir()
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id="ui-smoke", enabled=True)
    registry_path = root / "scopes.json"
    registry.save(registry_path)
    config = MnemosConfig()
    config.modules.surprisal = False
    config.modules.sleep = False
    config.modules.affective = False
    engine = MnemosEngine(
        config=config,
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(dim=64),
        store=SQLiteStore(str(root / "memory.sqlite")),
    )
    domain = LocalMemoryService(engine, registry)
    service = MnemosService(
        dispatcher=domain,
        descriptor_path=root / "runtime.json",
        lock_path=root / "runtime.lock",
        startup_timeout=3,
        dispatch_timeout=10,
    )
    service.start()
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    context = registry.resolve(
        project, user_profile_id="ui-smoke", session_id="ui-smoke-session", host="manual"
    )
    result = asyncio.run(
        client.request(
            "remember",
            {
                "context": context.model_dump(mode="json"),
                "event_id": "ui-smoke-seed",
                "content": HTML_FIXTURE,
                "scope": "project",
            },
            queue_on_failure=False,
        )
    )
    if result.get("status") != "committed":
        service.stop()
        raise RuntimeError(f"fixture seed failed: {result}")
    asyncio.run(
        client.request(
            "maintenance", {"context": context.model_dump(mode="json")}, queue_on_failure=False
        )
    )
    if exhausted_job:
        captured = asyncio.run(
            client.request(
                "remember",
                {
                    "context": context.model_dump(mode="json"),
                    "content": "The recovery fixture uses SQLite after the provider returns.",
                    "event_id": "ui-retry-seed",
                    "scope": "project",
                },
                queue_on_failure=False,
            )
        )
        for attempt in range(5):
            now = datetime.now(timezone.utc) + timedelta(minutes=attempt * 2)
            claimed = engine.store.claim_jobs(owner="ui-outage", now=now)[0]
            assert claimed.id == captured["job_id"]
            assert engine.store.fail_job(
                claimed.id,
                owner="ui-outage",
                lease_token=claimed.lease_token,
                error="ConnectionError",
                now=now,
            )
    if review_queues:
        engine.store.store(
            MemoryChunk(
                id="legacy-review-fixture",
                content="<b>Legacy convention</b>: review before assignment.",
                embedding=engine.embedder.embed("Legacy convention"),
                metadata={},
            )
        )
        for profile in ("ui-smoke", "foreign-profile", None):
            enqueue_routing_failure(
                host="codex",
                event="SessionStart",
                reason="project_not_enabled" if profile else "missing_user_profile",
                session_id="fixture-" + str(profile),
                project_hint=str(project),
                configured_profile=profile,
                directory=root / "routing-review",
            )
        engine.store.prune_processed_evidence(
            before=datetime.now(timezone.utc) + timedelta(seconds=1)
        )
    control = ControlPlaneService(
        cwd=project,
        env={"MNEMOS_USER_PROFILE_ID": "ui-smoke", "MNEMOS_SCOPE_REGISTRY": str(registry_path)},
        global_config_path=root / "Mnemos" / "mnemos.toml",
        client=client,
    )
    return service, control, registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exhausted-job", action="store_true")
    parser.add_argument("--review-queues", action="store_true")
    args = parser.parse_args()
    stop = threading.Event()
    with TemporaryDirectory(prefix="mnemos-ui-smoke-") as directory:
        root = Path(directory)
        # This variable belongs only to the disposable fixture process.
        os.environ["MNEMOS_ROUTING_REVIEW_DIR"] = str(root / "routing-review")
        service, control, _ = _build_fixture(
            root, exhausted_job=args.exhausted_job, review_queues=args.review_queues
        )
        router = MnemosUiRouter(control)

        class Handler(_MnemosUiHandler):
            pass

        Handler.router = router
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"http://127.0.0.1:{server.server_port}/", flush=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
        try:
            stop.wait()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
            service.stop()


if __name__ == "__main__":
    main()
