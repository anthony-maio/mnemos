"""Capture survives abrupt host termination around the service request."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mnemos.client import MnemosClient, Outbox
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore

CHILD = """
import asyncio,json,sys
from pathlib import Path
from mnemos.client import MnemosClient
descriptor,outbox,payload_file,marker,phase=sys.argv[1:]
original=MnemosClient._dispatch_http
async def held(self,*args,**kwargs):
    if phase!='before_send':
        result=await original(self,*args,**kwargs)
        assert result['status']=='committed'
    Path(marker).write_text('request-boundary',encoding='utf-8')
    await asyncio.Event().wait()
MnemosClient._dispatch_http=held
async def main():
    client=MnemosClient(descriptor_path=descriptor,outbox_path=outbox,auto_start=False)
    await client.capture(json.loads(Path(payload_file).read_text(encoding='utf-8')))
asyncio.run(main())
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_send", "after_response", "after_forget"])
async def test_capture_process_killed_at_http_boundary_replays_once_after_restart(
    tmp_path: Path, phase: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id="interrupt-profile", enabled=True)
    context = registry.resolve(
        project, user_profile_id="interrupt-profile", session_id="interrupt-session", host="codex"
    )
    descriptor = tmp_path / "runtime.json"
    outbox = tmp_path / "outbox.json"
    database = tmp_path / "memory.sqlite"
    payload = {
        "event_id": "interrupt-capture",
        "context": context.model_dump(mode="json"),
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "role": "assistant",
        "content": "The interruption recovery convention is willow-birch.",
        "source_kind": "codex_hook",
        "source_ref": "Stop:interrupted-turn",
    }
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps(payload), encoding="utf-8")
    marker = tmp_path / "entered-request.txt"

    def start() -> tuple[MnemosService, SQLiteStore]:
        store = SQLiteStore(str(database))
        engine = MnemosEngine(
            config=MnemosConfig(
                modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
            ),
            llm=UnavailableLLMProvider(),
            embedder=SimpleEmbeddingProvider(),
            store=store,
            scope_registry=registry,
            require_scope_context=True,
        )
        service = MnemosService(
            dispatcher=LocalMemoryService(engine, registry),
            descriptor_path=descriptor,
            lock_path=tmp_path / "runtime.lock",
        )
        service.start()
        return service, store

    service, store = start()
    child = None
    replay_client = None
    try:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                CHILD,
                str(descriptor),
                str(outbox),
                str(payload_file),
                str(marker),
                phase,
            ],
            cwd=tmp_path,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        async def wait_for_boundary() -> None:
            while not marker.exists():
                if child.poll() is not None:
                    _, error = child.communicate()
                    pytest.fail("capture child exited before boundary: " + error.decode())
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_boundary(), timeout=15)
        if phase == "after_forget":
            service.submit("maintenance", {})
            facts = store.get_all()
            assert len(facts) == 1
            service.submit(
                "forget",
                {
                    "context": context.model_dump(mode="json"),
                    "fact_id": facts[0].id,
                    "expected_version": facts[0].version,
                },
            )
        child.kill()
        child.wait(timeout=5)
        assert child.returncode != 0
        pending = Outbox(outbox).pending()
        assert len(pending) == 1, "Host termination lost the capture before durable acknowledgement"
        assert pending[0]["payload"]["event_id"] == payload["event_id"]
        assert len(store.list_episodic_events()) == (1 if phase == "after_response" else 0)
        service.stop()
        store.close()
        service, store = start()
        replay_client = MnemosClient(
            descriptor_path=descriptor, outbox_path=outbox, auto_start=False
        )
        result = await replay_client.replay_outbox()
        assert result == {"schema_version": 1, "acknowledged": 1, "failed": 0, "pending": 0}
        events = store.list_episodic_events()
        jobs = store.list_jobs()
        if phase == "after_forget":
            assert events == []
            assert store.get_all() == []
            assert len(jobs) == 1
            assert jobs[0].status == "completed"
            assert jobs[0].result == {"decision": "forgotten"}
            assert "content" not in jobs[0].payload
            service.submit("maintenance", {})
            assert store.get_all() == []
            return
        assert len(events) == len(jobs) == 1
        assert events[0].content == payload["content"]
        assert jobs[0].kind == "extract"
        assert await replay_client.replay_outbox() == {
            "schema_version": 1,
            "acknowledged": 0,
            "failed": 0,
            "pending": 0,
        }
        assert [e.id for e in store.list_episodic_events()] == [events[0].id]
        assert [j.id for j in store.list_jobs()] == [jobs[0].id]
    finally:
        if child is not None:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
        if replay_client is not None:
            await replay_client.close()
        service.stop()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["committed", "rejected", "blocked", "pending"])
async def test_capture_is_sanitized_and_durable_before_dispatch_until_terminal_reply(
    tmp_path: Path, status: str
) -> None:
    outbox_path = tmp_path / "outbox.json"
    seen = []

    class Dispatcher:
        async def dispatch(self, operation: str, payload: dict) -> dict:
            seen.append(Outbox(outbox_path).pending())
            if outbox_path.exists():
                assert "person@example.com" not in outbox_path.read_text(encoding="utf-8")
            assert operation == "capture"
            assert "person@example.com" not in payload["content"]
            return {"status": status}

    service = MnemosService(dispatcher=Dispatcher(), descriptor_path=tmp_path / "runtime.json")
    client = MnemosClient(
        descriptor_path=tmp_path / "runtime.json", outbox_path=outbox_path, auto_start=False
    )
    service.start()
    try:
        result = await client.capture({"content": "Contact person@example.com"})
        assert result["status"] == status
        assert len(seen) == 1
        assert len(seen[0]) == 1
        assert seen[0][0]["payload"]["event_id"]
        assert "person@example.com" not in json.dumps(seen[0])
        assert len(Outbox(outbox_path).pending()) == (1 if status == "pending" else 0)
    finally:
        await client.close()
        service.stop()
