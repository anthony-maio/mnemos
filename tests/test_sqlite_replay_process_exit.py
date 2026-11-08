"""Actual service replay survives exit on either side of a batch commit."""

import asyncio
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mnemos.client import MnemosClient
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService
from mnemos.sqlite_outbox import SQLiteOutbox
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


@pytest.mark.parametrize("after_commit", [False, True])
def test_process_exit_at_ack_boundary_replays_original_jobs(tmp_path, after_commit):
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="batch-exit", enabled=True)
    context = registry.resolve(
        tmp_path, user_profile_id="batch-exit", session_id="exit", host="codex"
    ).model_dump(mode="json")
    descriptor = tmp_path / "runtime.json"
    database = tmp_path / "memory.sqlite"
    queue_path = tmp_path / "outbox.sqlite"
    queue = SQLiteOutbox(queue_path)
    for i in range(35):
        queue.append({
            "event_id": f"exit-{i}", "context": context,
            "occurred_at": "2026-11-08T00:00:00+00:00", "role": "user",
            "content": f"Synthetic process exit record {i}.",
            "source_kind": "test_capture", "source_ref": f"exit:{i}",
        }, event_id=f"exit-{i}")
    original = queue.pending()
    queue.close()

    def start():
        store = SQLiteStore(str(database))
        engine = MnemosEngine(
            config=MnemosConfig(modules=ModuleConfig(**{
                name: False for name in ModuleConfig.model_fields
            })), llm=UnavailableLLMProvider(), embedder=SimpleEmbeddingProvider(),
            store=store, scope_registry=registry, require_scope_context=True,
        )
        service = MnemosService(
            dispatcher=LocalMemoryService(engine, registry), descriptor_path=descriptor,
            lock_path=tmp_path / "runtime.lock",
        )
        service.start()
        return service, store

    script = tmp_path / "exit_at_batch.py"
    script.write_text(
        "import asyncio, os, sys\n"
        "from mnemos.client import MnemosClient\n"
        "from mnemos.sqlite_outbox import SQLiteOutbox\n"
        "class ExitQueue(SQLiteOutbox):\n"
        "    def acknowledge_many(self, ids):\n"
        "        assert len(ids) == 32\n"
        "        if sys.argv[3] == 'after':\n"
        "            assert super().acknowledge_many(ids) == 32\n"
        "        os._exit(73)\n"
        "q = ExitQueue(sys.argv[1])\n"
        "client = MnemosClient(descriptor_path=sys.argv[2], outbox=q, auto_start=False)\n"
        "asyncio.run(client.replay_outbox())\n",
        encoding="utf-8",
    )
    service, store = start()
    try:
        result = subprocess.run(
            [sys.executable, str(script), str(queue_path), str(descriptor),
             "after" if after_commit else "before"],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])),
            cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 73, result.stderr
        events_before = {e.id for e in store.list_episodic_events()}
        jobs_before = {j.id for j in store.list_jobs()}
        assert len(events_before) == len(jobs_before) == 32
    finally:
        service.stop()
        store.close()

    queue = SQLiteOutbox(queue_path)
    assert queue.pending() == (original[32:] if after_commit else original)
    service, store = start()

    async def replay():
        client = MnemosClient(descriptor_path=descriptor, outbox=queue, auto_start=False)
        try:
            return await client.replay_outbox()
        finally:
            await client.close()

    try:
        result = asyncio.run(replay())
        assert result == {"schema_version": 1, "acknowledged": 3 if after_commit else 35,
                          "failed": 0, "pending": 0}
        events = store.list_episodic_events()
        jobs = store.list_jobs()
        assert len(events) == len(jobs) == 35
        assert events_before <= {e.id for e in events}
        assert jobs_before <= {j.id for j in jobs}
        assert {e.metadata["source_event_id"] for e in events} == {
            entry["payload"]["event_id"] for entry in original
        }
        assert queue.pending() == []
    finally:
        queue.close()
        service.stop()
        store.close()
