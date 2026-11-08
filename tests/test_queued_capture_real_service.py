import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import os
from pathlib import Path
import subprocess
import sys

from mnemos.client import MnemosClient, Outbox
from mnemos.engine import MnemosEngine
from mnemos.hook_worker import HookWorker
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore
from tests.test_hook_worker import configured, payload, send  # noqa: F401


def memory_owner(root):
    def factory():
        engine = MnemosEngine(
            llm=UnavailableLLMProvider(),
            embedder=SimpleEmbeddingProvider(),
            store=SQLiteStore(str(root / "memory.sqlite")),
        )
        for flag in type(engine.config.modules).model_fields:
            setattr(engine.config.modules, flag, False)
        return LocalMemoryService(engine, ScopeRegistry.load(root / "scopes.json"))

    return MnemosService(dispatcher_factory=factory, descriptor_path=root / "memory-runtime.json")


def wait_empty(root):
    deadline = time.monotonic() + 5
    while Outbox(root / "outbox.json").pending() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Outbox(root / "outbox.json").pending()


def test_duplicate_queue_delivery_completion_correction_and_forgotten_replay(configured):
    root = configured
    worker = HookWorker(
        "codex", descriptor_path=root / "worker.json", cwd=root, queue_captures=True
    )
    worker.start()
    try:
        result = send(root, payload(root))
        assert result["capture_state"] == "queued"
        original = Outbox(root / "outbox.json").pending()[0]["payload"]
    finally:
        worker.stop()

    memory = memory_owner(root)
    memory.start()
    restarted = HookWorker(
        "codex", descriptor_path=root / "worker.json", cwd=root, queue_captures=True
    )
    restarted.start()

    async def duplicate():
        client = MnemosClient(descriptor_path=root / "memory-runtime.json", auto_start=False)
        try:
            return await client.request("capture", original, queue_on_failure=False)
        finally:
            await client.close()

    try:
        # Two direct deliveries race the resident replay with the same identity.
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda _: asyncio.run(duplicate()), range(2)))
        assert all(r["status"] == "committed" for r in replies)
        assert len({r["job_id"] for r in replies}) == 1
        wait_empty(root)
        reader = SQLiteStore(str(root / "memory.sqlite"))
        try:
            assert len(reader.list_episodic_events()) == 1
            assert len(reader.list_jobs()) == 1
            assert reader.get_all() == []  # Durable capture is not completed processing.
            assert memory.submit("maintenance", {})["completed"] == 1
            jobs = reader.list_jobs()
            assert len(jobs) == 1 and jobs[0].status == "completed"
            fact = reader.get_all()[0]
            memory.submit(
                "correct",
                {
                    "context": original["context"],
                    "fact_id": fact.id,
                    "expected_version": fact.version,
                    "content": "Use the corrected test command.",
                    "event_id": "explicit-correction",
                },
            )
            assert memory.submit("maintenance", {})["completed"] == 1
            revised = reader.get(fact.id)
            assert revised.version == fact.version + 1
            assert revised.content == "Use the corrected test command."
            memory.submit(
                "forget",
                {
                    "context": original["context"],
                    "fact_id": fact.id,
                    "expected_version": revised.version,
                },
            )
            assert reader.get(fact.id) is None
        finally:
            reader.close()
    finally:
        restarted.stop()
        memory.stop()

    # A separate interpreter must observe the persisted suppression, not a
    # surviving owner's in-memory state. No service or host is started here.
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from mnemos.utils.storage import SQLiteStore; "
            "s=SQLiteStore(sys.argv[1]); "
            "assert s.get_all()==[]; "
            "assert s.is_suppressed(source_event_id=sys.argv[2]); s.close()",
            str(root / "memory.sqlite"),
            replies[0]["event_id"],
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert check.returncode == 0, check.stderr

    # Recreate the old queue after forgetting, then reopen both owners.
    Outbox(root / "outbox.json").append(original)
    memory = memory_owner(root)
    memory.start()
    restarted = HookWorker(
        "codex", descriptor_path=root / "worker.json", cwd=root, queue_captures=True
    )
    restarted.start()
    try:
        wait_empty(root)
        assert memory.submit("maintenance", {})["completed"] == 0
        reader = SQLiteStore(str(root / "memory.sqlite"))
        try:
            assert reader.get_all() == []
            assert reader.list_jobs(status="pending") == []
            assert reader.list_episodic_events() == []
        finally:
            reader.close()
    finally:
        restarted.stop()
        memory.stop()
