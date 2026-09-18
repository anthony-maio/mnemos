"""Real worker termination at selected SQLite capture and forget boundaries."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from uuid import uuid4

import pytest

import mnemos.utils.storage as storage_module
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
import asyncio,json,os,sys,threading
from pathlib import Path
import mnemos.utils.storage as storage_module
from mnemos.config import MnemosConfig,ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore

request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert Path(storage_module.__file__).resolve()==Path(request['storage_module']).resolve()
registry=ScopeRegistry.load(request['registry'])
store=SQLiteStore(request['database'])
engine=MnemosEngine(store=store,llm=UnavailableLLMProvider(),embedder=SimpleEmbeddingProvider(),
    config=MnemosConfig(modules=ModuleConfig(**{k:False for k in ModuleConfig.model_fields})))
memory=LocalMemoryService(engine,registry)
connection=store._conn

def hold(boundary):
    record={'boundary':boundary,'pid':os.getpid(),'parent_pid':os.getppid(),
        'nonce':request['nonce'],'operation':request['operation'],'database':request['database'],
        'in_transaction':connection.in_transaction}
    marker=Path(request['marker'])
    temporary=marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(record),encoding='utf-8')
    temporary.replace(marker)
    threading.Event().wait()

class BoundaryConnection:
    def __getattr__(self,name):
        return getattr(connection,name)

    def execute(self,statement,*args,**kwargs):
        result=connection.execute(statement,*args,**kwargs)
        normalized=' '.join(statement.upper().split())
        selectors={
            'after_event_insert':'INSERT INTO EPISODIC_EVENTS',
            'after_suppression_insert':'INSERT INTO MEMORY_SUPPRESSIONS',
            'after_source_delete':'DELETE FROM EPISODIC_EVENTS',
        }
        if normalized.startswith(selectors.get(request['boundary'],'NO MATCH')):
            hold(request['boundary'])
        return result

    def commit(self):
        if request['boundary']=='before_commit':
            hold('before_commit')
        connection.commit()
        if request['boundary']=='after_commit':
            hold('after_commit')

store._conn=BoundaryConnection()
asyncio.run(memory.dispatch(request['operation'],request['payload']))
raise RuntimeError('Requested transaction boundary was not reached')
"""

TABLES = (
    "mnemos_meta",
    "memory_chunks",
    "episodic_events",
    "memory_jobs",
    "fact_evidence",
    "fact_revisions",
    "memory_suppressions",
    "memory_edges",
    "memory_vector_values",
    "memory_chunk_access",
    "retrieval_feedback_events",
    "reconsolidation_jobs",
)


def snapshot(store):
    def encode(value):
        assert isinstance(value, bytes)
        return {"hex": value.hex()}

    return {
        table: hashlib.sha256(
            json.dumps(
                store._conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall(),
                default=encode,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        for table in TABLES
    }


def memory_for(database, registry):
    store = SQLiteStore(str(database))
    engine = MnemosEngine(
        store=store,
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(),
        config=MnemosConfig(
            modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
        ),
    )
    return LocalMemoryService(engine, registry)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "boundary"),
    [
        ("capture", "after_event_insert"),
        ("capture", "before_commit"),
        ("capture", "after_commit"),
        ("forget", "after_suppression_insert"),
        ("forget", "after_source_delete"),
        ("forget", "before_commit"),
        ("forget", "after_commit"),
    ],
)
async def test_worker_kill_keeps_capture_and_forget_atomic_through_replay(
    tmp_path: Path, operation: str, boundary: str
) -> None:
    registry = ScopeRegistry()
    contexts = []
    for name in ("project", "foreign"):
        project = tmp_path / name
        project.mkdir()
        registry.register_project(project, user_profile_id="crash-profile", enabled=True)
        contexts.append(
            registry.resolve(
                project, user_profile_id="crash-profile", session_id=name, host="codex"
            )
        )
    context = contexts[0]
    registry_path, database = tmp_path / "scopes.json", tmp_path / "memory.sqlite"
    registry.save(registry_path)
    memory = memory_for(database, registry)
    store = memory.engine.store
    payload = {
        "context": context.model_dump(mode="json"),
        "event_id": "crash-source",
        "role": "user",
        "content": "The crash recovery label is cedar-otter.",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "source_kind": "codex_hook",
    }
    durable_id = hashlib.sha256(
        json.dumps(
            [context.user_profile_id, context.host, context.session_id, payload["event_id"]]
        ).encode("utf-8")
    ).hexdigest()
    target_id = None
    try:
        for index, selected_context in enumerate(contexts):
            await memory.dispatch(
                "remember",
                {
                    "context": selected_context.model_dump(mode="json"),
                    "event_id": "control",
                    "content": f"Keep control {index} using river-slate labels.",
                },
            )
        assert await memory.maintenance.run_once(force=True) == 2
        controls = {chunk.id: chunk.model_dump(mode="json") for chunk in store.get_all()}
        if operation == "forget":
            assert (await memory.dispatch("capture", payload))["event_id"] == durable_id
            assert await memory.maintenance.run_once(force=True) == 1
            target = next(chunk for chunk in store.get_all() if chunk.id not in controls)
            target_id = target.id
            target_rowid = store._conn.execute(
                "SELECT rowid FROM memory_chunks WHERE id=?", (target_id,)
            ).fetchone()[0]
            # Seed one explicit graph derivation so forget must erase it too.
            neighbor = next(
                chunk.id
                for chunk in store.get_all()
                if chunk.id in controls and chunk.metadata["scope_id"] == context.project_id
            )
            store.replace_graph_preferences(target.id, {neighbor: 0.9})
            store.replace_graph_preferences(neighbor, {target.id: 0.9})
            store.project_graph_edges()
            assert store.get_graph_edges()[target.id][neighbor] == 0.9
        before = snapshot(store)
    finally:
        store.close()

    outbox_path = tmp_path / "outbox.json"
    Outbox(outbox_path).append(payload)
    request = {
        "nonce": str(uuid4()),
        "database": str(database),
        "registry": str(registry_path),
        "storage_module": str(Path(storage_module.__file__).resolve()),
        "operation": operation,
        "boundary": boundary,
        "marker": str(tmp_path / "boundary.json"),
        "payload": (
            payload
            if operation == "capture"
            else {
                "context": context.model_dump(mode="json"),
                "fact_id": target_id,
                "expected_version": 1,
            }
        ),
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("MNEMOS_")}
    environment["PYTHONPATH"] = str(Path(storage_module.__file__).resolve().parents[2])
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(request_path)],
        cwd=tmp_path,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    worker_pid = None
    killed = False
    try:
        marker = Path(request["marker"])

        async def wait_for_boundary():
            while not marker.exists():
                if child.poll() is not None:
                    _, error = child.communicate()
                    pytest.fail("Worker exited before its boundary: " + error.decode())
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_boundary(), timeout=15)
        reached = json.loads(marker.read_text(encoding="utf-8"))
        assert reached["nonce"] == request["nonce"]
        assert reached["operation"] == operation
        assert reached["database"] == str(database)
        assert reached["boundary"] == boundary
        # Windows venv Python can launch the actual interpreter as its child.
        assert reached["pid"] == child.pid or reached["parent_pid"] == child.pid
        worker_pid = reached["pid"]
        assert reached["in_transaction"] is (boundary != "after_commit")
        assert child.poll() is None
        os.kill(worker_pid, signal.SIGTERM)
        killed = True
        child.communicate(timeout=5)
        assert child.returncode != 0
    finally:
        if worker_pid is not None and not killed:
            try:
                os.kill(worker_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if child.poll() is None:
            if sys.platform == "win32":
                # A startup failure can occur before the interpreter publishes
                # its PID. Terminate this owned launcher's descendants as well.
                subprocess.run(
                    ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            if child.poll() is None:
                child.kill()
        child.communicate(timeout=5)

    memory = memory_for(database, ScopeRegistry.load(registry_path))
    store = memory.engine.store
    service = MnemosService(
        dispatcher=memory,
        descriptor_path=tmp_path / "runtime.json",
        lock_path=tmp_path / "runtime.lock",
    )
    client = MnemosClient(
        descriptor_path=service.descriptor_path, outbox_path=outbox_path, auto_start=False
    )
    try:
        assert store._conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert store._conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert len(client.outbox.pending()) == 1
        if boundary != "after_commit":
            assert snapshot(store) == before
        if operation == "capture":
            assert (store.get_episodic_event(durable_id) is not None) == (
                boundary == "after_commit"
            )
            assert (store.get_job("event:" + durable_id) is not None) == (
                boundary == "after_commit"
            )
        elif boundary == "after_commit":
            assert store.get(target_id) is None
            assert store.is_suppressed(source_event_id=durable_id)
        service.start()
        if operation == "forget" and boundary != "after_commit":
            assert (await memory.dispatch("forget", request["payload"]))["forgotten"]
        assert await client.replay_outbox() == {
            "schema_version": 1,
            "acknowledged": 1,
            "failed": 0,
            "pending": 0,
        }
        await memory.maintenance.run_once(force=True)
        for fact_id, original in controls.items():
            assert store.get(fact_id).model_dump(mode="json") == original
        if operation == "capture":
            assert len(store.get_all()) == 3
            assert len(store.list_episodic_events()) == 3
            assert len(store.list_jobs()) == 3
            assert store.get_episodic_event(durable_id).content == payload["content"]
            target = next(chunk for chunk in store.get_all() if chunk.id not in controls)
            assert len(store.list_fact_evidence(target.id)) == 1
            assert store.list_fact_evidence(target.id)[0].source_event_id == durable_id
            assert store.get_job("event:" + durable_id).status == "completed"
        else:
            assert {chunk.id for chunk in store.get_all()} == set(controls)
            assert store.get_episodic_event(durable_id) is None
            assert store.list_fact_evidence(target_id) == []
            assert store.list_fact_revisions(target_id) == []
            assert (
                store._conn.execute(
                    "SELECT 1 FROM memory_vector_values WHERE rowid=?", (target_rowid,)
                ).fetchone()
                is None
            )
            if store._sqlite_vec_enabled:
                assert (
                    store._conn.execute(
                        "SELECT 1 FROM memory_vec WHERE rowid=?", (target_rowid,)
                    ).fetchone()
                    is None
                )
            assert target_id not in store.get_graph_edges()
            assert all(target_id not in edges for edges in store.get_graph_edges().values())
            assert store.get_job("event:" + durable_id).result == {"decision": "forgotten"}
        final = snapshot(store)
        client.outbox.append(payload)
        assert (await client.replay_outbox())["acknowledged"] == 1
        assert snapshot(store) == final
        assert not client.outbox.pending()
    finally:
        await client.close()
        service.stop()
        store.close()
    reopened = SQLiteStore(str(database))
    try:
        assert snapshot(reopened) == final
        assert reopened._conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        reopened.close()
