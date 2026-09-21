"""Forgetting wins over earlier writes; only explicit re-teaching may override it."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from mnemos.backup import create_backup, restore_backup
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore

FACT = "The staging password rotation happens on Fridays."


def _service(tmp_path: Path) -> tuple[LocalMemoryService, dict[str, Any]]:
    project = tmp_path / "project"
    project.mkdir()
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id="local", enabled=True)
    context = registry.resolve(project, user_profile_id="local", session_id="s", host="codex")
    modules = ModuleConfig(
        surprisal=False, mutable_rag=False, affective=False, sleep=False, spreading=False
    )
    memory = LocalMemoryService(
        MnemosEngine(
            config=MnemosConfig(modules=modules),
            llm=UnavailableLLMProvider(),
            embedder=SimpleEmbeddingProvider(),
            store=SQLiteStore(str(tmp_path / "memory.sqlite")),
        ),
        registry,
    )
    return memory, context.model_dump()


async def _remember(
    memory: LocalMemoryService, context: dict[str, Any], event_id: str, content: str = FACT
) -> dict[str, Any]:
    return await memory.dispatch(
        "remember", {"context": context, "event_id": event_id, "content": content}
    )


async def _capture(
    memory: LocalMemoryService,
    context: dict[str, Any],
    event_id: str,
    *,
    content: str = FACT,
    occurred_at: datetime | None = None,
) -> dict[str, Any]:
    return await memory.dispatch(
        "capture",
        {
            "context": context,
            "event_id": event_id,
            "occurred_at": (occurred_at or datetime.now(timezone.utc)).isoformat(),
            "role": "user",
            "source_kind": "codex_user_prompt",
            "content": content,
        },
    )


async def _forget(memory: LocalMemoryService, context: dict[str, Any], fact_id: str) -> None:
    fact = memory.engine.store.get(fact_id)
    assert fact is not None
    result = await memory.dispatch(
        "forget", {"context": context, "fact_id": fact.id, "expected_version": fact.version}
    )
    assert result["forgotten"] is True


async def _settle(memory: LocalMemoryService) -> None:
    await memory.maintenance.run_once(force=True)


def _contents(store: SQLiteStore) -> list[str]:
    return [chunk.content for chunk in store.get_all()]


def _raw_copies(store: SQLiteStore, content: str = FACT) -> int:
    row = store._conn.execute(  # noqa: SLF001 - verify raw evidence removal
        "SELECT COUNT(*) FROM episodic_events WHERE content=?", (content,)
    ).fetchone()
    return int(row[0])


async def test_remember_queued_before_forget_cannot_recreate_the_fact(tmp_path: Path) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        await _remember(memory, context, "first")
        await _settle(memory)
        fact_id = store.get_all()[0].id
        queued = await _remember(memory, context, "queued-duplicate")
        await _forget(memory, context, fact_id)
        await _settle(memory)
        assert FACT not in _contents(store)
        job = store.get_job(queued["job_id"])
        assert job is not None and job.status == "completed"
    finally:
        store.close()


async def test_correction_queued_before_forget_cannot_write_forgotten_content(
    tmp_path: Path,
) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    other_text = "Rotation owner is the platform team."
    try:
        await _remember(memory, context, "secret")
        await _remember(memory, context, "other", other_text)
        await _settle(memory)
        secret = next(c for c in store.get_all() if c.content == FACT)
        other = next(c for c in store.get_all() if c.content == other_text)
        await memory.dispatch(
            "correct",
            {
                "context": context,
                "event_id": "rewrite",
                "fact_id": other.id,
                "expected_version": other.version,
                "content": FACT,
            },
        )
        await _forget(memory, context, secret.id)
        await _settle(memory)
        survivor = store.get(other.id)
        assert survivor is not None
        assert (survivor.content, survivor.version) == (other_text, other.version)
        assert FACT not in _contents(store)
    finally:
        store.close()


async def test_exhausted_remember_from_before_forget_cannot_be_retried_back(
    tmp_path: Path,
) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        await _remember(memory, context, "first")
        await _settle(memory)
        fact_id = store.get_all()[0].id
        queued = await _remember(memory, context, "outage-duplicate")
        store._conn.execute(  # noqa: SLF001 - simulate a provider outage exhausting the job
            "UPDATE memory_jobs SET status='exhausted', attempt_count=max_attempts, "
            "last_error='ConnectionError' WHERE id=?",
            (queued["job_id"],),
        )
        store._conn.commit()  # noqa: SLF001
        await _forget(memory, context, fact_id)
        job = store.get_job(queued["job_id"])
        assert job is not None and job.status == "completed"
        with pytest.raises(ValueError, match="Only exhausted jobs"):
            await memory.dispatch("retry_job", {"context": context, "job_id": queued["job_id"]})
        await _settle(memory)
        assert FACT not in _contents(store)
        assert _raw_copies(store) == 0
    finally:
        store.close()


async def test_copy_leased_during_forget_is_blocked_when_it_commits(tmp_path: Path) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        await _remember(memory, context, "first")
        await _settle(memory)
        fact_id = store.get_all()[0].id
        queued = await _remember(memory, context, "in-flight-copy")
        store._conn.execute(  # noqa: SLF001 - a worker holds the copy's lease
            "UPDATE memory_jobs SET status='leased', lease_owner='worker', "
            "lease_token='token', lease_until=? WHERE id=?",
            ((datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(), queued["job_id"]),
        )
        store._conn.commit()  # noqa: SLF001
        await _forget(memory, context, fact_id)
        assert _raw_copies(store) == 1  # forget does not revoke a live worker's lease
        store._conn.execute(  # noqa: SLF001 - that worker then fails and exhausts the job
            "UPDATE memory_jobs SET status='exhausted', lease_owner=NULL, lease_token=NULL, "
            "lease_until=NULL, attempt_count=max_attempts WHERE id=?",
            (queued["job_id"],),
        )
        store._conn.commit()  # noqa: SLF001
        await memory.dispatch("retry_job", {"context": context, "job_id": queued["job_id"]})
        await _settle(memory)
        assert FACT not in _contents(store)
        job = store.get_job(queued["job_id"])
        assert job is not None and (job.result or {}).get("decision") == "suppressed"
    finally:
        store.close()


async def test_forget_removes_queued_copies_of_the_forgotten_statement(tmp_path: Path) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        await _remember(memory, context, "first")
        await _settle(memory)
        queued = await _remember(memory, context, "queued-copy")
        await _forget(memory, context, store.get_all()[0].id)
        assert _raw_copies(store) == 0
        job = store.get_job(queued["job_id"])
        assert job is not None and job.status == "completed"
    finally:
        store.close()


async def test_automatic_capture_after_forget_without_reteach_is_suppressed(
    tmp_path: Path,
) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        await _remember(memory, context, "teach")
        await _settle(memory)
        await _forget(memory, context, store.get_all()[0].id)
        receipt = await _capture(memory, context, "assistant-context-echo")
        assert receipt["processing_status"] == "suppressed"
        await _settle(memory)
        assert FACT not in _contents(store)
    finally:
        store.close()


async def test_explicit_reteach_supersedes_forget_and_later_captures_reinforce_it(
    tmp_path: Path,
) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        await _remember(memory, context, "teach")
        await _settle(memory)
        await _forget(memory, context, store.get_all()[0].id)
        await _remember(memory, context, "reteach")
        await _settle(memory)
        retaught = [c for c in store.get_all() if c.content == FACT]
        assert len(retaught) == 1
        superseded = store._conn.execute(  # noqa: SLF001 - verify durable supersession
            "SELECT superseded_at FROM memory_suppressions WHERE canonical_key=?",
            (retaught[0].canonical_key,),
        ).fetchall()
        assert superseded and all(row[0] is not None for row in superseded)
        receipts = [await _capture(memory, context, f"later-{index}") for index in range(3)]
        assert all(receipt["processing_status"] != "suppressed" for receipt in receipts)
        await _settle(memory)
        reinforced = store.get(retaught[0].id)
        assert reinforced is not None
        assert reinforced.observation_count > retaught[0].observation_count
    finally:
        store.close()


async def test_pre_forget_event_replayed_after_reteach_stays_blocked(tmp_path: Path) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        spoken_before_forget = datetime.now(timezone.utc) - timedelta(minutes=5)
        await _remember(memory, context, "teach")
        await _settle(memory)
        await _forget(memory, context, store.get_all()[0].id)
        await _remember(memory, context, "reteach")
        await _settle(memory)
        retaught = next(c for c in store.get_all() if c.content == FACT)
        receipt = await _capture(
            memory, context, "old-transcript-line", occurred_at=spoken_before_forget
        )
        assert receipt["processing_status"] == "suppressed"
        await _settle(memory)
        current = store.get(retaught.id)
        assert current is not None
        assert current.observation_count == retaught.observation_count
    finally:
        store.close()


async def test_restore_keeps_retaught_fact_without_adding_suppressions(tmp_path: Path) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        await _remember(memory, context, "teach")
        await _settle(memory)
        await _forget(memory, context, store.get_all()[0].id)
        await _remember(memory, context, "reteach")
        await _settle(memory)
        live_ids = [c.id for c in store.get_all()]
        assert _contents(store) == [FACT]
        live_rows = store._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM memory_suppressions"
        ).fetchone()[0]
        archive = tmp_path / "current.zip"
        create_backup(store, archive, memory.registry)
        destination = tmp_path / "restored.sqlite"
        restore_backup(archive, destination, current_store=store, current_registry=memory.registry)
        restored = SQLiteStore(str(destination))
        try:
            assert [c.id for c in restored.get_all()] == live_ids
            restored_rows = restored._conn.execute(  # noqa: SLF001
                "SELECT COUNT(*) FROM memory_suppressions"
            ).fetchone()[0]
            assert restored_rows == live_rows
        finally:
            restored.close()
    finally:
        store.close()


async def test_restoring_pre_forget_backup_does_not_resurrect_the_fact(tmp_path: Path) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        await _remember(memory, context, "teach")
        await _settle(memory)
        archive = tmp_path / "before-forget.zip"
        create_backup(store, archive, memory.registry)
        await _forget(memory, context, store.get_all()[0].id)
        destination = tmp_path / "restored.sqlite"
        restore_backup(archive, destination, current_store=store, current_registry=memory.registry)
        restored = SQLiteStore(str(destination))
        try:
            assert FACT not in _contents(restored)
            assert _raw_copies(restored) == 0
        finally:
            restored.close()
    finally:
        store.close()


async def test_independent_fact_matching_an_old_revision_survives_forget_and_restore(
    tmp_path: Path,
) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        await _remember(memory, context, "f1", "Use npm")
        await _settle(memory)
        corrected = store.get_all()[0]
        await memory.dispatch(
            "correct",
            {
                "context": context,
                "event_id": "f2",
                "content": "Use pnpm",
                "fact_id": corrected.id,
                "expected_version": corrected.version,
            },
        )
        await _settle(memory)
        await _remember(memory, context, "g1", "Use npm")
        await _settle(memory)
        independent = next(c for c in store.get_all() if c.content == "Use npm")
        await _forget(memory, context, corrected.id)
        assert [c.id for c in store.get_all()] == [independent.id]
        archive = tmp_path / "after-forget.zip"
        create_backup(store, archive, memory.registry)
        destination = tmp_path / "restored.sqlite"
        restore_backup(archive, destination, current_store=store, current_registry=memory.registry)
        restored = SQLiteStore(str(destination))
        try:
            assert [c.id for c in restored.get_all()] == [independent.id]
        finally:
            restored.close()
        receipt = await _capture(memory, context, "npm-again", content="Use npm")
        assert receipt["processing_status"] != "suppressed"
    finally:
        store.close()


def test_schema_upgrade_adds_supersession_columns_and_keeps_rows_active(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    store = SQLiteStore(str(path))
    try:
        store.record_suppression(canonical_key="key-1", reason="explicit_forget")
        store._conn.execute(  # noqa: SLF001 - rebuild the pre-supersession layout
            "ALTER TABLE memory_suppressions DROP COLUMN superseded_at"
        )
        store._conn.execute(  # noqa: SLF001
            "ALTER TABLE memory_suppressions DROP COLUMN superseded_by_event_id"
        )
        store._conn.execute(  # noqa: SLF001
            "UPDATE mnemos_meta SET value='9' WHERE key='schema_version'"
        )
        store._conn.commit()  # noqa: SLF001
    finally:
        store.close()
    upgraded = SQLiteStore(str(path))
    try:
        rows = upgraded._conn.execute(  # noqa: SLF001
            "SELECT canonical_key, superseded_at, superseded_by_event_id FROM memory_suppressions"
        ).fetchall()
        assert rows == [("key-1", None, None)]
        assert upgraded.is_suppressed(canonical_key="key-1")
    finally:
        upgraded.close()
