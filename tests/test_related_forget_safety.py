"""Adversarial checks for reviewed deletion state and concurrent extraction."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, cast

import pytest

from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.jobs import DurableJob
from mnemos.utils.storage import SQLiteStore
from tests.test_related_forget import _preview_payload, _seed_overlap, _service


async def _capture(memory: Any, context: Any, assertion: str, event_id: str) -> dict[str, Any]:
    result = await memory.dispatch(
        "capture",
        {
            "context": context.model_dump(mode="json"),
            "event_id": event_id,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "role": "user",
            "content": f"A separate capture contains: {assertion}",
            "source_kind": "codex_hook",
        },
    )
    return cast(dict[str, Any], result)


def _selection(context: Any, target: Any, preview: dict[str, Any]) -> dict[str, Any]:
    return {
        **_preview_payload(context, target),
        "preview_digest": preview["preview_digest"],
        "preview_limit": preview["limit"],
        "selected_fact_ids": [target.id] + [item["fact_id"] for item in preview["candidates"]],
        "selected_source_event_ids": [item["source_event_id"] for item in preview["source_events"]],
    }


@pytest.mark.asyncio
async def test_new_matching_capture_invalidates_complete_preview(tmp_path: Path) -> None:
    memory, context, _, _ = _service(tmp_path)
    try:
        assertion, target, copy = await _seed_overlap(memory, context)
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        await _capture(memory, context, assertion, "new-after-preview")
        with pytest.raises(ValueError, match="stale"):
            await memory.dispatch("forget_related", _selection(context, target, preview))
        assert memory.engine.store.get(target.id) is not None
        assert memory.engine.store.get(copy.id) is not None
        assert not memory.engine.store.is_suppressed(fact_id=target.id)
    finally:
        cast(SQLiteStore, memory.engine.store).close()


@pytest.mark.asyncio
async def test_selected_pending_source_cannot_commit_from_an_existing_lease(tmp_path: Path) -> None:
    memory, context, _, _ = _service(tmp_path)
    store = cast(SQLiteStore, memory.engine.store)
    try:
        assertion, target, _ = await _seed_overlap(memory, context)
        captured = await _capture(memory, context, assertion, "leased-capture")
        lease = store.claim_jobs(owner="slow-worker", job_id=captured["job_id"])[0]
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        source = next(item for item in preview["source_events"] if lease.id in item["job_ids"])
        await memory.dispatch("forget_related", _selection(context, target, preview))
        late = MemoryChunk(
            id="late-extracted-copy",
            content=assertion,
            metadata={
                "scope": "project",
                "scope_id": context.project_id,
                "user_profile_id": context.user_profile_id,
                "source": "sleep_consolidation",
            },
        )
        assert (
            store.commit_fact_and_job(
                lease.id,
                owner="slow-worker",
                lease_token=lease.lease_token,
                chunk=late,
                processed_event_ids=[source["source_event_id"]],
            )
            is False
        )
        assert store.get(late.id) is None
        assert store.get_episodic_event(source["source_event_id"]) is None
        assert store.get_all() == []
        rows = store._conn.execute("SELECT * FROM memory_suppressions").fetchall()
        assert rows
        suppression_text = json.dumps([list(row) for row in rows])
        assert assertion not in suppression_text
        assert "related-forget-old" not in suppression_text
    finally:
        store.close()


@pytest.mark.asyncio
async def test_unknown_and_explicit_source_provenance_is_preserved(tmp_path: Path) -> None:
    memory, context, _, _ = _service(tmp_path)
    try:
        assertion, target, _ = await _seed_overlap(memory, context)
        for event_id, extra in (
            ("unknown-origin", {}),
            ("manual-origin", {"explicit_remember": True}),
        ):
            memory.engine.store.append_episodic_event(
                EpisodicEvent(
                    id=event_id,
                    event_fingerprint=event_id,
                    role="user",
                    content=assertion,
                    normalized_content=assertion.lower(),
                    scope="project",
                    scope_id=context.project_id,
                    metadata={"user_profile_id": context.user_profile_id, **extra},
                )
            )
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        source_ids = {item["source_event_id"] for item in preview["source_events"]}
        assert not source_ids.intersection({"unknown-origin", "manual-origin"})
        await memory.dispatch("forget_related", _selection(context, target, preview))
        assert memory.engine.store.get_episodic_event("unknown-origin") is not None
        assert memory.engine.store.get_episodic_event("manual-origin") is not None
    finally:
        cast(SQLiteStore, memory.engine.store).close()


@pytest.mark.asyncio
async def test_conflicting_profile_metadata_is_not_a_deletion_candidate(tmp_path: Path) -> None:
    memory, context, _, _ = _service(tmp_path)
    store = cast(SQLiteStore, memory.engine.store)
    try:
        _, target, copy = await _seed_overlap(memory, context)
        # Simulate inconsistent legacy/indexed identity without changing the
        # authoritative metadata to match the current profile.
        store._conn.execute(
            "UPDATE memory_chunks SET metadata=json_set(metadata, '$.user_profile_id', ?) WHERE id=?",
            ("foreign-profile", copy.id),
        )
        store._conn.commit()
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        assert copy.id not in {item["fact_id"] for item in preview["candidates"]}
        with pytest.raises(ValueError):
            await memory.dispatch(
                "forget_related",
                {**_selection(context, target, preview), "selected_fact_ids": [target.id, copy.id]},
            )
        assert store.get(target.id) is not None
        assert store.get(copy.id) is not None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_conflicting_source_profile_metadata_is_not_previewed(tmp_path: Path) -> None:
    memory, context, _, _ = _service(tmp_path)
    store = cast(SQLiteStore, memory.engine.store)
    try:
        assertion, target, _ = await _seed_overlap(memory, context)
        captured = await _capture(memory, context, assertion, "conflicting-source-profile")
        pending_job = next(job for job in store.list_jobs() if job.id == captured["job_id"])
        source_id = pending_job.payload["event_id"]
        store._conn.execute(
            "UPDATE episodic_events SET metadata=json_set(metadata, '$.user_profile_id', ?) WHERE id=?",
            ("foreign-profile", source_id),
        )
        store._conn.commit()
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        assert source_id not in {item["source_event_id"] for item in preview["source_events"]}
        with pytest.raises(ValueError):
            await memory.dispatch(
                "forget_related",
                {**_selection(context, target, preview), "selected_source_event_ids": [source_id]},
            )
        assert store.get_episodic_event(source_id) is not None
        assert store.get(target.id) is not None
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_owner", ["target", "copy"])
@pytest.mark.parametrize("dependency_field", ["event_id", "evidence_ids"])
@pytest.mark.parametrize("expire_source", [True, False])
async def test_expired_source_keeps_unselected_job_dependencies(
    tmp_path: Path, source_owner: str, dependency_field: str, expire_source: bool
) -> None:
    memory, context, _, _ = _service(tmp_path)
    store = cast(SQLiteStore, memory.engine.store)
    try:
        _, target, copy = await _seed_overlap(memory, context)
        owner = target if source_owner == "target" else copy
        source_id = next(
            item.source_event_id
            for item in store.list_fact_evidence(owner.id)
            if item.source_event_id
        )
        unrelated = MemoryChunk(
            id="unselected-expired-source-user",
            content="Another record must survive this rejected operation.",
            metadata={
                "scope": "project",
                "scope_id": context.project_id,
                "user_profile_id": context.user_profile_id,
                "source": "manual",
            },
        )
        store.store(unrelated)
        job = store.enqueue_job(
            DurableJob(
                kind="correct",
                payload={
                    "target_fact_id": unrelated.id,
                    dependency_field: source_id if dependency_field == "event_id" else [source_id],
                    "content": "Unselected job payload must survive.",
                },
            )
        )
        if expire_source:
            store._conn.execute("DELETE FROM episodic_events WHERE id=?", (source_id,))
            store._conn.commit()
        with pytest.raises(ValueError):
            preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
            await memory.dispatch("forget_related", _selection(context, target, preview))
        for fact_id in [target.id, copy.id, unrelated.id]:
            assert store.get(fact_id) is not None
        retained_job = store.get_job(job.id)
        assert retained_job is not None
        assert retained_job.payload == job.payload
        assert retained_job.status == job.status
    finally:
        store.close()
    reopened = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        for fact_id in [target.id, copy.id, unrelated.id]:
            assert reopened.get(fact_id) is not None
        retained_job = reopened.get_job(job.id)
        assert retained_job is not None
        assert retained_job.payload == job.payload
        assert retained_job.status == job.status
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_required_foreign_source_detail_is_redacted(tmp_path: Path) -> None:
    memory, context, _, _ = _service(tmp_path)
    store = cast(SQLiteStore, memory.engine.store)
    try:
        _, target, _ = await _seed_overlap(memory, context)
        source_id = next(
            item.source_event_id
            for item in store.list_fact_evidence(target.id)
            if item.source_event_id
        )
        secret = "Private foreign source text must not enter the preview."
        store._conn.execute(
            "UPDATE episodic_events SET content=?, scope_id=?, user_profile_id=?, "
            "metadata=json_set(metadata, '$.scope_id', ?, '$.user_profile_id', ?) WHERE id=?",
            (
                secret,
                "private-project",
                "private-profile",
                "private-project",
                "private-profile",
                source_id,
            ),
        )
        store._conn.commit()
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        detail = next(
            item
            for item in preview["target"]["affected_sources"]
            if item["source_event_id"] == source_id
        )
        assert detail["source_allowed"] is False
        for field in [
            "content",
            "scope",
            "scope_id",
            "user_profile_id",
            "job_ids",
            "derived_fact_ids",
        ]:
            assert not detail[field]
        serialized = json.dumps(preview)
        for private_value in [secret, "private-project", "private-profile"]:
            assert private_value not in serialized
        with pytest.raises(ValueError):
            await memory.dispatch("forget_related", _selection(context, target, preview))
        assert store.get(target.id) is not None
        retained_source = store.get_episodic_event(source_id)
        assert retained_source is not None
        assert retained_source.content == secret
    finally:
        store.close()


def _database_rows(store: SQLiteStore) -> dict[str, list[str]]:
    tables = store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {
        name: sorted(
            repr(tuple(row))
            for row in store._conn.execute('SELECT * FROM "' + name.replace('"', '""') + '"')
        )
        for (name,) in tables
    }


@pytest.mark.asyncio
async def test_related_rollback_preserves_all_tables_after_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, context, _, _ = _service(tmp_path)
    store = cast(SQLiteStore, memory.engine.store)
    try:
        _, target, copy = await _seed_overlap(memory, context)
        await memory.dispatch(
            "correct",
            {
                **_preview_payload(context, target),
                "content": "The project release channel is revised.",
            },
        )
        await memory.dispatch("maintenance", {})
        target = store.get(target.id)
        assert target is not None
        assert store.list_fact_revisions(target.id)
        store.replace_graph_neighbors(target.id, {copy.id: 0.75})
        store.replace_graph_neighbors(copy.id, {target.id: 0.75})
        assert store.get_graph_edges()
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        before = _database_rows(store)
        original = store._forget_fact_locked

        def fail_after_cleanup(*args: Any, **kwargs: Any) -> bool:
            original(*args, **kwargs)
            raise RuntimeError("injected after auxiliary cleanup")

        monkeypatch.setattr(store, "_forget_fact_locked", fail_after_cleanup)
        with pytest.raises(RuntimeError, match="injected"):
            await memory.dispatch("forget_related", _selection(context, target, preview))
        assert _database_rows(store) == before
    finally:
        store.close()
    reopened = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        assert _database_rows(reopened) == before
    finally:
        reopened.close()
