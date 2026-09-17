"""Previewed exact-assertion forget regressions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.contracts import ForgetPreviewRequest
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore, normalize_memory_text


def _service(
    tmp_path: Path, registry: ScopeRegistry | None = None
) -> tuple[LocalMemoryService, Any, ScopeRegistry, Path]:
    if registry is None:
        registry = ScopeRegistry()
        registry.register_project(tmp_path, user_profile_id="profile-a", enabled=True)
    context = registry.resolve(
        tmp_path,
        user_profile_id="profile-a",
        session_id="related-forget-session",
        host="claude-code",
    )
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    config = MnemosConfig(
        modules=ModuleConfig(
            surprisal=False,
            mutable_rag=False,
            affective=False,
            sleep=False,
            spreading=False,
        )
    )
    engine = MnemosEngine(
        config=config,
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(),
        store=store,
        scope_registry=registry,
        require_scope_context=True,
    )
    return LocalMemoryService(engine, registry), context, registry, tmp_path


async def _seed_overlap(memory: LocalMemoryService, context: Any) -> tuple[str, Any, Any]:
    assertion = "The project release channel is related-forget-old."
    prompt = f"Please remember: {assertion} Use Mnemos tools."
    payload_context = context.model_dump(mode="json")
    captured = await memory.dispatch(
        "capture",
        {
            "event_id": "captured-overlap-event",
            "context": payload_context,
            "scope": "project",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "role": "user",
            "content": prompt,
            "source_kind": "claude_code_transcript",
            "source_ref": "related-forget-transcript#1",
        },
    )
    await memory.dispatch("maintenance", {})
    remembered = await memory.dispatch(
        "remember",
        {
            "event_id": "explicit-overlap-event",
            "context": payload_context,
            "content": assertion,
        },
    )
    await memory.dispatch("maintenance", {})
    target = memory.engine.store.get_all()
    target_chunk = next(item for item in target if item.content == assertion)
    captured_chunk = next(item for item in target if item.content == prompt)
    return assertion, target_chunk, captured_chunk


def _preview_payload(context: Any, target: Any, **extra: Any) -> dict[str, Any]:
    return {
        "context": context.model_dump(mode="json"),
        "fact_id": target.id,
        "expected_version": target.version,
        **extra,
    }


def test_related_forget_request_rejects_coerced_scalars() -> None:
    context = {
        "user_profile_id": "profile-a",
        "project_id": "project-a",
        "workspace_id": None,
        "session_id": "session-a",
        "host": "test",
    }
    with pytest.raises(ValueError):
        ForgetPreviewRequest.model_validate(
            {
                "context": context,
                "fact_id": " fact-a ",
                "expected_version": True,
            }
        )


@pytest.mark.asyncio
async def test_historical_assertion_is_previewed_after_correction(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    try:
        old_assertion, target, captured = await _seed_overlap(memory, context)
        corrected = "The project release channel is related-forget-new."
        await memory.dispatch(
            "correct",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
                "content": corrected,
                "event_id": "related-forget-correction",
            },
        )
        await memory.dispatch("maintenance", {})
        current = memory.engine.store.get(target.id)
        assert current is not None and current.content == corrected
        preview = await memory.dispatch("forget_preview", _preview_payload(context, current))
        candidate = next(item for item in preview["candidates"] if item["fact_id"] == captured.id)
        assert old_assertion.lower() in candidate["matched_assertions"]
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_source_event_overflow_is_incomplete(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    try:
        assertion = "The project release channel is source-overflow-related-forget."
        payload_context = context.model_dump(mode="json")
        await memory.dispatch(
            "remember",
            {
                "event_id": "source-overflow-target",
                "context": payload_context,
                "content": assertion,
            },
        )
        await memory.dispatch("maintenance", {})
        target = memory.engine.store.get_all()[0]
        for index in range(2):
            await memory.dispatch(
                "capture",
                {
                    "event_id": f"source-overflow-{index}",
                    "context": payload_context,
                    "scope": "project",
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                    "role": "user",
                    "content": f"Please remember: {assertion} #{index}",
                    "source_kind": "claude_code_transcript",
                    "source_ref": f"source-overflow#{index}",
                },
            )
        preview = await memory.dispatch(
            "forget_preview", _preview_payload(context, target, limit=1)
        )
        assert preview["complete"] is False
        assert len(preview["source_events"]) == 1
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_shared_source_requires_reviewing_unselected_fact(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    try:
        _assertion, target, captured = await _seed_overlap(memory, context)
        match_preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        candidate = next(
            item for item in match_preview["candidates"] if item["fact_id"] == captured.id
        )
        source_id = candidate["source_event_ids"][0]
        manual = MemoryChunk(
            id="manual-shared-source",
            content="Manual note retained independently.",
            metadata={
                "user_profile_id": context.user_profile_id,
                "scope": "project",
                "scope_id": context.project_id,
                "source": "manual",
            },
        )
        memory.engine.store.store(manual)
        memory.engine.store._conn.execute(  # noqa: SLF001
            "INSERT INTO fact_evidence "
            "(evidence_id, fact_id, source_event_id, excerpt, source_ref, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "manual-shared-evidence",
                manual.id,
                source_id,
                "Manual note retained independently.",
                "manual",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        memory.engine.store._conn.commit()  # noqa: SLF001
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        source = next(
            item for item in preview["source_events"] if item["source_event_id"] == source_id
        )
        assert manual.id in source["derived_fact_ids"]
        with pytest.raises(ValueError, match="unselected derived fact"):
            await memory.dispatch(
                "forget_related",
                {
                    **_preview_payload(context, target),
                    "preview_digest": preview["preview_digest"],
                    "selected_fact_ids": [captured.id],
                    "selected_source_event_ids": [source_id],
                },
            )
        assert memory.engine.store.get(target.id) is not None
        assert memory.engine.store.get(captured.id) is not None
        assert memory.engine.store.get(manual.id) is not None
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_shared_source_only_referenced_by_jobs_is_rejected(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    store = memory.engine.store
    try:
        assertion, target, captured = await _seed_overlap(memory, context)
        payload_context = context.model_dump(mode="json")
        first_capture = await memory.dispatch(
            "capture",
            {
                "event_id": "job-only-shared-source",
                "context": payload_context,
                "scope": "project",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "role": "user",
                "content": f"A captured copy contains: {assertion}",
                "source_kind": "claude_code_transcript",
                "source_ref": "job-only-shared#1",
            },
        )
        manual = MemoryChunk(
            id="job-only-independent-fact",
            content="Manual fact sharing a pending source.",
            metadata={
                "user_profile_id": context.user_profile_id,
                "scope": "project",
                "scope_id": context.project_id,
                "source": "manual",
            },
        )
        store.store(manual)
        second_capture = await memory.dispatch(
            "capture",
            {
                "event_id": "job-only-second-job",
                "context": payload_context,
                "scope": "project",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "role": "user",
                "content": "A second pending job has no durable fact.",
                "source_kind": "claude_code_transcript",
                "source_ref": "job-only-shared#2",
            },
        )
        first_job = store.get_job(first_capture["job_id"])
        second_job = store.get_job(second_capture["job_id"])
        assert first_job is not None and second_job is not None
        source_id = first_job.payload["event_id"]
        assert source_id != second_job.payload["event_id"]
        store._conn.execute(  # noqa: SLF001
            "UPDATE memory_jobs SET payload=json_set(payload, '$.target_fact_id', ?) WHERE id=?",
            (captured.id, first_job.id),
        )
        store._conn.execute(  # noqa: SLF001
            "UPDATE memory_jobs SET payload=json_set(payload, '$.target_fact_id', ?, '$.event_id', ?) WHERE id=?",
            (manual.id, source_id, second_job.id),
        )
        store._conn.commit()  # noqa: SLF001
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        source = next(
            item for item in preview["source_events"] if item["source_event_id"] == source_id
        )
        assert captured.id in source["derived_fact_ids"]
        assert manual.id in source["derived_fact_ids"]
        candidate = next(item for item in preview["candidates"] if item["fact_id"] == captured.id)
        assert source_id in candidate["required_source_event_ids"]
        with pytest.raises(ValueError, match="unselected derived fact"):
            await memory.dispatch(
                "forget_related",
                {
                    **_preview_payload(context, target),
                    "preview_digest": preview["preview_digest"],
                    "selected_fact_ids": [target.id, captured.id],
                    "selected_source_event_ids": candidate["required_source_event_ids"],
                },
            )
        assert store.get(target.id) is not None
        assert store.get(captured.id) is not None
        assert store.get(manual.id) is not None
        assert store.get_episodic_event(source_id) is not None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_target_shared_source_is_rejected_before_any_deletion(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    store = memory.engine.store
    try:
        _assertion, target, _captured = await _seed_overlap(memory, context)
        target_evidence = store.list_fact_evidence(target.id)
        source_id = next(item.source_event_id for item in target_evidence if item.source_event_id)
        manual = MemoryChunk(
            id="manual-target-source",
            content="An independent manual note.",
            metadata={
                "user_profile_id": context.user_profile_id,
                "scope": "project",
                "scope_id": context.project_id,
                "source": "manual",
            },
        )
        store.store(manual)
        store._conn.execute(  # noqa: SLF001
            "INSERT INTO fact_evidence "
            "(evidence_id, fact_id, source_event_id, excerpt, source_ref, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "target-shared-evidence",
                manual.id,
                source_id,
                manual.content,
                "manual",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        store._conn.commit()  # noqa: SLF001
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        assert source_id in preview["target"]["source_event_ids"]
        with pytest.raises(ValueError, match="Target cleanup"):
            await memory.dispatch(
                "forget_related",
                {**_preview_payload(context, target), "preview_digest": preview["preview_digest"]},
            )
        assert store.get(target.id) is not None
        assert store.get(manual.id) is not None
        assert store.get_episodic_event(source_id) is not None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_expired_source_is_suppressed_without_selection(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    store = memory.engine.store
    try:
        _assertion, target, captured = await _seed_overlap(memory, context)
        candidate_preview = await memory.dispatch(
            "forget_preview", _preview_payload(context, target)
        )
        candidate = next(
            item for item in candidate_preview["candidates"] if item["fact_id"] == captured.id
        )
        expired_source = candidate["source_event_ids"][0]
        store._conn.execute(
            "DELETE FROM episodic_events WHERE id=?", (expired_source,)
        )  # noqa: SLF001
        store._conn.commit()  # noqa: SLF001
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        candidate = next(item for item in preview["candidates"] if item["fact_id"] == captured.id)
        assert candidate["expired_source_event_ids"] == [expired_source]
        await memory.dispatch(
            "forget_related",
            {
                **_preview_payload(context, target),
                "preview_digest": preview["preview_digest"],
                "selected_fact_ids": [target.id, captured.id],
                "selected_source_event_ids": [],
            },
        )
        assert store.get(captured.id) is None
        assert store.is_suppressed(source_event_id=expired_source)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_linked_nonmatching_capture_source_is_visible_and_selectable(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    store = memory.engine.store
    try:
        assertion, target, _captured = await _seed_overlap(memory, context)
        source_id = "linked-nonmatching-source"
        store.append_episodic_event(
            EpisodicEvent(
                id=source_id,
                event_fingerprint=source_id,
                role="user",
                content="The transcript line omits the durable assertion.",
                normalized_content=normalize_memory_text(
                    "The transcript line omits the durable assertion."
                ),
                scope="project",
                scope_id=context.project_id,
                source="claude-code",
                metadata={
                    "user_profile_id": context.user_profile_id,
                    "explicit_remember": False,
                },
            )
        )
        linked = MemoryChunk(
            id="linked-nonmatching-copy",
            content=f"Captured fact: {assertion}",
            metadata={
                "user_profile_id": context.user_profile_id,
                "scope": "project",
                "scope_id": context.project_id,
                "source": "sleep_consolidation",
            },
        )
        store.store(linked)
        store._conn.execute(  # noqa: SLF001
            "INSERT INTO fact_evidence "
            "(evidence_id, fact_id, source_event_id, excerpt, source_ref, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "linked-nonmatching-evidence",
                linked.id,
                source_id,
                linked.content,
                "transcript",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        store._conn.commit()  # noqa: SLF001
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        source = next(
            item for item in preview["source_events"] if item["source_event_id"] == source_id
        )
        assert source["match_type"] == "linked_capture_source"
        assert (
            source_id
            in next(item for item in preview["candidates"] if item["fact_id"] == linked.id)[
                "required_source_event_ids"
            ]
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_linked_source_overflow_is_incomplete_and_disclosed(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    store = memory.engine.store
    try:
        assertion, target, _captured = await _seed_overlap(memory, context)
        store.append_episodic_event(
            EpisodicEvent(
                id="overflow-direct-source",
                event_fingerprint="overflow-direct-source",
                role="user",
                content=f"Another capture contains: {assertion}",
                normalized_content=normalize_memory_text(f"Another capture contains: {assertion}"),
                scope="project",
                scope_id=context.project_id,
                source="claude-code",
                metadata={
                    "user_profile_id": context.user_profile_id,
                    "explicit_remember": False,
                },
            )
        )
        source_id = "linked-overflow-source"
        store.append_episodic_event(
            EpisodicEvent(
                id=source_id,
                event_fingerprint=source_id,
                role="user",
                content="A transcript line without the durable assertion.",
                normalized_content="a transcript line without the durable assertion.",
                scope="project",
                scope_id=context.project_id,
                source="claude-code",
                metadata={
                    "user_profile_id": context.user_profile_id,
                    "explicit_remember": False,
                },
            )
        )
        linked = MemoryChunk(
            id="linked-overflow-copy",
            content=f"Captured fact: {assertion}",
            metadata={
                "user_profile_id": context.user_profile_id,
                "scope": "project",
                "scope_id": context.project_id,
                "source": "sleep_consolidation",
            },
        )
        store.store(linked)
        store._conn.execute(  # noqa: SLF001
            "INSERT INTO fact_evidence "
            "(evidence_id, fact_id, source_event_id, excerpt, source_ref, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "linked-overflow-evidence",
                linked.id,
                source_id,
                linked.content,
                "transcript",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        store._conn.commit()  # noqa: SLF001
        preview = await memory.dispatch(
            "forget_preview", _preview_payload(context, target, limit=2)
        )
        candidate = next(item for item in preview["candidates"] if item["fact_id"] == linked.id)
        assert preview["complete"] is False
        assert candidate["unlisted_source_event_ids"] == [source_id]
        assert source_id not in {item["source_event_id"] for item in preview["source_events"]}
    finally:
        store.close()


@pytest.mark.asyncio
async def test_existing_forget_control_leaves_independent_copy(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    try:
        _assertion, target, captured = await _seed_overlap(memory, context)
        assert memory.engine.store.forget_fact(target.id, expected_version=target.version)
        assert memory.engine.store.get(captured.id) is not None
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_preview_and_related_apply_remove_exact_copy_and_replay(tmp_path: Path) -> None:
    memory, context, registry, root = _service(tmp_path)
    try:
        assertion, target, captured = await _seed_overlap(memory, context)
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        assert preview["status"] == "preview"
        assert preview["complete"] is True
        match = next(item for item in preview["candidates"] if item["fact_id"] == captured.id)
        source_ids = match["source_event_ids"]
        result = await memory.dispatch(
            "forget_related",
            {
                **_preview_payload(context, target),
                "preview_digest": preview["preview_digest"],
                "selected_fact_ids": [target.id, captured.id],
                "selected_source_event_ids": source_ids,
            },
        )
        assert result["status"] == "committed"
        assert memory.engine.store.get(target.id) is None
        assert memory.engine.store.get(captured.id) is None
        retrieved = await memory.dispatch(
            "retrieve",
            {"context": context.model_dump(mode="json"), "query": assertion, "top_k": 10},
        )
        assert retrieved["facts"] == []
        memory.engine.store.close()

        restarted = _service(root, registry)[0]
        try:
            replay = await restarted.dispatch(
                "capture",
                {
                    "event_id": "captured-overlap-event",
                    "context": context.model_dump(mode="json"),
                    "scope": "project",
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                    "role": "user",
                    "content": f"Please remember: {assertion} Use Mnemos tools.",
                    "source_kind": "claude_code_transcript",
                    "source_ref": "related-forget-transcript#1",
                },
            )
            assert replay["processing_status"] == "suppressed"
            assert await restarted.dispatch(
                "retrieve",
                {
                    "context": context.model_dump(mode="json"),
                    "query": assertion,
                    "top_k": 10,
                },
            ) == {
                "schema_version": 1,
                "profile": "baseline",
                "facts": [],
                "context": "",
                "token_count_method": "utf8_bytes_upper_bound",
                "context_token_upper_bound": 0,
            }
        finally:
            restarted.engine.store.close()
    finally:
        try:
            memory.engine.store.close()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_unrelated_and_foreign_scope_records_are_excluded(tmp_path: Path) -> None:
    memory, context, registry, root = _service(tmp_path)
    try:
        assertion, target, _captured = await _seed_overlap(memory, context)
        other_root = root / "other-project"
        other_root.mkdir()
        foreign_project = registry.register_project(
            other_root, user_profile_id="profile-a", enabled=True
        )
        registry.register_profile("profile-b")
        store = memory.engine.store
        store.store(
            MemoryChunk(
                id="foreign-project-copy",
                content=f"Foreign copy: {assertion}",
                metadata={
                    "user_profile_id": "profile-a",
                    "scope": "project",
                    "scope_id": foreign_project.id,
                    "source": "sleep_consolidation",
                },
            )
        )
        store.store(
            MemoryChunk(
                id="foreign-profile-copy",
                content=f"Other profile copy: {assertion}",
                metadata={
                    "user_profile_id": "profile-b",
                    "scope": "project",
                    "scope_id": context.project_id,
                    "source": "sleep_consolidation",
                },
            )
        )
        store.store(
            MemoryChunk(
                id="manual-overlap",
                content=f"Manual note: {assertion}",
                metadata={
                    "user_profile_id": context.user_profile_id,
                    "scope": "project",
                    "scope_id": context.project_id,
                    "source": "manual",
                },
            )
        )
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        candidate_ids = {item["fact_id"] for item in preview["candidates"]}
        assert "foreign-project-copy" not in candidate_ids
        assert "foreign-profile-copy" not in candidate_ids
        assert "manual-overlap" not in candidate_ids
        assert any(item["fact_id"] == _captured.id for item in preview["candidates"])
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_pending_capture_is_previewed_and_suppressed_without_maintenance(
    tmp_path: Path,
) -> None:
    memory, context, _registry, root = _service(tmp_path)
    try:
        assertion = "The project release channel is pending-related-forget."
        payload_context = context.model_dump(mode="json")
        target_result = await memory.dispatch(
            "remember",
            {"event_id": "pending-target", "context": payload_context, "content": assertion},
        )
        await memory.dispatch("maintenance", {})
        target = memory.engine.store.get_all()[0]
        pending_result = await memory.dispatch(
            "capture",
            {
                "event_id": "pending-copy-event",
                "context": payload_context,
                "scope": "project",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "role": "user",
                "content": f"Please remember: {assertion}",
                "source_kind": "claude_code_transcript",
                "source_ref": "pending-transcript#1",
            },
        )
        pending_job = next(
            job for job in memory.engine.store.list_jobs() if job.id == pending_result["job_id"]
        )
        memory.engine.store._conn.execute(  # noqa: SLF001
            "UPDATE memory_jobs SET payload=json_set(payload, '$.content', ?) WHERE id=?",
            (f"raw pending content: {assertion}", pending_job.id),
        )
        memory.engine.store._conn.commit()  # noqa: SLF001
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        pending = next(item for item in preview["source_events"] if item["status"] == "pending")
        assert pending["status"] == "pending"
        result = await memory.dispatch(
            "forget_related",
            {
                **_preview_payload(context, target),
                "preview_digest": preview["preview_digest"],
                "selected_fact_ids": [],
                "selected_source_event_ids": [pending["source_event_id"]],
            },
        )
        assert result["forgotten"] is True
        assert memory.engine.store.get_episodic_event(pending["source_event_id"]) is None
        assert "content" not in memory.engine.store.get_job(pending_job.id).payload
        memory.engine.store.close()
        restarted = _service(root, _registry)[0]
        try:
            maintenance = await restarted.dispatch("maintenance", {})
            assert maintenance["completed"] == 0
            assert restarted.engine.store.get_all() == []
        finally:
            restarted.engine.store.close()
    finally:
        try:
            memory.engine.store.close()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_stale_or_tampered_selection_fails_without_deletion(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    try:
        _assertion, target, captured = await _seed_overlap(memory, context)
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        match = next(item for item in preview["candidates"] if item["fact_id"] == captured.id)
        with pytest.raises(ValueError, match="Selection"):
            await memory.dispatch(
                "forget_related",
                {
                    **_preview_payload(context, target),
                    "preview_digest": preview["preview_digest"],
                    "selected_fact_ids": ["tampered-fact"],
                    "selected_source_event_ids": [],
                },
            )
        assert memory.engine.store.get(target.id) is not None
        updated = captured.model_copy(update={"content": captured.content + " changed"})
        memory.engine.store.store(updated)
        with pytest.raises(ValueError, match="stale"):
            await memory.dispatch(
                "forget_related",
                {
                    **_preview_payload(context, target),
                    "preview_digest": preview["preview_digest"],
                    "selected_fact_ids": [captured.id],
                    "selected_source_event_ids": match["source_event_ids"],
                },
            )
        assert memory.engine.store.get(target.id) is not None
        assert memory.engine.store.get(captured.id) is not None
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_related_apply_rolls_back_target_and_copy_together(
    tmp_path: Path, monkeypatch
) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    try:
        _assertion, target, captured = await _seed_overlap(memory, context)
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        match = next(item for item in preview["candidates"] if item["fact_id"] == captured.id)
        original = memory.engine.store._forget_fact_locked  # noqa: SLF001
        calls = 0

        def fail_after_first(*args: Any, **kwargs: Any) -> bool:
            nonlocal calls
            calls += 1
            result = original(*args, **kwargs)
            if calls == 1:
                raise RuntimeError("injected related forget failure")
            return result

        monkeypatch.setattr(memory.engine.store, "_forget_fact_locked", fail_after_first)
        with pytest.raises(RuntimeError, match="injected"):
            await memory.dispatch(
                "forget_related",
                {
                    **_preview_payload(context, target),
                    "preview_digest": preview["preview_digest"],
                    "selected_fact_ids": [captured.id],
                    "selected_source_event_ids": match["source_event_ids"],
                },
            )
        assert memory.engine.store.get(target.id) is not None
        assert memory.engine.store.get(captured.id) is not None
        assert not memory.engine.store.is_suppressed(fact_id=target.id)
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_preview_is_incomplete_when_match_count_exceeds_bound(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    try:
        assertion, target, _captured = await _seed_overlap(memory, context)
        memory.engine.store.store(
            MemoryChunk(
                id="second-auto-copy",
                content=f"Another capture: {assertion}",
                metadata={
                    "user_profile_id": context.user_profile_id,
                    "scope": "project",
                    "scope_id": context.project_id,
                    "source": "sleep_consolidation",
                },
            )
        )
        preview = await memory.dispatch(
            "forget_preview", _preview_payload(context, target, limit=1)
        )
        assert preview["complete"] is False
        with pytest.raises(ValueError, match="incomplete"):
            await memory.dispatch(
                "forget_related",
                {
                    **_preview_payload(context, target),
                    "preview_digest": preview["preview_digest"],
                    "selected_fact_ids": [],
                    "selected_source_event_ids": [],
                    "preview_limit": preview["limit"],
                },
            )
    finally:
        memory.engine.store.close()
