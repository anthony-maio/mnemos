"""Reviewed deletion of paraphrased captures, using controlled stored vectors."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore, normalize_memory_text
from tests.test_related_forget import _preview_payload, _service


async def _seed(tmp_path: Path) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    memory, context, _, _ = _service(tmp_path)
    await memory.dispatch(
        "remember",
        {
            "context": context.model_dump(mode="json"),
            "event_id": "similarity-target",
            "content": "Use maple-heron as the release label convention for this project.",
        },
    )
    await memory.dispatch("maintenance", {})
    target = memory.engine.store.get_all()[0]
    payload = {
        "context": context.model_dump(mode="json"),
        "event_id": "similarity-captured-reply",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "role": "assistant",
        "content": "The current project continuity convention is to use maple-heron for release labels.",
        "source_kind": "codex_hook",
    }
    await memory.dispatch("capture", payload)
    await memory.dispatch("maintenance", {})
    copy = next(item for item in memory.engine.store.get_all() if item.id != target.id)
    assert normalize_memory_text(target.content) not in normalize_memory_text(copy.content)
    size = len(target.embedding)
    target.embedding = [1.0] + [0.0] * (size - 1)
    copy.embedding = [0.9, 0.1] + [0.0] * (size - 2)
    memory.engine.store.update(target.id, target)
    memory.engine.store.update(copy.id, copy)
    return memory, context, target, copy, payload


def _apply(context: Any, target: Any, preview: Any, copy: Any = None) -> dict[str, Any]:
    candidate = next(
        (item for item in preview["candidates"] if copy and item["fact_id"] == copy.id), None
    )
    return {
        **_preview_payload(context, target),
        "preview_digest": preview["preview_digest"],
        "preview_limit": preview["limit"],
        "selected_fact_ids": [copy.id] if copy else [],
        "selected_source_event_ids": candidate["required_source_event_ids"] if candidate else [],
    }


@pytest.mark.asyncio
async def test_paraphrase_can_be_reviewed_deleted_and_replay_suppressed(tmp_path: Path) -> None:
    memory, context, target, copy, payload = await _seed(tmp_path)
    store = memory.engine.store
    try:
        before = store._conn.total_changes
        facts_before = [item.model_dump() for item in store.get_all()]
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        assert store._conn.total_changes == before
        assert [item.model_dump() for item in store.get_all()] == facts_before
        candidate = next(item for item in preview["candidates"] if item["fact_id"] == copy.id)
        assert candidate["match_type"] == "similarity_candidate"
        assert candidate["matched_assertions"] == []
        assert candidate["similarity"] > 0.9
        assert candidate["required_source_event_ids"]
        assert preview["similarity_search"]["exhaustive"] is False
        await memory.dispatch("forget_related", _apply(context, target, preview, copy))
        assert store.get(copy.id) is None
        result = await memory.dispatch("capture", payload)
        assert result["processing_status"] == "suppressed"
        await memory.dispatch("maintenance", {})
    finally:
        store.close()
    reopened = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        assert reopened.get(target.id) is None
        assert reopened.get(copy.id) is None
        assert reopened.is_suppressed(fact_id=copy.id)
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_similarity_suggestion_is_never_automatically_selected(tmp_path: Path) -> None:
    memory, context, target, copy, _ = await _seed(tmp_path)
    try:
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        assert copy.id in {item["fact_id"] for item in preview["candidates"]}
        await memory.dispatch("forget_related", _apply(context, target, preview))
        assert memory.engine.store.get(copy.id) is not None
        assert not memory.engine.store.is_suppressed(fact_id=copy.id)
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["target_vector", "copy_vector", "content", "version"])
async def test_similarity_preview_rejects_changed_review_state(
    tmp_path: Path, changed: str
) -> None:
    memory, context, target, copy, _ = await _seed(tmp_path)
    store = memory.engine.store
    try:
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        assert copy.id in {item["fact_id"] for item in preview["candidates"]}
        updated = (
            target.model_copy(deep=True)
            if changed == "target_vector"
            else copy.model_copy(deep=True)
        )
        if changed.endswith("vector"):
            # Scaling preserves cosine rank/score but must invalidate the reviewed input.
            updated.embedding = [value * 2 for value in updated.embedding]
        elif changed == "content":
            updated.content += " This is a different assertion."
        else:
            updated.version += 1
        store.update(updated.id, updated)
        with pytest.raises(ValueError, match="stale"):
            await memory.dispatch("forget_related", _apply(context, target, preview, copy))
        assert store.get(target.id) is not None and store.get(copy.id) is not None
        assert not store.is_suppressed(fact_id=target.id)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_similarity_filters_scope_and_manual_facts_before_ranking(tmp_path: Path) -> None:
    memory, context, target, copy, _ = await _seed(tmp_path)
    try:
        for index, override in enumerate(
            [
                {"scope_id": "foreign-project"},
                {"user_profile_id": "foreign-profile"},
                {"scope": "global"},
                {"source": "manual"},
            ]
        ):
            for number in range(22):
                memory.engine.store.store(
                    MemoryChunk(
                        id=f"foreign-{index}-{number}",
                        content="A closer but ineligible memory.",
                        embedding=target.embedding,
                        metadata={**copy.metadata, **override},
                    )
                )
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        assert [item["fact_id"] for item in preview["candidates"]] == [copy.id]
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_similarity_bound_is_disclosed_and_exact_matches_keep_priority(
    tmp_path: Path,
) -> None:
    memory, context, target, copy, _ = await _seed(tmp_path)
    store = memory.engine.store
    try:
        for number in range(25):
            store.store(
                MemoryChunk(
                    id=f"similar-{number:02d}",
                    content=f"Possible paraphrase {number}.",
                    embedding=copy.embedding,
                    metadata=copy.metadata,
                )
            )
        store.store(
            MemoryChunk(
                id="z-exact",
                content="Captured: " + target.content,
                metadata=copy.metadata,
            )
        )
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        assert preview["candidates"][0]["fact_id"] == "z-exact"
        assert len(preview["candidates"]) == 21
        assert [item["fact_id"] for item in preview["candidates"][1:]] == sorted(
            [copy.id] + [f"similar-{number:02d}" for number in range(25)]
        )[:20]
        assert preview["similarity_search"]["truncated"] is True
        assert preview["similarity_search"]["limit"] == 20
        assert preview["similarity_search"]["exhaustive"] is False
        assert preview["complete"] is True  # The returned selections and dependencies fit.
        small = await memory.dispatch("forget_preview", _preview_payload(context, target, limit=1))
        assert small["complete"] is False
        with pytest.raises(ValueError, match="incomplete"):
            await memory.dispatch("forget_related", _apply(context, target, small))
        assert store.get(target.id) is not None
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("embedding", [None, [0.0] * 384])
async def test_missing_target_vector_keeps_exact_path_and_discloses_gap(
    tmp_path: Path, embedding: Any
) -> None:
    memory, context, target, copy, _ = await _seed(tmp_path)
    store = memory.engine.store
    try:
        target.embedding = embedding
        store.update(target.id, target)
        store.store(
            MemoryChunk(
                id="exact-no-vector",
                content="Captured: " + target.content,
                metadata=copy.metadata,
            )
        )
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        assert [item["fact_id"] for item in preview["candidates"]] == ["exact-no-vector"]
        assert preview["similarity_search"]["status"] == "unavailable"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_similarity_requires_reviewed_sources_and_rejects_changed_dependencies(
    tmp_path: Path,
) -> None:
    memory, context, target, copy, _ = await _seed(tmp_path)
    store = memory.engine.store
    try:
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        request = _apply(context, target, preview, copy)
        with pytest.raises(ValueError, match="explicit source selection"):
            await memory.dispatch("forget_related", {**request, "selected_source_event_ids": []})
        assert store.get(target.id) is not None and store.get(copy.id) is not None
        source_id = request["selected_source_event_ids"][0]
        store.store(
            MemoryChunk(
                id="shared-manual",
                content="A separate retained preference.",
                metadata={**copy.metadata, "source": "manual"},
            )
        )
        store._conn.execute(
            "INSERT INTO fact_evidence "
            "(evidence_id, fact_id, source_event_id, excerpt, source_ref, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "shared-similarity-evidence",
                "shared-manual",
                source_id,
                "",
                "fixture",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        store._conn.commit()
        with pytest.raises(ValueError, match="stale"):
            await memory.dispatch("forget_related", request)
        fresh = await memory.dispatch("forget_preview", _preview_payload(context, target))
        with pytest.raises(ValueError, match="unselected derived fact"):
            await memory.dispatch("forget_related", _apply(context, target, fresh, copy))
        assert store.get(target.id) is not None and store.get(copy.id) is not None
        assert store.get("shared-manual") is not None
        assert store.get_episodic_event(source_id) is not None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_similarity_excludes_unusable_vectors_and_reports_skips(tmp_path: Path) -> None:
    memory, context, target, copy, _ = await _seed(tmp_path)
    store = memory.engine.store
    try:
        for name, embedding in [
            ("missing", None),
            ("zero", [0.0] * len(target.embedding)),
            ("opposite", [-value for value in target.embedding]),
        ]:
            store.store(
                MemoryChunk(
                    id=name,
                    content=f"Capture with {name} vector.",
                    embedding=embedding,
                    metadata=copy.metadata,
                )
            )
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        assert [item["fact_id"] for item in preview["candidates"]] == [copy.id]
        assert preview["similarity_search"]["unsearchable_fact_count"] == 2
    finally:
        store.close()


@pytest.mark.asyncio
async def test_similarity_foreign_linked_source_is_redacted_and_blocks_deletion(
    tmp_path: Path,
) -> None:
    memory, context, target, copy, _ = await _seed(tmp_path)
    store = memory.engine.store
    try:
        original = await memory.dispatch("forget_preview", _preview_payload(context, target))
        source_id = _apply(context, target, original, copy)["selected_source_event_ids"][0]
        store._conn.execute(
            "UPDATE episodic_events SET scope_id=?, content=? WHERE id=?",
            ("foreign-project", "Private foreign-source canary.", source_id),
        )
        store._conn.commit()
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        candidate = next(item for item in preview["candidates"] if item["fact_id"] == copy.id)
        assert candidate["blocked_source_event_ids"] == [source_id]
        source = next(
            item for item in preview["source_events"] if item["source_event_id"] == source_id
        )
        assert source["content"] == ""
        assert "Private foreign-source canary" not in str(preview)
        with pytest.raises(ValueError, match="unavailable linked source"):
            await memory.dispatch("forget_related", _apply(context, target, preview, copy))
        assert store.get(target.id) is not None and store.get(copy.id) is not None
    finally:
        store.close()
