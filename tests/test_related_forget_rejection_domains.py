"""Storage-level domains for bounded related-forget rejections."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from mnemos.errors import (
    RELATED_FORGET_PREVIEW_INCOMPLETE,
    RELATED_FORGET_REJECTIONS,
    RELATED_FORGET_SELECTION_CONFLICT,
    RELATED_FORGET_SELECTION_INVALID,
    RelatedForgetPreviewIncomplete,
    RelatedForgetSelectionConflict,
    RelatedForgetSelectionInvalid,
)
from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore
from tests.test_related_forget import _preview_payload, _seed_overlap, _service


@pytest.mark.asyncio
async def test_duplicate_related_forget_ids_are_bounded_invalid_rejections(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    try:
        _assertion, target, copy = await _seed_overlap(memory, context)
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        with pytest.raises(RelatedForgetSelectionInvalid, match="Duplicate selected fact IDs"):
            await memory.dispatch(
                "forget_related",
                {
                    **_preview_payload(context, target),
                    "preview_digest": preview["preview_digest"],
                    "selected_fact_ids": [copy.id, copy.id],
                    "selected_source_event_ids": [],
                },
            )
    finally:
        cast(SQLiteStore, memory.engine.store).close()


@pytest.mark.asyncio
async def test_incomplete_preview_is_a_bounded_rejection(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    try:
        assertion, target, _copy = await _seed_overlap(memory, context)
        memory.engine.store.store(
            MemoryChunk(
                id="bounded-preview-extra-copy",
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
        with pytest.raises(RelatedForgetPreviewIncomplete, match="incomplete"):
            await memory.dispatch(
                "forget_related",
                {
                    **_preview_payload(context, target, preview_limit=1),
                    "preview_digest": preview["preview_digest"],
                    "selected_fact_ids": [],
                    "selected_source_event_ids": [],
                },
            )
    finally:
        cast(SQLiteStore, memory.engine.store).close()


@pytest.mark.asyncio
async def test_shared_source_dependency_is_a_selection_conflict(tmp_path: Path) -> None:
    memory, context, _registry, _root = _service(tmp_path)
    try:
        _assertion, target, captured = await _seed_overlap(memory, context)
        initial = await memory.dispatch("forget_preview", _preview_payload(context, target))
        candidate = next(item for item in initial["candidates"] if item["fact_id"] == captured.id)
        source_id = candidate["source_event_ids"][0]
        manual = MemoryChunk(
            id="unselected-private-fact",
            content="Manual note retained independently.",
            metadata={
                "user_profile_id": context.user_profile_id,
                "scope": "project",
                "scope_id": context.project_id,
                "source": "manual",
            },
        )
        memory.engine.store.store(manual)
        store = cast(SQLiteStore, memory.engine.store)
        store._conn.execute(  # noqa: SLF001
            "INSERT INTO fact_evidence "
            "(evidence_id, fact_id, source_event_id, excerpt, source_ref, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "unselected-private-evidence",
                manual.id,
                source_id,
                manual.content,
                "manual",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        store._conn.commit()  # noqa: SLF001
        preview = await memory.dispatch("forget_preview", _preview_payload(context, target))
        with pytest.raises(RelatedForgetSelectionConflict, match="unselected derived fact"):
            await memory.dispatch(
                "forget_related",
                {
                    **_preview_payload(context, target),
                    "preview_digest": preview["preview_digest"],
                    "selected_fact_ids": [captured.id],
                    "selected_source_event_ids": [source_id],
                },
            )
    finally:
        store.close()


def test_rejection_map_contains_only_safe_messages() -> None:
    assert RELATED_FORGET_REJECTIONS[RELATED_FORGET_PREVIEW_INCOMPLETE].startswith(
        "The related-forget preview is incomplete."
    )
    assert RELATED_FORGET_REJECTIONS[RELATED_FORGET_SELECTION_CONFLICT].startswith(
        "The reviewed selection conflicts"
    )
    assert RELATED_FORGET_SELECTION_INVALID in RELATED_FORGET_REJECTIONS
    assert str(RelatedForgetSelectionConflict("private detail")) == "private detail"
