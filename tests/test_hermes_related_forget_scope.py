"""Hermes reviewed deletion preserves scope boundaries and unselected copies."""

import json

import pytest

from mnemos.adapters.hermes_tools import dispatch_tool
from mnemos.types import MemoryChunk
from tests.test_hermes_related_forget import LocalServiceClient
from tests.test_related_forget import _seed_overlap, _service
from tests.test_related_forget_similarity import _seed


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_field", ["scope_id", "user_profile_id", None])
async def test_provider_rejects_unlisted_selection_without_deleting_target(tmp_path, foreign_field):
    memory, context, _, _ = _service(tmp_path)
    context = context.model_copy(update={"host": "hermes"})
    client = LocalServiceClient(memory)
    store = memory.engine.store
    try:
        _, target, copy = await _seed_overlap(memory, context)
        if foreign_field:
            store.store(
                MemoryChunk(
                    id="foreign-copy",
                    content=copy.content,
                    embedding=copy.embedding,
                    metadata={**copy.metadata, foreign_field: "another-identity"},
                )
            )
        preview = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_preview",
                {"fact_id": target.id, "expected_version": target.version},
            )
        )
        assert "foreign-copy" not in json.dumps(preview)
        assert copy.id in {item["fact_id"] for item in preview["candidates"]}
        result = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_related",
                {
                    "fact_id": target.id,
                    "expected_version": target.version,
                    "preview_digest": preview["preview_digest"],
                    "selected_fact_ids": [copy.id, "foreign-copy"],
                },
            )
        )
        assert result["status"] == "degraded" and result["committed"] is False
        assert store.get(target.id) is not None
        assert store.get(copy.id) is not None
        assert not store.is_suppressed(fact_id=target.id)
        assert not store.is_suppressed(fact_id=copy.id)
        if foreign_field:
            assert store.get("foreign-copy") is not None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_provider_does_not_select_advisory_similarity_copy_by_default(tmp_path):
    memory, context, target, copy, _ = await _seed(tmp_path)
    client = LocalServiceClient(memory)
    try:
        preview = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_preview",
                {"fact_id": target.id, "expected_version": target.version},
            )
        )
        candidate = next(item for item in preview["candidates"] if item["fact_id"] == copy.id)
        assert candidate["match_type"] == "similarity_candidate"
        assert preview["complete"] is True
        assert preview["similarity_search"]["exhaustive"] is False
        result = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_related",
                {
                    "fact_id": target.id,
                    "expected_version": target.version,
                    "preview_digest": preview["preview_digest"],
                },
            )
        )
        assert result["status"] == "committed"
        assert memory.engine.store.get(target.id) is None
        assert memory.engine.store.get(copy.id) is not None
        assert not memory.engine.store.is_suppressed(fact_id=copy.id)
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["mnemos_forget_preview", "mnemos_forget_related"])
@pytest.mark.parametrize(
    "override",
    ["context", "user_profile_id", "project_id", "workspace_id", "allowed_scopes", "profile"],
)
async def test_reviewed_provider_tools_reject_scope_overrides_before_service(
    tmp_path, tool, override
):
    memory, context, _, _ = _service(tmp_path)
    client = LocalServiceClient(memory)
    try:
        args = {"fact_id": "target", "expected_version": 1, override: "foreign"}
        if tool == "mnemos_forget_related":
            args["preview_digest"] = "a" * 64
        result = json.loads(await dispatch_tool(client, context, tool, args))
        assert result["status"] == "rejected"
        assert client.calls == []
    finally:
        memory.engine.store.close()
