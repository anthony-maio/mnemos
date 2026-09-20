"""Copyable preview bindings must retain all existing apply protections."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from mnemos.adapters.hermes_tools import dispatch_tool
from tests.test_hermes_related_forget import LocalServiceClient
from tests.test_related_forget import _seed_overlap, _service


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [100, 37])
async def test_copy_preview_preconditions_into_real_hermes_apply(
    tmp_path: Path, limit: int
) -> None:
    memory, context, _, _ = _service(tmp_path)
    client = LocalServiceClient(memory)
    try:
        _, target, captured = await _seed_overlap(memory, context)
        preview = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_preview",
                {
                    "fact_id": target.id,
                    "expected_version": target.version,
                    "limit": limit,
                },
            )
        )
        preconditions = preview["apply_preconditions"]
        assert preconditions == {
            "fact_id": target.id,
            "expected_version": target.version,
            "preview_digest": preview["preview_digest"],
            "preview_limit": limit,
        }
        args = {
            **preconditions,
            "selected_fact_ids": [captured.id],
            "selected_source_event_ids": [
                source["source_event_id"]
                for source in preview["source_events"]
                if source.get("selection_allowed") is True
            ],
        }
        applied = json.loads(await dispatch_tool(client, context, "mnemos_forget_related", args))
        assert applied["status"] == "committed"
        assert memory.engine.store.get(target.id) is None
        assert memory.engine.store.get(captured.id) is None
        assert applied["deletion_coverage"]["retained_eligible_source_event_count"] == 0
        assert all(
            memory.engine.store.get_episodic_event(source_id) is None
            for source_id in args["selected_source_event_ids"]
        )
        assert (
            memory.engine.store._conn.execute(
                "SELECT COUNT(*) FROM memory_suppressions"
            ).fetchone()[0]
            > 0
        )
        assert all(not queued for _, _, queued in client.calls)
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_incomplete_preview_has_no_copyable_apply_preconditions(tmp_path: Path) -> None:
    memory, context, _, _ = _service(tmp_path)
    try:
        assertion, target, _ = await _seed_overlap(memory, context)
        for number in range(2):
            await memory.dispatch(
                "capture",
                {
                    "context": context.model_dump(mode="json"),
                    "event_id": f"extra-copy-{number}",
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                    "role": "user",
                    "content": f"Please remember: {assertion} Copy {number}",
                    "source_kind": "claude_code_transcript",
                },
            )
        preview = await memory.dispatch(
            "forget_preview",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
                "limit": 1,
            },
        )
        assert preview["complete"] is False
        assert preview["apply_preconditions"] is None
        assert memory.engine.store.get(target.id) is not None
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["changed_limit", "new_capture"])
async def test_copied_preconditions_do_not_bypass_stale_preview_checks(
    tmp_path: Path, mutation: str
) -> None:
    memory, context, _, _ = _service(tmp_path)
    client = LocalServiceClient(memory)
    try:
        assertion, target, captured = await _seed_overlap(memory, context)
        preview = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_preview",
                {
                    "fact_id": target.id,
                    "expected_version": target.version,
                },
            )
        )
        args = {
            **preview["apply_preconditions"],
            "selected_fact_ids": [captured.id],
            "selected_source_event_ids": [
                source["source_event_id"]
                for source in preview["source_events"]
                if source.get("selection_allowed") is True
            ],
        }
        if mutation == "changed_limit":
            args["preview_limit"] = 20
        else:
            await memory.dispatch(
                "capture",
                {
                    "context": context.model_dump(mode="json"),
                    "event_id": "after-preview",
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                    "role": "user",
                    "content": f"Please remember: {assertion}",
                    "source_kind": "claude_code_transcript",
                },
            )
        applied = json.loads(await dispatch_tool(client, context, "mnemos_forget_related", args))
        assert applied["status"] == "degraded"
        assert memory.engine.store.get(target.id) is not None
        assert memory.engine.store.get(captured.id) is not None
        assert (
            memory.engine.store._conn.execute(
                "SELECT COUNT(*) FROM memory_suppressions"
            ).fetchone()[0]
            == 0
        )
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_advertised_mcp_preview_description_agrees_with_read_only_annotation() -> None:
    from mnemos.mcp_server import _create_service_mcp_server

    server = _create_service_mcp_server()
    tool = next(tool for tool in await server.list_tools() if tool.name == "mnemos_forget_preview")
    assert tool.annotations.readOnlyHint is True
    assert "read-only" in tool.description.lower()
    assert "apply_preconditions" in tool.description
