import json
from datetime import datetime, timezone
from typing import Any

import pytest

from mnemos.adapters.hermes_tools import dispatch_tool
from tests.test_related_forget import _seed_overlap, _service


def selection(context, target, preview):
    return {
        "fact_id": target.id,
        "expected_version": target.version,
        "preview_digest": preview["preview_digest"],
        "preview_limit": preview["limit"],
        "selected_fact_ids": [target.id] + [item["fact_id"] for item in preview["candidates"]],
        "selected_source_event_ids": [item["source_event_id"] for item in preview["source_events"]],
    }


class LocalServiceClient:
    def __init__(self, memory: Any):
        self.memory = memory
        self.calls: list[tuple[str, dict[str, Any], bool]] = []

    async def request(self, operation, payload, *, queue_on_failure=True):
        self.calls.append((operation, payload, queue_on_failure))
        return await self.memory.dispatch(operation, payload)


@pytest.mark.asyncio
async def test_hermes_provider_can_preview_and_apply_reviewed_copies(tmp_path):
    memory, context, _, _ = _service(tmp_path)
    client = LocalServiceClient(memory)
    try:
        _, target, copy = await _seed_overlap(memory, context)
        preview = await dispatch_tool(
            client,
            context,
            "mnemos_forget_preview",
            {"fact_id": target.id, "expected_version": target.version},
        )
        preview_payload = json.loads(preview)
        assert preview_payload["target"]["fact_id"] == target.id
        assert preview_payload["complete"] is True
        assert any(item["fact_id"] == copy.id for item in preview_payload["candidates"])

        applied = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_related",
                selection(context, target, preview_payload),
            )
        )
        assert applied["status"] == "committed"
        assert target.id in applied["selected_fact_ids"]
        assert memory.engine.store.get(target.id) is None
        assert memory.engine.store.get(copy.id) is None
        assert all(queued is False for _, _, queued in client.calls)
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_hermes_related_forget_defaults_are_target_only(tmp_path):
    memory, context, _, _ = _service(tmp_path)
    client = LocalServiceClient(memory)
    try:
        _, target, copy = await _seed_overlap(memory, context)
        preview = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_preview",
                {"fact_id": target.id, "expected_version": target.version},
            )
        )
        target_only = json.loads(
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
        assert target_only["status"] == "committed"
        assert memory.engine.store.get(target.id) is None
        assert memory.engine.store.get(copy.id) is not None

    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_hermes_related_forget_stale_preview_is_atomic(tmp_path):
    memory, context, _, _ = _service(tmp_path)
    client = LocalServiceClient(memory)
    try:
        assertion, target, copy = await _seed_overlap(memory, context)
        preview = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_preview",
                {"fact_id": target.id, "expected_version": target.version},
            )
        )
        await memory.dispatch(
            "capture",
            {
                "context": context.model_dump(mode="json"),
                "event_id": "hermes-new-after-preview",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "role": "user",
                "content": f"A new capture contains: {assertion}",
                "source_kind": "hermes_memory_provider",
            },
        )
        await memory.dispatch("maintenance", {})
        stale = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_related",
                {
                    "fact_id": target.id,
                    "expected_version": target.version,
                    "preview_digest": preview["preview_digest"],
                    "preview_limit": preview["limit"],
                    "selected_fact_ids": [copy.id],
                },
            )
        )
        assert stale["status"] == "degraded"
        assert memory.engine.store.get(target.id) is not None
        assert memory.engine.store.get(copy.id) is not None
    finally:
        memory.engine.store.close()
