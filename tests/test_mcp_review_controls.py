"""MCP review tools use resolved host scope and strict item identities."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mnemos.client import MnemosClient
from mnemos.mcp_server import ServiceMCPContext, _create_service_mcp_server
from tests.test_source_review_controls import create_review


@pytest.mark.asyncio
async def test_mcp_review_roundtrip_uses_resolved_context_and_never_queues(tmp_path: Path) -> None:
    memory, scope, store, job, _ = await create_review(tmp_path)
    calls: list[tuple[str, dict[str, Any]]] = []

    class Client:
        async def request(
            self, operation: str, payload: dict[str, Any], *, queue_on_failure: bool
        ) -> dict[str, Any]:
            assert queue_on_failure is False
            assert payload["context"] == scope.model_dump(mode="json")
            calls.append((operation, payload))
            return await memory.dispatch(operation, payload)

    context = SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=ServiceMCPContext(client=cast(MnemosClient, Client()), scope=scope)
        )
    )
    server = _create_service_mcp_server()

    async def run(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = server._tool_manager.get_tool(name)
        assert tool is not None
        content, _ = await tool.run(arguments, context=context, convert_result=True)
        return cast(dict[str, Any], json.loads(content[0].text))

    try:
        rows = (await run("mnemos_reviews", {}))["reviews"]
        assert rows[0]["job_id"] == job.id
        item = {"job_id": job.id, "candidate_index": 0}
        inspected = await run("mnemos_inspect_review", item)
        assert (
            inspected["review"]["excerpt"] == "The marker changed, but no replacement was approved."
        )
        assert (await run("mnemos_discard_review", item))["status"] == "discarded"
        assert (await run("mnemos_discard_review", item))["status"] == "unchanged"
        assert (await run("mnemos_inspect_review", item))["review"]["excerpt"] is None
        count = len(calls)
        with pytest.raises(Exception):
            await run("mnemos_discard_review", {"job_id": job.id, "candidate_index": True})
        assert len(calls) == count
        assert store.get_all() == []
    finally:
        store.close()
