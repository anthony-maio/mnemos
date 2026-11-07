"""Lossless bounded MCP review and digest-bound short selections."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from mnemos.mcp_server import ServiceMCPContext, _register_service_tools
from mnemos.scopes import ScopeContext


def _preview() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "preview",
        "scope": {"user_profile_id": "profile-a", "project_id": "project-a"},
        "target": {"fact_id": "target", "version": 2, "content": "Release is cedar."},
        "candidates": [
            {
                "fact_id": f"fact-{i}",
                "version": 1,
                "content": ("Recorded release is cedar. " * 1000) + str(i),
                "required_source_event_ids": [str(i).zfill(64)],
                "match_type": "exact_assertion",
            }
            for i in range(3)
        ],
        "source_events": [
            {
                "source_event_id": str(i).zfill(64),
                "content": "Source says cedar. " * 1000,
                "selection_allowed": i != 2,
                "derived_fact_ids": [f"fact-{i}"],
            }
            for i in range(3)
        ],
        "complete": True,
        "limit": 100,
        "preview_digest": "a" * 64,
        "apply_preconditions": {
            "fact_id": "target",
            "expected_version": 2,
            "preview_digest": "a" * 64,
            "preview_limit": 100,
        },
    }


class _Client:
    def __init__(self, preview: dict[str, Any]) -> None:
        self.preview = preview
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def request(self, operation: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.calls.append((operation, deepcopy(payload)))
        if operation == "forget_preview":
            return deepcopy(self.preview)
        return {"status": "committed", "selected_fact_ids": payload["selected_fact_ids"]}


def _tools(client: Any, context: ScopeContext | None = None) -> tuple[dict[str, Any], Any]:
    class Mcp:
        tools: dict[str, Any]

        def __init__(self) -> None:
            self.tools = {}

        def tool(self, **_: Any) -> Any:
            return lambda fn: self.tools.setdefault(fn.__name__, fn)

        def resource(self, _: str) -> Any:
            return lambda fn: fn

    mcp = Mcp()
    _register_service_tools(mcp)
    context = context or ScopeContext(
        user_profile_id="profile-a", project_id="project-a", session_id="review", host="manual"
    )
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context=ServiceMCPContext(client, context))
    )
    return mcp.tools, ctx


def test_large_mcp_preview_stays_inline() -> None:
    tools, ctx = _tools(_Client(_preview()))
    result = asyncio.run(tools["mnemos_forget_preview"]("target", 2, ctx=ctx))
    assert len(json.dumps({"result": result}).encode("utf-8")) <= 12_000
    page = json.loads(result)
    assert page["review"]["next_cursor"]
    assert page["apply_preconditions"] is None


def _put(root: dict[str, Any], path: list[str | int], value: Any) -> None:
    node: Any = root
    for i, key in enumerate(path):
        final = i == len(path) - 1
        if isinstance(node, list):
            assert isinstance(key, int)
            while len(node) <= key:
                node.append(None)
            if final:
                node[key] = value
                return
            if node[key] is None:
                node[key] = [] if isinstance(path[i + 1], int) else {}
        else:
            if final:
                node[key] = value
                return
            node.setdefault(key, [] if isinstance(path[i + 1], int) else {})
        node = node[key]


async def _collect(tools: dict[str, Any], ctx: Any) -> tuple[dict[str, Any], list[Any]]:
    cursor = None
    recovered: dict[str, Any] = {}
    strings: dict[tuple[Any, ...], str] = {}
    pages = []
    next_fragment = 0
    while True:
        text = await tools["mnemos_forget_preview"]("target", 2, cursor=cursor, ctx=ctx)
        assert len(json.dumps({"result": text}).encode("utf-8")) <= 12_000
        page = json.loads(text)
        pages.append(page)
        assert page["review"]["first_fragment"] == next_fragment
        next_fragment += len(page["fragments"])
        for fragment in page["fragments"]:
            path = fragment["path"]
            if "text" in fragment:
                key = tuple(path)
                prior = strings.get(key, "")
                assert fragment["offset"] == len(prior)
                strings[key] = prior + fragment["text"]
                if len(strings[key]) == fragment["total_characters"]:
                    _put(recovered, path, strings[key])
            else:
                _put(recovered, path, fragment["value"])
        cursor = page["review"]["next_cursor"]
        if cursor is None:
            assert next_fragment == page["review"]["total_fragments"]
            assert page["review"]["last_page"] is True
            return recovered, pages
        assert page["apply_preconditions"] is None
        assert len(pages) < 200


def test_pages_preserve_all_text_and_collections_without_mutating_snapshot() -> None:
    original = _preview()
    original["candidates"][0]["content"] = ('\u96ea\U0001f9e0\\"\n' * 1500) + "final qualifier"
    original["target"]["historical_assertions"] = ["old " + "x" * 300 for _ in range(40)]
    original["source_events"][0]["derived_fact_ids"] = ["id-" + str(i) * 100 for i in range(50)]
    client = _Client(original)
    before = deepcopy(original)
    tools, ctx = _tools(client)
    recovered, pages = asyncio.run(_collect(tools, ctx))
    assert len(pages) > 1
    for candidate in recovered["candidates"]:
        candidate.pop("selection_ref")
        candidate.pop("required_source_refs")
    for source in recovered["source_events"]:
        source.pop("selection_ref")
    assert recovered == original
    assert client.preview == before
    assert pages[-1]["apply_preconditions"] == original["apply_preconditions"]
    assert all(operation == "forget_preview" for operation, _ in client.calls)


@pytest.mark.parametrize("change", ["text", "scope", "order"])
def test_page_cursor_rejects_changed_snapshot_even_with_same_storage_digest(change: str) -> None:
    client = _Client(_preview())
    tools, ctx = _tools(client)
    first = json.loads(asyncio.run(tools["mnemos_forget_preview"]("target", 2, ctx=ctx)))
    if change == "text":
        client.preview["candidates"][0]["content"] += "Changed qualification."
    elif change == "scope":
        client.preview["scope"]["project_id"] = "other-project"
    else:
        client.preview["candidates"].reverse()
    with pytest.raises(ValueError, match="stale"):
        asyncio.run(
            tools["mnemos_forget_preview"](
                "target", 2, cursor=first["review"]["next_cursor"], ctx=ctx
            )
        )


@pytest.mark.parametrize("cursor", [True, "", "bad", "a" * 64 + ":-1", "a" * 64 + ":01"])
def test_invalid_page_cursor_is_rejected(cursor: Any) -> None:
    tools, ctx = _tools(_Client(_preview()))
    with pytest.raises(ValueError):
        asyncio.run(tools["mnemos_forget_preview"]("target", 2, cursor=cursor, ctx=ctx))


def test_last_page_does_not_make_incomplete_storage_preview_applicable() -> None:
    preview = _preview()
    preview["complete"] = False
    preview["apply_preconditions"] = None
    tools, ctx = _tools(_Client(preview))
    recovered, pages = asyncio.run(_collect(tools, ctx))
    assert recovered["complete"] is False
    assert all(page["apply_preconditions"] is None for page in pages)


def test_short_selections_resolve_only_against_matching_preview() -> None:
    client = _Client(_preview())
    tools, ctx = _tools(client)
    asyncio.run(
        tools["mnemos_forget_related"](
            "target",
            2,
            "a" * 64,
            selected_fact_refs=["f1"],
            selected_source_refs=["s1"],
            ctx=ctx,
        )
    )
    operation, request = client.calls[-1]
    assert operation == "forget_related"
    assert request["selected_fact_ids"] == ["fact-0"]
    assert request["selected_source_event_ids"] == ["0" * 64]
    assert request["preview_digest"] == "a" * 64
    assert "selected_fact_refs" not in request


@pytest.mark.parametrize(
    "selection",
    [
        {"selected_fact_refs": ["f99"]},
        {"selected_fact_refs": ["f0"]},
        {"selected_fact_refs": ["f01"]},
        {"selected_fact_refs": ["s1"]},
        {"selected_fact_refs": [True]},
        {"selected_fact_refs": ["f1", "f1"]},
        {"selected_source_refs": ["s3"]},
        {"selected_source_refs": ["target"]},
        {"selected_fact_refs": ["f1"], "selected_fact_ids": ["fact-1"]},
    ],
)
def test_invalid_or_mixed_short_selections_never_reach_mutation(selection: dict[str, Any]) -> None:
    client = _Client(_preview())
    tools, ctx = _tools(client)
    with pytest.raises(ValueError):
        asyncio.run(tools["mnemos_forget_related"]("target", 2, "a" * 64, ctx=ctx, **selection))
    assert all(operation == "forget_preview" for operation, _ in client.calls)


def test_stale_short_selection_never_reaches_mutation() -> None:
    client = _Client(_preview())
    client.preview["preview_digest"] = "b" * 64
    tools, ctx = _tools(client)
    with pytest.raises(ValueError, match="stale"):
        asyncio.run(
            tools["mnemos_forget_related"](
                "target", 2, "a" * 64, selected_fact_refs=["f1"], ctx=ctx
            )
        )
    assert [operation for operation, _ in client.calls] == ["forget_preview"]


def test_apply_preconditions_are_not_exposed_in_early_fragments() -> None:
    # Service transports can sort dictionary keys, placing apply_preconditions first.
    preview = dict(sorted(_preview().items()))
    tools, ctx = _tools(_Client(preview))
    _, pages = asyncio.run(_collect(tools, ctx))
    assert all(
        fragment["path"][0] != "apply_preconditions"
        for page in pages[:-1]
        for fragment in page["fragments"]
    )


def test_incomplete_short_selection_never_reaches_mutation() -> None:
    preview = _preview()
    preview["complete"] = False
    client = _Client(preview)
    tools, ctx = _tools(client)
    with pytest.raises(ValueError, match="incomplete"):
        asyncio.run(
            tools["mnemos_forget_related"](
                "target", 2, "a" * 64, selected_fact_refs=["f1"], ctx=ctx
            )
        )
    assert all(operation == "forget_preview" for operation, _ in client.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("interleave", [False, True])
async def test_real_store_keeps_atomic_guards_after_short_reference_resolution(
    tmp_path: Path, interleave: bool
) -> None:
    from tests.test_related_forget import _seed_overlap, _service

    memory, context, _, _ = _service(tmp_path)
    try:
        _, target, captured = await _seed_overlap(memory, context)

        class Client:
            async def request(
                self, operation: str, payload: dict[str, Any], **_: Any
            ) -> dict[str, Any]:
                if operation == "forget_related" and interleave:
                    # A separate connection changes a candidate after alias resolution.
                    with sqlite3.connect(tmp_path / "memory.sqlite") as concurrent:
                        concurrent.execute(
                            "UPDATE memory_chunks SET content=? WHERE id=?",
                            (captured.content + " Keep this new qualification.", captured.id),
                        )
                    concurrent.close()
                return await memory.dispatch(operation, payload)

        tools, ctx = _tools(Client(), context)
        preview = json.loads(
            await tools["mnemos_forget_preview"](target.id, target.version, ctx=ctx)
        )
        candidate = next(x for x in preview["candidates"] if x["fact_id"] == captured.id)
        operation = tools["mnemos_forget_related"](
            **preview["apply_preconditions"],
            selected_fact_refs=[candidate["selection_ref"]],
            selected_source_refs=candidate["required_source_refs"],
            ctx=ctx,
        )
        if interleave:
            with pytest.raises(ValueError, match="stale"):
                await operation
            assert memory.engine.store.get(target.id) is not None
            assert memory.engine.store.get(captured.id) is not None
            assert (
                memory.engine.store._conn.execute(
                    "SELECT COUNT(*) FROM memory_suppressions"
                ).fetchone()[0]
                == 0
            )
        else:
            result = json.loads(await operation)
            assert result["status"] == "committed"
            assert set(result["selected_fact_ids"]) == {target.id, captured.id}
            assert memory.engine.store.get(target.id) is None
            assert memory.engine.store.get(captured.id) is None
        assert memory.engine.store._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not memory.engine.store._conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_short_fact_reference_does_not_automatically_select_required_source(
    tmp_path: Path,
) -> None:
    from tests.test_related_forget import _seed_overlap, _service

    memory, context, _, _ = _service(tmp_path)
    try:
        _, target, captured = await _seed_overlap(memory, context)

        class Client:
            async def request(
                self, operation: str, payload: dict[str, Any], **_: Any
            ) -> dict[str, Any]:
                return await memory.dispatch(operation, payload)

        tools, ctx = _tools(Client(), context)
        preview = json.loads(
            await tools["mnemos_forget_preview"](target.id, target.version, ctx=ctx)
        )
        candidate = next(x for x in preview["candidates"] if x["fact_id"] == captured.id)
        with pytest.raises(ValueError, match="explicit source selection"):
            await tools["mnemos_forget_related"](
                **preview["apply_preconditions"],
                selected_fact_refs=[candidate["selection_ref"]],
                ctx=ctx,
            )
        assert memory.engine.store.get(target.id) is not None
        assert memory.engine.store.get(captured.id) is not None
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_real_fastmcp_accepts_short_references_and_rejects_boolean_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.server.fastmcp import Context, FastMCP
    from mcp.server.fastmcp.exceptions import ToolError
    import mnemos.mcp_server as module

    client = _Client(_preview())
    _, ctx = _tools(client)
    monkeypatch.setattr(module, "Context", Context, raising=False)
    monkeypatch.setattr(
        module, "_service_context_from_ctx", lambda *_: ctx.request_context.lifespan_context
    )
    server = FastMCP("forget-review-transport")
    _register_service_tools(server)
    arguments = {
        "fact_id": "target",
        "expected_version": 2,
        "preview_digest": "a" * 64,
        "selected_fact_refs": [True],
        "selected_source_refs": ["s1"],
    }
    with pytest.raises(ToolError):
        await server._tool_manager.call_tool("mnemos_forget_related", arguments)
    assert client.calls == []
    arguments["selected_fact_refs"] = ["f1"]
    await server._tool_manager.call_tool("mnemos_forget_related", arguments)
    assert client.calls[-1][1]["selected_fact_ids"] == ["fact-0"]
    assert client.calls[-1][1]["selected_source_event_ids"] == ["0" * 64]
