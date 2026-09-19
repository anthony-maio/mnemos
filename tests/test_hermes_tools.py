import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from mnemos.adapters.hermes_tools import dispatch_tool, get_tool_schemas
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.service import MnemosService
from mnemos.scopes import ScopeContext
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


def context() -> ScopeContext:
    return ScopeContext(
        project_id="project-a",
        workspace_id="workspace-a",
        user_profile_id="profile-a",
        session_id="hermes-session",
        host="hermes",
    )


class RecordingClient:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"status": "committed", "fact_id": "fact-1"}

    async def request(self, operation, payload, *, queue_on_failure=True):
        self.calls.append((operation, payload, queue_on_failure))
        return self.result


def test_schemas_are_narrow_and_scope_is_provider_owned():
    schemas = get_tool_schemas()
    assert {item["name"] for item in schemas} == {
        "mnemos_retrieve",
        "mnemos_inspect",
        "mnemos_remember",
        "mnemos_correct",
        "mnemos_forget",
        "mnemos_job_status",
    }
    for schema in schemas:
        assert schema["parameters"]["additionalProperties"] is False
        assert "context" not in schema["parameters"]["properties"]
        assert "project_id" not in schema["parameters"]["properties"]
        assert "user_profile_id" not in schema["parameters"]["properties"]


def test_retrieve_injects_initialized_context_and_rejects_scope_override():
    client = RecordingClient({"status": "ok", "facts": []})

    result = json.loads(
        asyncio.run(dispatch_tool(client, context(), "mnemos_retrieve", {"query": "release"}))
    )
    assert result == {"status": "ok", "facts": []}
    operation, payload, queue_on_failure = client.calls[-1]
    assert operation == "retrieve"
    assert payload["context"] == context().model_dump(mode="json")
    assert payload["allowed_scopes"] == ["project"]
    assert queue_on_failure is False

    rejected = json.loads(
        asyncio.run(
            dispatch_tool(
                client,
                context(),
                "mnemos_retrieve",
                {"query": "release", "project_id": "foreign"},
            )
        )
    )
    assert rejected["status"] == "rejected"
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("name", "args"),
    [
        (
            "mnemos_correct",
            {"fact_id": "fact-1", "content": "new", "expected_version": 1, "request_id": "r1"},
        ),
        ("mnemos_forget", {"fact_id": "fact-1", "expected_version": 1}),
        ("mnemos_remember", {"content": "new fact", "request_id": "r1"}),
    ],
)
def test_explicit_mutations_use_nonqueued_requests(name, args):
    client = RecordingClient()
    result = json.loads(asyncio.run(dispatch_tool(client, context(), name, args)))
    assert result == client.result
    assert client.calls[-1][2] is False
    assert client.calls[-1][1]["context"] == context().model_dump(mode="json")


def test_mutations_require_exact_identity_and_version():
    client = RecordingClient()
    missing_version = json.loads(
        asyncio.run(
            dispatch_tool(
                client,
                context(),
                "mnemos_correct",
                {"fact_id": "fact-1", "content": "new", "request_id": "r1"},
            )
        )
    )
    assert missing_version["status"] == "rejected"
    extra = json.loads(
        asyncio.run(
            dispatch_tool(
                client,
                context(),
                "mnemos_forget",
                {"fact_id": "fact-1", "expected_version": 1, "query": "fuzzy"},
            )
        )
    )
    assert extra["status"] == "rejected"
    assert client.calls == []


@pytest.mark.parametrize(
    "args",
    [
        {"content": "x", "source_ref": 7, "request_id": "r1"},
        {"content": "x", "source_ref": "", "request_id": "r1"},
        {"content": "x", "source_ref": "r" * 4097, "request_id": "r1"},
        {"content": "x" * 10001, "request_id": "r1"},
    ],
)
def test_runtime_validation_matches_tool_schema(args):
    client = RecordingClient()
    result = json.loads(asyncio.run(dispatch_tool(client, context(), "mnemos_remember", args)))
    assert result["status"] == "rejected"
    assert client.calls == []


def test_real_service_lifecycle_is_scoped_and_version_checked(tmp_path: Path):
    from mnemos.client import MnemosClient
    from mnemos.scopes import ScopeRegistry

    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="profile-a", enabled=True)
    foreign_root = tmp_path / "foreign"
    foreign_root.mkdir()
    registry.register_project(foreign_root, user_profile_id="profile-a", enabled=True)
    resolved = registry.resolve(
        tmp_path, user_profile_id="profile-a", session_id="hermes-session", host="hermes"
    )
    foreign = registry.resolve(
        foreign_root, user_profile_id="profile-a", session_id="foreign-session", host="hermes"
    )
    service = MnemosService(
        dispatcher_factory=lambda: LocalMemoryService(
            MnemosEngine(
                config=MnemosConfig(
                    modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
                ),
                llm=UnavailableLLMProvider(),
                embedder=SimpleEmbeddingProvider(),
                store=SQLiteStore(str(tmp_path / "memory.sqlite")),
            ),
            registry,
        ),
        descriptor_path=tmp_path / "runtime.json",
    )
    service.start()
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)

    def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return json.loads(asyncio.run(dispatch_tool(client, resolved, name, args)))

    try:
        remembered = call(
            "mnemos_remember", {"content": "Release label is cedar.", "request_id": "remember-1"}
        )
        assert remembered["status"] == "committed"
        asyncio.run(client.request("maintenance", queue_on_failure=False))
        facts = call("mnemos_retrieve", {"query": "release label"})["facts"]
        assert len(facts) == 1
        fact_id = facts[0]["id"]
        inspected = call("mnemos_inspect", {"fact_id": fact_id})
        assert inspected["fact"]["version"] == 1

        corrected = call(
            "mnemos_correct",
            {
                "fact_id": fact_id,
                "content": "Release label is maple.",
                "expected_version": 1,
                "request_id": "correct-1",
            },
        )
        assert corrected["status"] == "committed"
        asyncio.run(client.request("maintenance", queue_on_failure=False))
        assert "maple" in call("mnemos_retrieve", {"query": "release label"})["facts"][0]["content"]

        stale = call(
            "mnemos_correct",
            {
                "fact_id": fact_id,
                "content": "Release label is birch.",
                "expected_version": 1,
                "request_id": "correct-2",
            },
        )
        assert stale["status"] == "degraded"
        assert stale["error"]
        stale_forget = call("mnemos_forget", {"fact_id": fact_id, "expected_version": 1})
        assert stale_forget["status"] == "degraded"

        foreign_remember = asyncio.run(
            client.request(
                "remember",
                {
                    "context": foreign.model_dump(mode="json"),
                    "content": "Foreign project fact.",
                    "role": "user",
                    "scope": "project",
                    "event_id": "foreign-seed",
                },
                queue_on_failure=False,
            )
        )
        assert foreign_remember["status"] == "committed"
        asyncio.run(client.request("maintenance", queue_on_failure=False))
        foreign_facts = asyncio.run(
            client.request(
                "retrieve",
                {
                    "context": foreign.model_dump(mode="json"),
                    "query": "foreign project fact",
                    "allowed_scopes": ["project"],
                    "top_k": 5,
                    "max_context_tokens": 1200,
                    "profile": "baseline",
                },
                queue_on_failure=False,
            )
        )["facts"]
        foreign_id = foreign_facts[0]["id"]
        for name, args in (
            ("mnemos_inspect", {"fact_id": foreign_id}),
            (
                "mnemos_correct",
                {
                    "fact_id": foreign_id,
                    "content": "wrong scope",
                    "expected_version": 1,
                    "request_id": "foreign-correct",
                },
            ),
            ("mnemos_forget", {"fact_id": foreign_id, "expected_version": 1}),
            ("mnemos_job_status", {"job_id": foreign_remember["job_id"]}),
        ):
            denied = call(name, args)
            assert denied["status"] == "degraded"
        forgotten = call("mnemos_forget", {"fact_id": fact_id, "expected_version": 2})
        assert forgotten["forgotten"] is True
    finally:
        asyncio.run(client.close())
        service.stop()


def test_remember_retry_reuses_event_identity_after_lost_response(tmp_path: Path):
    from mnemos.client import MnemosClient
    from mnemos.scopes import ScopeRegistry

    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="profile-a", enabled=True)
    resolved = registry.resolve(
        tmp_path, user_profile_id="profile-a", session_id="hermes-session", host="hermes"
    )
    service = MnemosService(
        dispatcher_factory=lambda: LocalMemoryService(
            MnemosEngine(
                config=MnemosConfig(
                    modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
                ),
                llm=UnavailableLLMProvider(),
                embedder=SimpleEmbeddingProvider(),
                store=SQLiteStore(str(tmp_path / "memory.sqlite")),
            ),
            registry,
        ),
        descriptor_path=tmp_path / "runtime.json",
    )
    service.start()
    real_client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)

    class LoseFirstResponse:
        def __init__(self) -> None:
            self.first = True
            self.committed: dict[str, Any] = {}

        async def request(self, operation: str, payload: dict[str, Any], *, queue_on_failure: bool):
            result = await real_client.request(
                operation, payload, queue_on_failure=queue_on_failure
            )
            if self.first:
                self.first = False
                self.committed = result
                raise TimeoutError("response lost after commit")
            return result

    client = LoseFirstResponse()
    args = {"content": "A retry-safe remembered fact.", "request_id": "retry-1"}
    try:
        first = json.loads(asyncio.run(dispatch_tool(client, resolved, "mnemos_remember", args)))
        second = json.loads(asyncio.run(dispatch_tool(client, resolved, "mnemos_remember", args)))
        assert first["status"] == "degraded"
        assert first["committed"] is False
        assert second["status"] == "committed"
        assert second["event_id"] == client.committed["event_id"]
        assert second["job_id"] == client.committed["job_id"]
        changed = json.loads(
            asyncio.run(
                dispatch_tool(
                    client,
                    resolved,
                    "mnemos_remember",
                    {"content": "Changed payload.", "request_id": "retry-1"},
                )
            )
        )
        assert changed["status"] == "degraded"
        asyncio.run(real_client.request("maintenance", queue_on_failure=False))
        facts = json.loads(
            asyncio.run(
                dispatch_tool(
                    real_client, resolved, "mnemos_retrieve", {"query": "retry-safe fact"}
                )
            )
        )["facts"]
        assert len(facts) == 1
        import sqlite3

        with sqlite3.connect(tmp_path / "memory.sqlite") as db:
            assert db.execute("SELECT count(*) FROM episodic_events").fetchone()[0] == 1
            assert (
                db.execute("SELECT count(*) FROM memory_jobs WHERE kind='remember'").fetchone()[0]
                == 1
            )
    finally:
        asyncio.run(real_client.close())
        service.stop()
