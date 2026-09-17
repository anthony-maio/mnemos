"""Authenticated continuity contract shared by native host adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemos.adapters.claude_code import ClaudeCodeAdapter
from mnemos.adapters.codex import CodexAdapter
from mnemos.adapters.hermes import MnemosHermesProvider
from mnemos.config import MnemosConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore
from mnemos.client import MnemosClient


def _host_contexts(project: Path, registry: ScopeRegistry) -> dict[str, dict]:
    profile = "cross-host-profile"
    codex = CodexAdapter(registry=registry, user_profile_id=profile)
    claude = ClaudeCodeAdapter(registry=registry, user_profile_id=profile)
    hermes = MnemosHermesProvider(registry=registry, user_profile_id=profile)
    common = {"cwd": str(project), "session_id": "continuity-session"}
    contexts = {
        "codex": codex._context({**common, "turn_id": "turn-1"}),
        "claude-code": claude._context(common),
    }
    hermes.initialize("continuity-session", cwd=str(project), agent_identity=profile)
    contexts["hermes"] = hermes._current_context()
    return {host: context.model_dump(mode="json") for host, context in contexts.items()}


def _start_service(tmp_path: Path, registry: ScopeRegistry) -> MnemosService:
    config = MnemosConfig()
    config.modules.surprisal = False
    config.modules.sleep = False
    config.modules.affective = False
    engine = MnemosEngine(
        config=config,
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(dim=64),
        store=SQLiteStore(str(tmp_path / "continuity.sqlite")),
    )
    domain = LocalMemoryService(engine, registry)
    service = MnemosService(
        dispatcher=domain,
        descriptor_path=tmp_path / "runtime" / "descriptor.json",
        lock_path=tmp_path / "runtime" / "service.lock",
        startup_timeout=3,
        dispatch_timeout=10,
    )
    service.start()
    return service


@pytest.mark.asyncio
async def test_codex_claude_hermes_share_durable_scoped_lifecycle(tmp_path: Path) -> None:
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()
    registry = ScopeRegistry()
    profile = "cross-host-profile"
    project_a_identity = registry.register_project(project_a, user_profile_id=profile, enabled=True)
    project_b_identity = registry.register_project(project_b, user_profile_id=profile, enabled=True)
    contexts = _host_contexts(project_a, registry)
    service = _start_service(tmp_path, registry)
    descriptor = service.descriptor_path
    client = MnemosClient(descriptor_path=descriptor, auto_start=False, request_timeout=5)
    try:
        assert service.descriptor and service.descriptor["host"] == "127.0.0.1"
        for host, context in contexts.items():
            result = await client.request(
                "remember",
                {
                    "context": context,
                    "event_id": f"{host}-event",
                    "content": f"Durable fact from {host}.",
                    "scope": "project",
                },
                queue_on_failure=False,
            )
            assert result["status"] == "committed"
            processed = await client.request(
                "maintenance", {"context": context}, queue_on_failure=False
            )
            assert processed.get("completed", 0) >= 0

        listed = await client.request(
            "list",
            {"context": contexts["codex"], "allowed_scopes": ["project"], "limit": 20},
            queue_on_failure=False,
        )
        facts = listed["facts"]
        assert len(facts) == 3
        target = next(
            fact for fact in facts if fact["content"].startswith("Durable fact from codex")
        )
        corrected = await client.request(
            "correct",
            {
                "context": contexts["claude-code"],
                "fact_id": target["id"],
                "expected_version": target["version"],
                "content": "Durable fact corrected through Claude Code.",
            },
            queue_on_failure=False,
        )
        assert corrected["status"] == "committed"
        await client.request(
            "maintenance", {"context": contexts["claude-code"]}, queue_on_failure=False
        )
        inspected = await client.request(
            "inspect",
            {"context": contexts["hermes"], "fact_id": target["id"]},
            queue_on_failure=False,
        )
        assert inspected["fact"]["id"] == target["id"]
        assert inspected["fact"]["version"] == target["version"] + 1
        assert inspected["fact"]["content"] == "Durable fact corrected through Claude Code."

        # A restart must retain the corrected identity and version.
        service.stop()
        service = _start_service(tmp_path, registry)
        client = MnemosClient(
            descriptor_path=service.descriptor_path, auto_start=False, request_timeout=5
        )
        after_restart = await client.request(
            "inspect",
            {"context": contexts["codex"], "fact_id": target["id"]},
            queue_on_failure=False,
        )
        assert after_restart["fact"]["id"] == target["id"]
        assert after_restart["fact"]["version"] == target["version"] + 1

        forgotten = await client.request(
            "forget",
            {
                "context": contexts["hermes"],
                "fact_id": target["id"],
                "expected_version": target["version"] + 1,
            },
            queue_on_failure=False,
        )
        assert forgotten["forgotten"] is True
        replay = await client.request(
            "remember",
            {
                "context": contexts["codex"],
                "event_id": "codex-event",
                "content": "Durable fact from codex.",
                "scope": "project",
            },
            queue_on_failure=False,
        )
        assert replay["processing_status"] == "suppressed"

        foreign_context = registry.resolve(
            project_b, user_profile_id=profile, session_id="foreign-session", host="codex"
        ).model_dump(mode="json")
        await client.request(
            "remember",
            {
                "context": foreign_context,
                "event_id": "foreign-event",
                "content": "Foreign project canary.",
                "scope": "project",
            },
            queue_on_failure=False,
        )
        foreign_view = await client.request(
            "retrieve",
            {
                "context": contexts["codex"],
                "query": "Foreign project canary",
                "allowed_scopes": ["project"],
                "top_k": 10,
                "max_context_tokens": 500,
                "profile": "baseline",
            },
            queue_on_failure=False,
        )
        assert all(
            fact["content"] != "Foreign project canary" for fact in foreign_view.get("facts", [])
        )
        assert project_a_identity.id != project_b_identity.id
    finally:
        service.stop()
