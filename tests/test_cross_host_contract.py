"""Authenticated continuity contract shared by native host adapters."""

from __future__ import annotations

import json
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

SCENARIO_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "scenarios" / "local-memory-continuity.jsonl"
)


def _continuity_scenario() -> dict:
    lines = [
        line for line in SCENARIO_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 1
    scenario = json.loads(lines[0])
    assert scenario["schema_version"] == 1
    assert scenario["evidence"]["native_codex"] == "NOT RUN"
    assert scenario["evidence"]["native_claude_code"] == "NOT RUN"
    assert scenario["evidence"]["native_hermes"] == "NOT RUN"
    source_ids = [event["id"] for event in scenario["source_events"]]
    source_events = {event["id"]: event for event in scenario["source_events"]}
    step_ids = [step["id"] for step in scenario["steps"]]
    assert len(source_ids) == len(set(source_ids))
    assert len(step_ids) == len(set(step_ids))
    assert all(
        (event.get("content") or event.get("replacement_content", "")).strip()
        for event in scenario["source_events"]
    )
    assert all(
        step["operation"]
        in {"remember", "retrieve", "correct", "forget", "maintenance", "restart_service"}
        for step in scenario["steps"]
    )
    for step in scenario["steps"]:
        if "source_event" in step:
            assert step["source_event"] in source_ids
    teach = next(step for step in scenario["steps"] if step["id"] == "teach")
    assert set(teach["hosts"]) == {"codex", "claude-code", "hermes"}
    for host, event_id in teach["source_events"].items():
        assert event_id in source_events
        assert source_events[event_id]["host"] == host
        assert source_events[event_id]["operation"] == "remember"
    return scenario


def _host_contexts(project: Path, registry: ScopeRegistry, profile: str) -> dict[str, dict]:
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


@pytest.mark.parametrize("teaching_host", ["codex", "claude-code", "hermes"])
@pytest.mark.asyncio
async def test_codex_claude_hermes_share_durable_scoped_lifecycle(
    tmp_path: Path, teaching_host: str
) -> None:
    scenario = _continuity_scenario()
    source_events = {event["id"]: event for event in scenario["source_events"]}
    steps = {step["id"]: step for step in scenario["steps"]}
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()
    registry = ScopeRegistry()
    profile = scenario["profile"]
    project_a_identity = registry.register_project(project_a, user_profile_id=profile, enabled=True)
    project_b_identity = registry.register_project(project_b, user_profile_id=profile, enabled=True)
    contexts = _host_contexts(project_a, registry, profile)
    service = _start_service(tmp_path, registry)
    client = MnemosClient(
        descriptor_path=service.descriptor_path, auto_start=False, request_timeout=5
    )

    async def retrieve_for(step_id: str, host: str, context: dict | None = None) -> dict:
        step = steps[step_id]
        result = await client.request(
            "retrieve",
            {
                "context": context or contexts[host],
                "query": step["query"],
                "allowed_scopes": ["project"],
                "top_k": 10,
                "max_context_tokens": 500,
                "profile": "baseline",
            },
            queue_on_failure=False,
        )
        assert result.get("degraded") is not True and "error" not in result, (
            step_id,
            host,
            result,
        )
        assert isinstance(result.get("facts"), list) and isinstance(result.get("context"), str), (
            step_id,
            host,
            result,
        )
        contents = [fact["content"] for fact in result.get("facts", [])]
        rendered = result.get("context", "")
        expected = step.get("expected_content")
        forbidden = [
            step[key] for key in ("forbidden_content", "forbidden_content_also") if key in step
        ]
        if expected is not None:
            assert expected in contents, (step_id, host, result)
        for value in forbidden:
            assert value not in contents
            assert value not in rendered
        return result

    def restart_for(step_id: str) -> None:
        nonlocal client, service
        assert steps[step_id]["operation"] == "restart_service"
        service.stop()
        service = _start_service(tmp_path, registry)
        client = MnemosClient(
            descriptor_path=service.descriptor_path, auto_start=False, request_timeout=5
        )

    try:
        assert service.descriptor and service.descriptor["host"] == "127.0.0.1"
        teach_event = source_events[steps["teach"]["source_events"][teaching_host]]
        taught = await client.request(
            "remember",
            {
                "context": contexts[teaching_host],
                "event_id": teach_event["id"],
                "content": teach_event["content"],
                "role": teach_event["role"],
                "source_ref": teach_event["source_ref"],
                "scope": teach_event["scope"],
            },
            queue_on_failure=False,
        )
        assert taught["status"] == "committed"
        processed = await client.request(
            "maintenance", {"context": contexts["codex"]}, queue_on_failure=False
        )
        assert processed.get("completed", 0) >= 0

        listed = await client.request(
            "list",
            {"context": contexts["codex"], "allowed_scopes": ["project"], "limit": 20},
            queue_on_failure=False,
        )
        facts = listed["facts"]
        assert len(facts) == 1
        target = facts[0]
        assert target["content"] == teach_event["content"]
        initial_inspection = await client.request(
            "inspect",
            {"context": contexts["codex"], "fact_id": target["id"]},
            queue_on_failure=False,
        )
        assert any(
            source.get("source_ref") == teach_event["source_ref"]
            for source in initial_inspection.get("evidence", [])
        )
        restart_for("restart-after-teach")
        for host in steps["recall-after-teach-restart"]["hosts"]:
            await retrieve_for("recall-after-teach-restart", host)

        correction_step = steps["correct"]
        correction_event = source_events[correction_step["source_event"]]
        corrected = await client.request(
            "correct",
            {
                "context": contexts["claude-code"],
                "fact_id": target["id"],
                "expected_version": correction_step["expected_version"],
                "event_id": correction_event["id"],
                "source_ref": correction_event["source_ref"],
                "content": correction_event["replacement_content"],
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
        assert inspected["fact"]["version"] == correction_step["expected_version_after"]
        assert inspected["fact"]["content"] == correction_event["replacement_content"]
        for host in steps["recall-corrected-before-restart"]["hosts"]:
            await retrieve_for("recall-corrected-before-restart", host)

        # A restart must retain the corrected identity and version, and every
        # other host must retrieve the current content through the same HTTP service.
        restart_for("restart-after-correction")
        restart_step = steps["recall-corrected-after-restart"]
        for host in restart_step["hosts"]:
            await retrieve_for("recall-corrected-after-restart", host)
        after_restart = await client.request(
            "inspect",
            {"context": contexts["codex"], "fact_id": target["id"]},
            queue_on_failure=False,
        )
        assert after_restart["fact"]["id"] == target["id"]
        assert after_restart["fact"]["version"] == correction_step["expected_version_after"]
        assert after_restart["fact"]["content"] == correction_event["replacement_content"]

        # Prove the canary is visible in its own project before checking that
        # the project-A context cannot see it while A's corrected fact exists.
        foreign_event = source_events[steps["foreign-write"]["source_event"]]
        foreign_contexts = _host_contexts(project_b, registry, profile)
        foreign_context = foreign_contexts["codex"]
        foreign_write = await client.request(
            "remember",
            {
                "context": foreign_context,
                "event_id": foreign_event["id"],
                "content": foreign_event["content"],
                "role": foreign_event["role"],
                "source_ref": foreign_event["source_ref"],
                "scope": foreign_event["scope"],
            },
            queue_on_failure=False,
        )
        assert foreign_write["status"] == "committed"
        foreign_maintenance = await client.request(
            "maintenance", {"context": foreign_context}, queue_on_failure=False
        )
        assert foreign_maintenance.get("maintenance") is not None
        for host in contexts:
            await retrieve_for("foreign-read-project-b", host, foreign_contexts[host])
            await retrieve_for("foreign-read-project-a-canary", host)
            await retrieve_for("foreign-read-project-a-corrected", host)
            await retrieve_for("foreign-read-project-b-excludes-a", host, foreign_contexts[host])

        forget_step = steps["forget"]
        forgotten = await client.request(
            "forget",
            {
                "context": contexts["hermes"],
                "fact_id": target["id"],
                "expected_version": forget_step["expected_version"],
            },
            queue_on_failure=False,
        )
        assert forgotten["forgotten"] is True
        replay_event = source_events[steps["replay"]["source_events"][teaching_host]]
        replay = await client.request(
            "remember",
            {
                "context": contexts[teaching_host],
                "event_id": replay_event["id"],
                "content": replay_event["content"],
                "role": replay_event["role"],
                "source_ref": replay_event["source_ref"],
                "scope": replay_event["scope"],
            },
            queue_on_failure=False,
        )
        assert replay["processing_status"] == steps["replay"]["expected_processing_status"]
        replay_maintenance = await client.request(
            "maintenance", {"context": contexts[teaching_host]}, queue_on_failure=False
        )
        assert replay_maintenance.get("maintenance") is not None
        restart_for("restart-after-forget")

        for host in steps["post-forget-read"]["hosts"]:
            await retrieve_for("post-forget-read", host)
        for host in contexts:
            await retrieve_for("foreign-read-project-b-after-restart", host, foreign_contexts[host])
        assert project_a_identity.id != project_b_identity.id
    finally:
        service.stop()
