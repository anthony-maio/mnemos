"""
tests/test_mcp_compatibility.py — Tier 1 MCP end-to-end compatibility checks.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from mnemos.config import MnemosConfig, SurprisalConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import SQLiteStore

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "project-compat"
PROFILE_ID = "profile-compat"
TOOL_NAMES = {
    "mnemos_store",
    "mnemos_capture",
    "mnemos_retrieve",
    "mnemos_feedback",
    "mnemos_consolidate",
    "mnemos_correct",
    "mnemos_job_status",
    "mnemos_forget",
    "mnemos_stats",
    "mnemos_health",
    "mnemos_inspect",
    "mnemos_list",
}
RESOURCE_URIS = {
    "mnemos://stats",
    "mnemos://architecture",
}
ENV_VAR_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _load_server_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["mcpServers"]["mnemos"]


def _expand_template(value: str, env: dict[str, str]) -> str:
    return ENV_VAR_RE.sub(lambda match: env.get(match.group(1), match.group(0)), value)


def _call_result_text(result: Any) -> str:
    texts: list[str] = []
    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)


def _server_params_from_spec(
    spec: dict[str, Any],
    *,
    cwd: Path,
    env_overrides: dict[str, str],
) -> StdioServerParameters:
    env = dict(os.environ)
    for key, value in spec.get("env", {}).items():
        env[key] = _expand_template(str(value), env)
    env.update(env_overrides)
    command = _expand_template(str(spec["command"]), env)
    if command == "python" and env.get("MNEMOS_TEST_PYTHON"):
        command = env["MNEMOS_TEST_PYTHON"]
    args = [_expand_template(str(arg), env) for arg in spec.get("args", [])]
    return StdioServerParameters(
        command=command,
        args=args,
        env=env,
        cwd=str(cwd),
    )


def _start_scoped_service(tmp_path: Path, db_name: str) -> tuple[MnemosService, Path]:
    """Start one parent-owned service for a child stdio MCP process."""
    registry = ScopeRegistry(profiles=[PROFILE_ID])
    project = registry.register_project(ROOT, user_profile_id=PROFILE_ID, enabled=True)
    # Keep the identifier deterministic for assertions while retaining the
    # registry's canonicalized worktree root.
    project.id = PROJECT_ID
    registry.projects = {PROJECT_ID: project}
    registry_path = tmp_path / f"{db_name}.scopes.json"
    registry.save(registry_path)

    config = MnemosConfig(surprisal=SurprisalConfig(threshold=0.0, min_content_length=0))
    # Exercise transport and durable lifecycle deterministically.  This test
    # deliberately leaves cognition out of the service fixture.
    config.modules.surprisal = False
    config.modules.sleep = False
    engine = MnemosEngine(
        config=config,
        llm=MockLLMProvider(),
        embedder=SimpleEmbeddingProvider(dim=config.surprisal.embedding_dim),
        store=SQLiteStore(str(tmp_path / f"{db_name}.db")),
    )
    domain = LocalMemoryService(engine, registry)
    runtime_dir = tmp_path / f"{db_name}.runtime"
    service = MnemosService(
        dispatcher=domain,
        descriptor_path=runtime_dir / "runtime.json",
        lock_path=runtime_dir / "runtime.lock",
        startup_timeout=3,
        dispatch_timeout=10,
    )
    service.start()
    return service, registry_path


def _service_env(service: MnemosService, registry_path: Path) -> dict[str, str]:
    descriptor = service.descriptor
    assert descriptor is not None
    return {
        "MNEMOS_RUNTIME_DESCRIPTOR": str(service.descriptor_path),
        "MNEMOS_SCOPE_REGISTRY": str(registry_path),
        "MNEMOS_USER_PROFILE_ID": PROFILE_ID,
        "MNEMOS_PROJECT_ROOT": str(ROOT),
        "MNEMOS_SESSION_ID": "mcp-compat-session",
        "MNEMOS_HOST": "manual",
        "MNEMOS_AUTO_START": "0",
    }


async def _assert_tier1_roundtrip(params: StdioServerParameters) -> None:
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            assert TOOL_NAMES.issubset({tool.name for tool in tools.tools})

            resources = await session.list_resources()
            assert RESOURCE_URIS.issubset({str(resource.uri) for resource in resources.resources})

            store_result = await session.call_tool(
                "mnemos_capture",
                {
                    "content": "Current project uses uv and Ruff for Python tooling.",
                    "role": "user",
                    "source_kind": "compatibility-test",
                    "scope": "project",
                    "scope_id": PROJECT_ID,
                },
            )
            store_payload = json.loads(_call_result_text(store_result))
            assert store_payload["event_committed"] is True
            assert store_payload["scope"] == "project"
            assert store_payload["scope_id"] == PROJECT_ID
            assert store_payload["event_id"]
            assert store_payload["processing_status"] in {"pending", "processing", "consolidated"}

            job_result = await session.call_tool(
                "mnemos_job_status", {"job_id": store_payload["job_id"]}
            )
            job_payload = json.loads(_call_result_text(job_result))
            assert job_payload["schema_version"] == 1
            assert job_payload["job"]["id"] == store_payload["job_id"]

            maintenance_result = await session.call_tool("mnemos_consolidate", {})
            maintenance_payload = json.loads(_call_result_text(maintenance_result))
            assert maintenance_payload["schema_version"] == 1
            assert "completed" in maintenance_payload

            retrieve_result = await session.call_tool(
                "mnemos_retrieve",
                {
                    "query": "python tooling",
                    "top_k": 3,
                    "reconsolidate": False,
                    "current_scope": "project",
                    "scope_id": PROJECT_ID,
                    "allowed_scopes": "project,global",
                },
            )
            retrieve_payload = json.loads(_call_result_text(retrieve_result))
            assert retrieve_payload["schema_version"] == 1
            assert retrieve_payload["facts"]
            fact = retrieve_payload["facts"][0]
            fact_id = fact["id"]
            assert fact["scope"] == "project"
            assert fact["scope_id"] == PROJECT_ID
            assert fact["scores"]

            inspect_result = await session.call_tool(
                "mnemos_inspect",
                {
                    "chunk_id": fact_id,
                },
            )
            inspect_payload = json.loads(_call_result_text(inspect_result))
            assert inspect_payload["schema_version"] == 1
            assert inspect_payload["fact"]["id"] == fact_id
            assert inspect_payload["fact"]["metadata"]["scope"] == "project"
            assert inspect_payload["fact"]["metadata"]["scope_id"] == PROJECT_ID
            assert inspect_payload["evidence"]

            feedback_result = await session.call_tool(
                "mnemos_feedback",
                {
                    "event_type": "helpful",
                    "query": "python tooling",
                    "scope": "project",
                    "scope_id": PROJECT_ID,
                    "chunk_ids": [fact_id],
                    "notes": "This recall was correct.",
                },
            )
            feedback_payload = json.loads(_call_result_text(feedback_result))
            assert feedback_payload["stored"] is True
            assert feedback_payload["event_type"] == "helpful"
            assert feedback_payload["chunk_ids"] == [fact_id]

            correct_result = await session.call_tool(
                "mnemos_correct",
                {
                    "fact_id": fact_id,
                    "content": "Current project uses uv and Ruff for Python tooling, with pytest for tests.",
                    "expected_version": inspect_payload["fact"]["version"],
                    "source_ref": "compatibility-test",
                },
            )
            correct_payload = json.loads(_call_result_text(correct_result))
            assert correct_payload["status"] == "committed"
            corrected_maintenance = await session.call_tool("mnemos_consolidate", {})
            assert json.loads(_call_result_text(corrected_maintenance))["schema_version"] == 1

            corrected_inspect_result = await session.call_tool(
                "mnemos_inspect", {"chunk_id": fact_id}
            )
            corrected_inspect = json.loads(_call_result_text(corrected_inspect_result))
            assert "pytest for tests" in corrected_inspect["fact"]["content"]
            assert corrected_inspect["fact"]["version"] > inspect_payload["fact"]["version"]

            forget_result = await session.call_tool(
                "mnemos_forget",
                {
                    "chunk_id": fact_id,
                    "expected_version": corrected_inspect["fact"]["version"],
                },
            )
            forget_payload = json.loads(_call_result_text(forget_result))
            assert forget_payload["status"] == "committed"
            assert forget_payload["forgotten"] is True

            health_result = await session.call_tool("mnemos_health", {})
            health_payload = json.loads(_call_result_text(health_result))
            assert health_payload["schema_version"] == 1
            assert health_payload["status"] == "running"
            assert health_payload["store"]["total_chunks"] >= 0

            stats_resource = await session.read_resource("mnemos://stats")
            stats_payload = json.loads(stats_resource.contents[0].text)
            assert stats_payload["status"] == "running"

            architecture = await session.read_resource("mnemos://architecture")
            assert architecture.contents
            architecture_payload = json.loads(architecture.contents[0].text)
            assert "pipeline" in architecture_payload
            assert "retrieve" in architecture_payload["pipeline"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "config_path"),
    [
        ("generic-stdio", ROOT / "docs" / "mcp-configs" / "generic-stdio.json"),
        ("claude-desktop", ROOT / "docs" / "mcp-configs" / "claude-desktop.json"),
    ],
)
async def test_tier1_stdio_configs_execute_real_roundtrip(
    label: str, config_path: Path, tmp_path: Path
) -> None:
    spec = _load_server_spec(config_path)
    service, registry_path = _start_scoped_service(tmp_path, label)
    try:
        params = _server_params_from_spec(
            spec,
            cwd=ROOT,
            env_overrides={
                "MNEMOS_TEST_PYTHON": sys.executable,
                **_service_env(service, registry_path),
            },
        )
        await _assert_tier1_roundtrip(params)
    finally:
        service.stop()


@pytest.mark.asyncio
async def test_tier1_claude_code_plugin_manifest_executes_real_roundtrip(tmp_path: Path) -> None:
    spec = _load_server_spec(ROOT / ".claude-plugin" / "plugin.json")
    service, registry_path = _start_scoped_service(tmp_path, "claude-code-plugin")
    try:
        params = _server_params_from_spec(
            spec,
            cwd=ROOT,
            env_overrides={
                "CLAUDE_PLUGIN_ROOT": str(ROOT),
                "MNEMOS_TEST_PYTHON": sys.executable,
                "MNEMOS_PLUGIN_PYTHON": sys.executable,
                **_service_env(service, registry_path),
            },
        )
        await _assert_tier1_roundtrip(params)
    finally:
        service.stop()


def test_tier2_codex_config_and_guide_validate() -> None:
    codex_config = json.loads(
        (ROOT / "docs" / "mcp-configs" / "codex.json").read_text(encoding="utf-8")
    )
    codex_guide = (ROOT / "docs" / "codex.md").read_text(encoding="utf-8")
    agents_guide = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    codex_server = codex_config["mcpServers"]["mnemos"]

    assert codex_server["command"] == "python"
    assert codex_server["args"] == ["-m", "mnemos.mcp_server"]
    assert "AGENTS.md" in codex_guide
    assert "mnemos-cli antigravity codex" in codex_guide
    assert "mnemos_retrieve" in agents_guide
    assert "mnemos_store" in agents_guide
    assert "mnemos_consolidate" in agents_guide
