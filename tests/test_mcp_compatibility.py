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

ROOT = Path(__file__).resolve().parents[1]
TOOL_NAMES = {
    "mnemos_store",
    "mnemos_retrieve",
    "mnemos_consolidate",
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
    args = [_expand_template(str(arg), env) for arg in spec.get("args", [])]
    return StdioServerParameters(
        command=command,
        args=args,
        env=env,
        cwd=str(cwd),
    )


async def _assert_tier1_roundtrip(params: StdioServerParameters) -> None:
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            assert TOOL_NAMES.issubset({tool.name for tool in tools.tools})

            resources = await session.list_resources()
            assert RESOURCE_URIS.issubset({str(resource.uri) for resource in resources.resources})

            store_result = await session.call_tool(
                "mnemos_store",
                {
                    "content": "Current project uses uv and Ruff for Python tooling.",
                    "scope": "project",
                    "scope_id": "repo-alpha",
                },
            )
            store_payload = json.loads(_call_result_text(store_result))
            assert store_payload["stored"] is True
            assert store_payload["scope"] == "project"
            assert store_payload["scope_id"] == "repo-alpha"
            assert store_payload["chunk_id"]

            inspect_result = await session.call_tool(
                "mnemos_inspect",
                {
                    "chunk_id": store_payload["chunk_id"],
                },
            )
            inspect_payload = json.loads(_call_result_text(inspect_result))
            assert inspect_payload["scope"] == "project"
            assert inspect_payload["scope_id"] == "repo-alpha"
            assert inspect_payload["provenance"]["stored_by"] == "surprisal_gate"
            assert inspect_payload["provenance"]["ingest_channel"] == "manual"
            assert "graph" in inspect_payload

            retrieve_result = await session.call_tool(
                "mnemos_retrieve",
                {
                    "query": "python tooling",
                    "top_k": 3,
                    "reconsolidate": False,
                    "current_scope": "project",
                    "scope_id": "repo-alpha",
                    "allowed_scopes": "project,global",
                },
            )
            retrieve_payload = json.loads(_call_result_text(retrieve_result))
            assert retrieve_payload
            assert retrieve_payload[0]["scope"] == "project"

            health_result = await session.call_tool("mnemos_health", {})
            health_payload = json.loads(_call_result_text(health_result))
            assert "scope_isolation" in health_payload
            assert "runtime" in health_payload
            assert health_payload["runtime"]["total_chunks"] >= 1

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
    params = _server_params_from_spec(
        spec,
        cwd=ROOT,
        env_overrides={
            "MNEMOS_LLM_PROVIDER": "mock",
            "MNEMOS_EMBEDDING_PROVIDER": "simple",
            "MNEMOS_STORE_TYPE": "sqlite",
            "MNEMOS_SQLITE_PATH": str(tmp_path / f"{label}.db"),
            "MNEMOS_SURPRISAL_THRESHOLD": "0.0",
        },
    )
    await _assert_tier1_roundtrip(params)


@pytest.mark.asyncio
async def test_tier1_claude_code_plugin_manifest_executes_real_roundtrip(tmp_path: Path) -> None:
    spec = _load_server_spec(ROOT / ".claude-plugin" / "plugin.json")
    params = _server_params_from_spec(
        spec,
        cwd=ROOT,
        env_overrides={
            "CLAUDE_PLUGIN_ROOT": str(ROOT),
            "MNEMOS_PLUGIN_PYTHON": "python",
            "MNEMOS_LLM_PROVIDER": "mock",
            "MNEMOS_EMBEDDING_PROVIDER": "simple",
            "MNEMOS_STORE_TYPE": "sqlite",
            "MNEMOS_SQLITE_PATH": str(tmp_path / "claude-code-plugin.db"),
            "MNEMOS_SURPRISAL_THRESHOLD": "0.0",
        },
    )
    await _assert_tier1_roundtrip(params)


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
