"""
tests/test_mcp_compatibility.py — Tier 1 MCP compatibility smoke checks.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

from mnemos import mcp_server


def _install_fake_fastmcp(monkeypatch: Any) -> None:
    fake_mcp = types.ModuleType("mcp")
    fake_server = types.ModuleType("mcp.server")
    fake_fastmcp = types.ModuleType("mcp.server.fastmcp")

    class FakeFastMCP:
        def __init__(self, name: str, dependencies: list[str], lifespan: Any) -> None:
            self.name = name
            self.dependencies = dependencies
            self.lifespan = lifespan
            self.tools: list[str] = []
            self.resources: list[str] = []
            self.transport: str | None = None

        def tool(self) -> Any:
            def _decorator(fn: Any) -> Any:
                self.tools.append(fn.__name__)
                return fn

            return _decorator

        def resource(self, uri: str) -> Any:
            def _decorator(fn: Any) -> Any:
                _ = fn
                self.resources.append(uri)
                return fn

            return _decorator

        def run(self, transport: str) -> None:
            self.transport = transport

    fake_fastmcp.FastMCP = FakeFastMCP  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)


def test_tier1_generic_stdio_contract_registers_tools_and_resources(monkeypatch: Any) -> None:
    _install_fake_fastmcp(monkeypatch)

    server = mcp_server.create_mcp_server()

    assert set(server.tools) >= {
        "mnemos_store",
        "mnemos_retrieve",
        "mnemos_consolidate",
        "mnemos_forget",
        "mnemos_stats",
        "mnemos_health",
        "mnemos_inspect",
        "mnemos_list",
    }
    assert set(server.resources) >= {
        "mnemos://stats",
        "mnemos://architecture",
    }


def test_tier1_mcp_main_uses_stdio_transport(monkeypatch: Any) -> None:
    class DummyServer:
        def __init__(self) -> None:
            self.transport: str | None = None

        def run(self, transport: str) -> None:
            self.transport = transport

    dummy = DummyServer()
    monkeypatch.setattr(mcp_server, "create_mcp_server", lambda: dummy)

    mcp_server.main()

    assert dummy.transport == "stdio"


def test_tier1_claude_code_and_desktop_configs_validate() -> None:
    plugin_manifest = json.loads(Path(".claude-plugin/plugin.json").read_text(encoding="utf-8"))
    desktop_config = json.loads(
        Path("docs/mcp-configs/claude-desktop.json").read_text(encoding="utf-8")
    )
    stdio_config = json.loads(
        Path("docs/mcp-configs/generic-stdio.json").read_text(encoding="utf-8")
    )

    plugin_server = plugin_manifest["mcpServers"]["mnemos"]
    desktop_server = desktop_config["mcpServers"]["mnemos"]
    stdio_server = stdio_config["mcpServers"]["mnemos"]

    assert plugin_server["command"] == "python"
    assert desktop_server["command"] == "mnemos-mcp"
    assert stdio_server["command"] == "mnemos-mcp"
