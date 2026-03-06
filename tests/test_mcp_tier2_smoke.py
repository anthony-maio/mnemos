"""
tests/test_mcp_tier2_smoke.py — Tier 2 compatibility smoke checks (best effort).
"""

from __future__ import annotations

import json
from pathlib import Path


def test_tier2_cursor_windsurf_cline_configs_validate() -> None:
    for name in ("cursor", "windsurf", "cline"):
        config_path = Path(f"docs/mcp-configs/{name}.json")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        server = payload["mcpServers"]["mnemos"]
        assert server["command"] == "mnemos-mcp"
