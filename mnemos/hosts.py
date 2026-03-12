"""
mnemos/hosts.py — Host integration preview/apply helpers for onboarding.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import unified_diff
from pathlib import Path
from typing import Any, Literal

from .antigravity import build_antigravity_artifact
from .settings import _emit_toml_sections

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

HostName = Literal["claude-code", "cursor", "codex"]


@dataclass(slots=True)
class HostIntegrationPreview:
    host: HostName
    config_path: Path
    preview_text: str
    rendered_config: str


@dataclass(slots=True)
class HostIntegrationResult:
    host: HostName
    config_path: Path
    backup_path: Path | None
    preview_text: str


def _home_path(home: str | Path | None) -> Path:
    return Path.home() if home is None else Path(home)


def _cwd_path(cwd: str | Path | None) -> Path:
    return Path.cwd() if cwd is None else Path(cwd)


def _find_plugin_wrapper(cwd: Path) -> Path | None:
    resolved = cwd.resolve()
    for candidate_dir in (resolved, *resolved.parents):
        wrapper = candidate_dir / ".claude-plugin" / "run_mnemos_mcp.py"
        if wrapper.exists():
            return wrapper
    return None


def _config_path_for_host(host: HostName, *, cwd: Path, home: Path) -> Path:
    if host == "cursor":
        return cwd / ".cursor" / "mcp.json"
    if host == "codex":
        return home / ".codex" / "config.toml"
    return home / ".claude" / "claude_desktop_config.json"


def _cursor_rule_path(cwd: Path) -> Path:
    return cwd / ".cursor" / "rules" / "mnemos-memory.mdc"


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _preview_diff(path: Path, old_text: str, new_text: str) -> str:
    diff = unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        fromfile=str(path),
        tofile=str(path),
        lineterm="",
    )
    return "\n".join(diff)


def _merge_json_host_config(
    existing_text: str,
    *,
    command: str,
    args: list[str] | None,
    mnemos_config_path: Path,
) -> str:
    if existing_text.strip():
        payload = json.loads(existing_text)
    else:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
        payload["mcpServers"] = servers

    mnemos_entry: dict[str, Any] = {
        "command": command,
        "env": {
            "MNEMOS_CONFIG_PATH": mnemos_config_path.as_posix(),
        },
    }
    if args:
        mnemos_entry["args"] = args
    servers["mnemos"] = mnemos_entry
    return json.dumps(payload, indent=2) + "\n"


def _merge_codex_config(existing_text: str, *, mnemos_config_path: Path) -> str:
    if existing_text.strip():
        payload = tomllib.loads(existing_text)
    else:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    servers = payload.get("mcp_servers")
    if not isinstance(servers, dict):
        servers = {}
        payload["mcp_servers"] = servers

    servers["mnemos"] = {
        "command": "mnemos-mcp",
        "env": {
            "MNEMOS_CONFIG_PATH": mnemos_config_path.as_posix(),
        },
    }
    return "\n".join(_emit_toml_sections(payload)).strip() + "\n"


def _render_host_config(
    host: HostName,
    *,
    config_path: Path,
    existing_text: str,
    mnemos_config_path: Path,
    cwd: Path,
) -> str:
    if host == "codex":
        return _merge_codex_config(existing_text, mnemos_config_path=mnemos_config_path)

    if host == "cursor":
        return _merge_json_host_config(
            existing_text,
            command="mnemos-mcp",
            args=None,
            mnemos_config_path=mnemos_config_path,
        )

    wrapper = _find_plugin_wrapper(cwd)
    if wrapper is not None:
        return _merge_json_host_config(
            existing_text,
            command=sys.executable,
            args=[str(wrapper)],
            mnemos_config_path=mnemos_config_path,
        )
    return _merge_json_host_config(
        existing_text,
        command="mnemos-mcp",
        args=None,
        mnemos_config_path=mnemos_config_path,
    )


def preview_host_integration(
    host: HostName,
    *,
    mnemos_config_path: str | Path,
    cwd: str | Path | None = None,
    home: str | Path | None = None,
) -> HostIntegrationPreview:
    resolved_cwd = _cwd_path(cwd)
    resolved_home = _home_path(home)
    config_path = _config_path_for_host(host, cwd=resolved_cwd, home=resolved_home)
    existing_text = _read_text(config_path)
    rendered_config = _render_host_config(
        host,
        config_path=config_path,
        existing_text=existing_text,
        mnemos_config_path=Path(mnemos_config_path),
        cwd=resolved_cwd,
    )
    preview_text = _preview_diff(config_path, existing_text, rendered_config)
    if host == "cursor":
        rule_path = _cursor_rule_path(resolved_cwd)
        existing_rule_text = _read_text(rule_path)
        rendered_rule = build_antigravity_artifact("cursor", "cursor-rule")
        rule_diff = _preview_diff(rule_path, existing_rule_text, rendered_rule)
        if rule_diff:
            preview_text = "\n\n".join(part for part in (preview_text, rule_diff) if part)
    return HostIntegrationPreview(
        host=host,
        config_path=config_path,
        preview_text=preview_text,
        rendered_config=rendered_config,
    )


def _backup_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return path.with_name(f"{path.name}.bak.{stamp}")


def apply_host_integration(
    host: HostName,
    *,
    mnemos_config_path: str | Path,
    cwd: str | Path | None = None,
    home: str | Path | None = None,
) -> HostIntegrationResult:
    preview = preview_host_integration(
        host,
        mnemos_config_path=mnemos_config_path,
        cwd=cwd,
        home=home,
    )
    preview.config_path.parent.mkdir(parents=True, exist_ok=True)

    backup_path: Path | None = None
    if preview.config_path.exists():
        backup_path = _backup_path(preview.config_path)
        backup_path.write_text(_read_text(preview.config_path), encoding="utf-8")

    preview.config_path.write_text(preview.rendered_config, encoding="utf-8")
    if host == "cursor":
        rule_path = _cursor_rule_path(_cwd_path(cwd))
        rule_path.parent.mkdir(parents=True, exist_ok=True)
        if rule_path.exists():
            rule_backup_path = _backup_path(rule_path)
            rule_backup_path.write_text(_read_text(rule_path), encoding="utf-8")
        rule_path.write_text(
            build_antigravity_artifact("cursor", "cursor-rule"),
            encoding="utf-8",
        )
    return HostIntegrationResult(
        host=host,
        config_path=preview.config_path,
        backup_path=backup_path,
        preview_text=preview.preview_text,
    )
