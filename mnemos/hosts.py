"""
mnemos/hosts.py — Host integration preview/apply helpers for onboarding.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import unified_diff
from pathlib import Path
from typing import Any, Literal, Mapping

from .antigravity import build_antigravity_artifact
from .settings import _emit_toml_sections

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli

HostName = Literal["claude-code", "cursor", "codex", "hermes"]
IntegrationEnv = Mapping[str, str] | None

# Host integrations receive only the Mnemos selectors needed to resolve the
# configured service.  In particular, do not copy the caller's process
# environment into a host config file.
_INTEGRATION_ENV_KEYS = frozenset(
    {
        "MNEMOS_CONFIG_PATH",
        "MNEMOS_USER_PROFILE_ID",
        "MNEMOS_SCOPE_REGISTRY",
        "MNEMOS_RUNTIME_DESCRIPTOR",
    }
)


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
    if host == "hermes":
        return home / ".hermes" / "config.yaml"
    # Claude Code reads project MCP configuration from ``.mcp.json``.  The
    # Claude Desktop JSON file is a different host and must not be rewritten
    # by this project onboarding action.
    return cwd / ".mcp.json"


def _cursor_rule_path(cwd: Path) -> Path:
    return cwd / ".cursor" / "rules" / "mnemos-memory.mdc"


def _claude_agents_dir(home: Path) -> Path:
    return home / ".claude" / "agents"


def _repo_agent_path(cwd: Path, agent_name: str) -> Path:
    candidate = cwd / "agents" / agent_name
    if candidate.exists():
        return candidate
    shipped = Path(__file__).resolve().parents[1] / "agents" / agent_name
    if shipped.is_file():
        return shipped
    return Path(__file__).resolve().parent / "resources" / "agents" / agent_name


def _codex_agents_path(cwd: Path) -> Path:
    return cwd / "AGENTS.md"


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


def _render_claude_agent(cwd: Path, agent_name: str) -> str:
    return _read_text(_repo_agent_path(cwd, agent_name))


def _merge_markdown_section(existing_text: str, *, heading: str, rendered_section: str) -> str:
    section = rendered_section.strip()
    if not existing_text.strip():
        return f"{section}\n"

    pattern = re.compile(rf"(?ms)^##\s+{re.escape(heading)}\s*\n.*?(?=^#{{1,6}}\s|\Z)")
    if pattern.search(existing_text):
        updated = pattern.sub(section, existing_text, count=1)
        return updated.rstrip() + "\n"

    return existing_text.rstrip() + "\n\n" + section + "\n"


def _integration_environment(
    mnemos_config_path: Path,
    integration_env: IntegrationEnv,
) -> dict[str, str]:
    """Build the small environment payload safe to write into host configs."""
    values: dict[str, str] = {
        "MNEMOS_CONFIG_PATH": mnemos_config_path.as_posix(),
    }
    if integration_env is None:
        return values
    unsupported = sorted(set(integration_env) - _INTEGRATION_ENV_KEYS)
    if unsupported:
        raise ValueError("integration_env contains unsupported key(s): " + ", ".join(unsupported))
    for key, value in integration_env.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"integration_env value for {key} must be a non-empty string")
        values[key] = value
    return values


def _merge_json_host_config(
    existing_text: str,
    *,
    command: str,
    args: list[str] | None,
    mnemos_config_path: Path,
    integration_env: IntegrationEnv = None,
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

    existing_entry = servers.get("mnemos")
    mnemos_entry: dict[str, Any] = (
        dict(existing_entry) if isinstance(existing_entry, Mapping) else {}
    )
    existing_env = mnemos_entry.get("env")
    env: dict[str, Any] = dict(existing_env) if isinstance(existing_env, Mapping) else {}
    rendered_env = _integration_environment(mnemos_config_path, integration_env)
    env.update(rendered_env)
    mnemos_entry["command"] = command
    mnemos_entry["env"] = env
    mnemos_entry["args"] = args or []
    servers["mnemos"] = mnemos_entry
    return json.dumps(payload, indent=2) + "\n"


def _merge_hermes_config(existing_text: str) -> str:
    """Set only Hermes' provider selector while retaining its YAML settings."""
    lines = existing_text.splitlines()
    memory_start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() == "memory:" and not line.startswith(" ")
        ),
        None,
    )
    if memory_start is None:
        base = existing_text.rstrip()
        return (base + "\n\n" if base else "") + "memory:\n  provider: mnemos\n"

    memory_end = len(lines)
    for index in range(memory_start + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "\t", "#")):
            memory_end = index
            break
    provider_line = next(
        (
            index
            for index in range(memory_start + 1, memory_end)
            if lines[index].lstrip().startswith("provider:")
        ),
        None,
    )
    if provider_line is None:
        lines.insert(memory_start + 1, "  provider: mnemos")
    else:
        indentation = lines[provider_line][
            : len(lines[provider_line]) - len(lines[provider_line].lstrip())
        ]
        lines[provider_line] = f"{indentation}provider: mnemos"
    return "\n".join(lines).rstrip() + "\n"


def _merge_codex_config(
    existing_text: str,
    *,
    mnemos_config_path: Path,
    integration_env: IntegrationEnv = None,
) -> str:
    if existing_text.strip():
        if sys.version_info >= (3, 11):
            payload = tomllib.loads(existing_text)
        else:  # pragma: no cover - exercised on Python 3.10
            payload = tomli.loads(existing_text)
    else:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    servers = payload.get("mcp_servers")
    if not isinstance(servers, dict):
        servers = {}
        payload["mcp_servers"] = servers

    existing_entry = servers.get("mnemos")
    mnemos_entry: dict[str, Any] = (
        dict(existing_entry) if isinstance(existing_entry, Mapping) else {}
    )
    existing_env = mnemos_entry.get("env")
    env: dict[str, Any] = dict(existing_env) if isinstance(existing_env, Mapping) else {}
    rendered_env = _integration_environment(mnemos_config_path, integration_env)
    env.update(rendered_env)
    # MCP and hooks must use the installation that generated this config.
    # A bare console script can resolve to an older installation on PATH.
    mnemos_entry["command"] = sys.executable
    mnemos_entry["args"] = ["-m", "mnemos.mcp_server"]
    mnemos_entry["env"] = env
    servers["mnemos"] = mnemos_entry
    base = {key: value for key, value in payload.items() if key != "hooks"}
    lines = _emit_toml_sections(base)
    hook_lines = _render_codex_hooks(
        payload.get("hooks"),
        integration_env=rendered_env if integration_env is not None else None,
    )
    if hook_lines:
        lines.extend(["", *hook_lines])
    return "\n".join(lines).strip() + "\n"


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list) and all(not isinstance(item, Mapping) for item in value):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    if isinstance(value, Mapping):
        fields = ", ".join(f"{key} = {_toml_scalar(item)}" for key, item in value.items())
        return "{" + fields + "}"
    return _toml_string(str(value))


def _powershell_executable() -> str | None:
    """Return the shell Codex 0.146 selects first on Windows, when present."""
    if sys.platform != "win32":
        return None
    return shutil.which("pwsh") or shutil.which("powershell")


def _powershell_literal(value: str) -> str:
    """Quote one PowerShell argument without interpolation or expansion."""
    return "'" + value.replace("'", "''") + "'"


def _validate_hook_interpreter_path(value: str, *, cmd_fallback: bool) -> None:
    """Reject path bytes that cannot be represented safely in a hook command."""
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise ValueError("sys.executable cannot contain NUL or newline characters")
    if cmd_fallback and any(character in value for character in ('"', "%")):
        raise ValueError(
            "sys.executable contains a quote or percent sign unsupported by Cmd hook quoting"
        )


def _codex_hook_command(integration_env: IntegrationEnv) -> str:
    """Return a hook command with explicit env without shell interpolation.

    Codex's command hook schema has no documented ``env`` field.  A compact
    Python bootstrap carries the allowlisted values as base64 JSON and sets
    them before running the adapter.  The command contains only generated
    interpreter/module syntax and an alphanumeric payload.
    """
    _validate_hook_interpreter_path(sys.executable, cmd_fallback=False)
    powershell = _powershell_executable()
    if powershell is not None:
        interpreter = _powershell_literal(sys.executable)
        command_prefix = f"& {interpreter}"
    elif sys.platform == "win32":
        _validate_hook_interpreter_path(sys.executable, cmd_fallback=True)
        # Codex's tagged Windows detector falls back to Cmd when PowerShell
        # cannot be found. Percent expansion and embedded quotes cannot be
        # represented safely in the generated Cmd command.
        command_prefix = f'"{sys.executable}"'
    else:
        # Codex runs POSIX hooks through a shell. shlex.quote preserves
        # literal interpreter paths containing shell metacharacters.
        command_prefix = shlex.quote(sys.executable)
    command = f"{command_prefix} -m mnemos.adapters.codex"
    if integration_env is None:
        return command
    import base64

    encoded = base64.b64encode(
        json.dumps(dict(integration_env), separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    bootstrap = (
        "import base64,json,os,runpy;"
        f"os.environ.update(json.loads(base64.b64decode({encoded!r})));"
        "runpy.run_module('mnemos.adapters.codex',run_name='__main__')"
    )
    if powershell is not None:
        return f"{command_prefix} -c {_powershell_literal(bootstrap)}"
    return f'{command_prefix} -c "{bootstrap}"'


def _render_codex_hooks(
    existing: Any,
    *,
    integration_env: IntegrationEnv = None,
) -> list[str]:
    """Render Codex array-of-table hooks while preserving unrelated entries."""
    events: dict[str, list[dict[str, Any]]] = {}
    if isinstance(existing, Mapping):
        for event, groups in existing.items():
            if not isinstance(groups, list):
                continue
            kept: list[dict[str, Any]] = []
            for group in groups:
                if not isinstance(group, Mapping):
                    continue
                handlers = group.get("hooks")
                if isinstance(handlers, list):
                    retained_handlers = [
                        handler
                        for handler in handlers
                        if not (
                            isinstance(handler, Mapping)
                            and "mnemos.adapters.codex" in str(handler.get("command", ""))
                        )
                    ]
                    retained_group = dict(group)
                    retained_group["hooks"] = retained_handlers
                    if retained_handlers:
                        kept.append(retained_group)
                else:
                    kept.append(dict(group))
            events[str(event)] = kept

    # Codex executes command hooks as a command line.  Quote the interpreter
    # once for paths such as ``C:\\Program Files\\Python\\python.exe`` and
    # let the TOML serializer escape that command as a whole.
    command = _codex_hook_command(integration_env)
    generated: dict[str, dict[str, Any]] = {
        "SessionStart": {
            "matcher": "startup|resume|clear|compact",
            "additionalContextLimit": 1200,
        },
        "UserPromptSubmit": {"additionalContextLimit": 1200},
        "PreCompact": {"matcher": "manual|auto", "timeout": 10},
        # PostCompact's command output schema cannot carry additionalContext.
        "PostCompact": {"matcher": "manual|auto"},
        "Stop": {"timeout": 3},
        "SessionEnd": {"matcher": "other", "timeout": 3},
    }
    lines: list[str] = []
    for event, groups in events.items():
        for group in groups:
            lines.append(f"[[hooks.{event}]]")
            for key, value in group.items():
                if key == "hooks" or (
                    isinstance(value, list) and value and isinstance(value[0], Mapping)
                ):
                    continue
                lines.append(f"{key} = {_toml_scalar(value)}")
            handlers = group.get("hooks", [])
            if isinstance(handlers, list):
                for handler in handlers:
                    if not isinstance(handler, Mapping):
                        continue
                    lines.append("")
                    lines.append(f"[[hooks.{event}.hooks]]")
                    for key, value in handler.items():
                        if isinstance(value, list) and value and isinstance(value[0], Mapping):
                            continue
                        lines.append(f"{key} = {_toml_scalar(value)}")
            lines.append("")
    for event, options in generated.items():
        lines.append(f"[[hooks.{event}]]")
        if "matcher" in options:
            lines.append(f"matcher = {_toml_scalar(options['matcher'])}")
        lines.append("")
        lines.append(f"[[hooks.{event}.hooks]]")
        lines.append('type = "command"')
        lines.append(f"command = {_toml_string(command)}")
        if "timeout" in options:
            lines.append(f"timeout = {options['timeout']}")
        if "additionalContextLimit" in options:
            lines.append(f"additionalContextLimit = {options['additionalContextLimit']}")
        lines.append("")
    return lines[:-1] if lines else lines


def _render_host_config(
    host: HostName,
    *,
    config_path: Path,
    existing_text: str,
    mnemos_config_path: Path,
    cwd: Path,
    integration_env: IntegrationEnv = None,
) -> str:
    if host == "codex":
        return _merge_codex_config(
            existing_text,
            mnemos_config_path=mnemos_config_path,
            integration_env=integration_env,
        )

    if host == "hermes":
        return _merge_hermes_config(existing_text)

    if host == "cursor":
        return _merge_json_host_config(
            existing_text,
            command=sys.executable,
            args=["-m", "mnemos.mcp_server"],
            mnemos_config_path=mnemos_config_path,
            integration_env=integration_env,
        )

    wrapper = _find_plugin_wrapper(cwd)
    if wrapper is not None:
        return _merge_json_host_config(
            existing_text,
            command=sys.executable,
            args=[str(wrapper)],
            mnemos_config_path=mnemos_config_path,
            integration_env=integration_env,
        )
    return _merge_json_host_config(
        existing_text,
        command=sys.executable,
        args=["-m", "mnemos.mcp_server"],
        mnemos_config_path=mnemos_config_path,
        integration_env=integration_env,
    )


def preview_host_integration(
    host: HostName,
    *,
    mnemos_config_path: str | Path,
    cwd: str | Path | None = None,
    home: str | Path | None = None,
    integration_env: IntegrationEnv = None,
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
        integration_env=integration_env,
    )
    preview_text = _preview_diff(config_path, existing_text, rendered_config)
    if host == "cursor":
        rule_path = _cursor_rule_path(resolved_cwd)
        existing_rule_text = _read_text(rule_path)
        rendered_rule = build_antigravity_artifact("cursor", "cursor-rule")
        rule_diff = _preview_diff(rule_path, existing_rule_text, rendered_rule)
        if rule_diff:
            preview_text = "\n\n".join(part for part in (preview_text, rule_diff) if part)
    elif host == "claude-code":
        agents_dir = _claude_agents_dir(resolved_home)
        for agent_name in ("mnemos-recall.md", "mnemos-curator.md"):
            agent_path = agents_dir / agent_name
            agent_diff = _preview_diff(
                agent_path,
                _read_text(agent_path),
                _render_claude_agent(resolved_cwd, agent_name),
            )
            if agent_diff:
                preview_text = "\n\n".join(part for part in (preview_text, agent_diff) if part)
    elif host == "codex":
        agents_path = _codex_agents_path(resolved_cwd)
        existing_agents_text = _read_text(agents_path)
        rendered_agents = _merge_markdown_section(
            existing_agents_text,
            heading="Mnemos Memory",
            rendered_section=build_antigravity_artifact("codex", "codex-agents"),
        )
        agents_diff = _preview_diff(agents_path, existing_agents_text, rendered_agents)
        if agents_diff:
            preview_text = "\n\n".join(part for part in (preview_text, agents_diff) if part)
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
    integration_env: IntegrationEnv = None,
) -> HostIntegrationResult:
    preview = preview_host_integration(
        host,
        mnemos_config_path=mnemos_config_path,
        cwd=cwd,
        home=home,
        integration_env=integration_env,
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
    elif host == "claude-code":
        agents_dir = _claude_agents_dir(_home_path(home))
        agents_dir.mkdir(parents=True, exist_ok=True)
        for agent_name in ("mnemos-recall.md", "mnemos-curator.md"):
            agent_path = agents_dir / agent_name
            agent_path.write_text(
                _render_claude_agent(_cwd_path(cwd), agent_name),
                encoding="utf-8",
            )
    elif host == "codex":
        agents_path = _codex_agents_path(_cwd_path(cwd))
        agents_path.parent.mkdir(parents=True, exist_ok=True)
        existing_agents_text = _read_text(agents_path)
        if agents_path.exists():
            agents_backup_path = _backup_path(agents_path)
            agents_backup_path.write_text(existing_agents_text, encoding="utf-8")
        agents_path.write_text(
            _merge_markdown_section(
                existing_agents_text,
                heading="Mnemos Memory",
                rendered_section=build_antigravity_artifact("codex", "codex-agents"),
            ),
            encoding="utf-8",
        )
    elif host == "hermes":
        # Hermes setup owns provider-specific sidecar values.  This host
        # integration changes only the native provider selector above.
        pass
    return HostIntegrationResult(
        host=host,
        config_path=preview.config_path,
        backup_path=backup_path,
        preview_text=preview.preview_text,
    )
