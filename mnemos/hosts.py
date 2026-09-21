"""
mnemos/hosts.py — Host integration preview/apply helpers for onboarding.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import unified_diff
from pathlib import Path
from typing import Any, Literal, Mapping

from .antigravity import build_antigravity_artifact

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
        "MNEMOS_AUTO_START",
        "MNEMOS_RUNTIME_DIR",
        "MNEMOS_STORE_TYPE",
        "MNEMOS_SQLITE_PATH",
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
    known_host: Literal["claude-code"] | None = None,
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
    if known_host is not None:
        env["MNEMOS_HOST"] = known_host
    mnemos_entry["command"] = command
    mnemos_entry["env"] = env
    mnemos_entry["args"] = args or []
    if known_host == "claude-code":
        # Claude defers MCP startup by default.  Keep an existing explicit
        # false so onboarding does not override a user's host preference.
        mnemos_entry.setdefault("alwaysLoad", True)
    servers["mnemos"] = mnemos_entry
    return json.dumps(payload, indent=2) + "\n"


_YAML_MEMORY_KEY = re.compile(r"memory[ \t]*:[ \t]*(#.*)?")
_YAML_PROVIDER_KEY = re.compile(r"(?P<key>provider[ \t]*:)[^#\n]*?(?P<comment>[ \t]+#.*)?")


def _edit_hermes_provider(existing_text: str) -> str:
    """Point the top-level memory block at Mnemos, editing lines in place."""
    lines = existing_text.splitlines()
    memory_start = next(
        (index for index, line in enumerate(lines) if _YAML_MEMORY_KEY.fullmatch(line.rstrip())),
        None,
    )
    if memory_start is None:
        if any(re.match(r"memory[ \t]*:", line) for line in lines):
            # A flow or scalar memory value. Appending a second memory key
            # would shadow it, so leave the text for verification to judge.
            return existing_text
        base = existing_text.rstrip()
        return (base + "\n\n" if base else "") + "memory:\n  provider: mnemos\n"
    memory_end = len(lines)
    for index in range(memory_start + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "\t", "#")):
            memory_end = index
            break
    children = [
        index
        for index in range(memory_start + 1, memory_end)
        if lines[index].strip() and not lines[index].lstrip().startswith("#")
    ]
    # Only a direct child of memory counts; a nested provider belongs to
    # some other setting, such as memory.embeddings.provider.
    indentation = (
        lines[children[0]][: len(lines[children[0]]) - len(lines[children[0]].lstrip())]
        if children
        else "  "
    )
    provider_line = next(
        (
            index
            for index in children
            if lines[index].startswith(indentation)
            and _YAML_PROVIDER_KEY.fullmatch(lines[index][len(indentation) :].rstrip())
        ),
        None,
    )
    if provider_line is None:
        lines.insert(memory_start + 1, f"{indentation}provider: mnemos")
    else:
        match = _YAML_PROVIDER_KEY.fullmatch(lines[provider_line][len(indentation) :].rstrip())
        assert match is not None
        comment = match.group("comment") or ""
        lines[provider_line] = f"{indentation}provider: mnemos{comment}"
    return "\n".join(lines).rstrip() + "\n"


def _merge_hermes_config(existing_text: str) -> str:
    """Set only Hermes' provider selector while retaining its YAML settings.

    The file is edited as text so comments and layout survive, then both
    versions are parsed. Unless memory.provider is the only difference,
    nothing is written, since Hermes keeps the last of two duplicate keys
    and a bad edit can silently drop the rest of its memory settings.
    """
    import yaml

    unsafe = (
        "Hermes config.yaml uses a layout setup cannot edit safely; nothing was written. "
        "Set memory.provider to mnemos by hand."
    )
    try:
        before = yaml.safe_load(existing_text)
    except yaml.YAMLError as exc:
        raise ValueError("Hermes config.yaml could not be parsed; nothing was written") from exc
    merged = _edit_hermes_provider(existing_text)
    try:
        after = yaml.safe_load(merged)
    except yaml.YAMLError as exc:
        raise ValueError(unsafe) from exc
    if before is None:
        before = {}
    if not isinstance(before, Mapping):
        raise ValueError("Hermes config.yaml is not a mapping; nothing was written")
    memory = before.get("memory")
    expected = {
        **before,
        "memory": {**(memory if isinstance(memory, Mapping) else {}), "provider": "mnemos"},
    }
    if after != expected or not isinstance(memory, (Mapping, type(None))):
        raise ValueError(unsafe)
    return merged


_CODEX_HOOK_MARKER = "mnemos.adapters.codex"
_CODEX_HOOK_EVENTS: dict[str, dict[str, Any]] = {
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


def _is_mnemos_codex_hook(handler: Any) -> bool:
    return isinstance(handler, Mapping) and _CODEX_HOOK_MARKER in str(handler.get("command", ""))


def codex_config_without_mnemos(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a parsed Codex config with every Mnemos-owned entry removed.

    Setup must leave this view unchanged, and checks its own output with it
    before writing.
    """
    data = copy.deepcopy(dict(payload))
    servers = data.get("mcp_servers")
    if isinstance(servers, dict):
        servers.pop("mnemos", None)
        if not servers:
            data.pop("mcp_servers")
    hooks = data.get("hooks")
    if isinstance(hooks, dict):
        for event in list(hooks):
            groups = hooks[event]
            if not isinstance(groups, list):
                continue
            kept = []
            for group in groups:
                handlers = group.get("hooks") if isinstance(group, dict) else None
                if isinstance(handlers, list):
                    retained = [item for item in handlers if not _is_mnemos_codex_hook(item)]
                    if not retained:
                        continue
                    group = {**group, "hooks": retained}
                kept.append(group)
            if kept:
                hooks[event] = kept
            else:
                del hooks[event]
        if not hooks:
            data.pop("hooks")
    return data


def _load_toml(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    if sys.version_info >= (3, 11):
        return tomllib.loads(text)
    return tomli.loads(text)  # pragma: no cover - exercised on Python 3.10


def _merge_codex_config(
    existing_text: str,
    *,
    mnemos_config_path: Path,
    integration_env: IntegrationEnv = None,
) -> str:
    """Edit only Mnemos entries in a Codex config and keep everything else as written.

    Comments, quoting and layout survive because the document is edited in
    place. The result is parsed back and must match the original outside the
    Mnemos-owned entries, or setup raises before anything is written.
    Replacing existing Mnemos hooks can shift blank lines on the first rerun,
    so the merge is repeated until it stops changing and setup is idempotent.
    """
    rendered = existing_text
    if rendered and not rendered.endswith("\n"):
        # tomlkit glues appended tables onto an unterminated last line.
        rendered += "\n"
    for _ in range(4):
        merged = _merge_codex_config_once(
            rendered, mnemos_config_path=mnemos_config_path, integration_env=integration_env
        )
        if merged == rendered:
            return merged
        rendered = merged
    raise ValueError("Codex setup output did not settle; nothing was written")


def _merge_codex_config_once(
    existing_text: str,
    *,
    mnemos_config_path: Path,
    integration_env: IntegrationEnv,
) -> str:
    import tomlkit
    from tomlkit.items import AoT

    before = _load_toml(existing_text)
    document = tomlkit.parse(existing_text) if existing_text.strip() else tomlkit.document()
    servers = document.get("mcp_servers")
    if servers is None:
        servers = tomlkit.table(is_super_table=True)
        document["mcp_servers"] = servers
    previous = servers.get("mnemos")
    entry = tomlkit.table()
    if isinstance(previous, Mapping):
        for key, value in previous.items():
            if key not in {"command", "args", "env"}:
                entry[key] = value
    # MCP and hooks must use the installation that generated this config.
    # A bare console script can resolve to an older installation on PATH.
    entry["command"] = sys.executable
    entry["args"] = ["-m", "mnemos.mcp_server"]
    env = tomlkit.table()
    previous_env = previous.get("env") if isinstance(previous, Mapping) else None
    if isinstance(previous_env, Mapping):
        for key, value in previous_env.items():
            env[key] = value
    rendered_env = _integration_environment(mnemos_config_path, integration_env)
    for key, value in {**rendered_env, "MNEMOS_HOST": "codex"}.items():
        env[key] = value
    entry["env"] = env
    servers["mnemos"] = entry

    hooks = document.get("hooks")
    if hooks is None:
        hooks = tomlkit.table(is_super_table=True)
        document["hooks"] = hooks
    for event in list(hooks.keys()):
        groups = hooks[event]
        if not isinstance(groups, AoT):
            continue
        for group_index in reversed(range(len(groups))):
            handlers = groups[group_index].get("hooks")
            if isinstance(handlers, AoT):
                for handler_index in reversed(range(len(handlers))):
                    if _is_mnemos_codex_hook(handlers[handler_index]):
                        del handlers[handler_index]
                if len(handlers) == 0:
                    del groups[group_index]
        if len(groups) == 0:
            del hooks[event]
    # Codex runs command hooks as a command line. The interpreter path is
    # quoted once inside the command; TOML escapes the command as a whole.
    command = _codex_hook_command(rendered_env if integration_env is not None else None)
    for event, options in _CODEX_HOOK_EVENTS.items():
        group = tomlkit.table()
        if "matcher" in options:
            group["matcher"] = options["matcher"]
        handler = tomlkit.table()
        handler["type"] = "command"
        handler["command"] = command
        for key in ("timeout", "additionalContextLimit"):
            if key in options:
                handler[key] = options[key]
        group_handlers = tomlkit.aot()
        group_handlers.append(handler)
        group["hooks"] = group_handlers
        if event not in hooks:
            hooks[event] = tomlkit.aot()
        existing_groups = hooks[event]
        if not isinstance(existing_groups, AoT):
            raise ValueError(f"Codex hooks.{event} uses a layout setup cannot edit safely")
        existing_groups.append(group)

    rendered = tomlkit.dumps(document)
    if codex_config_without_mnemos(_load_toml(rendered)) != codex_config_without_mnemos(before):
        raise ValueError("Codex setup would change settings it does not own; nothing was written")
    return rendered


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
            known_host="claude-code",
        )
    return _merge_json_host_config(
        existing_text,
        command=sys.executable,
        args=["-m", "mnemos.mcp_server"],
        mnemos_config_path=mnemos_config_path,
        integration_env=integration_env,
        known_host="claude-code",
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


def _write_backup(path: Path) -> Path:
    """Copy a file to a new backup that never replaces an earlier backup."""
    content = path.read_bytes()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    for attempt in range(1000):
        suffix = stamp if attempt == 0 else f"{stamp}-{attempt}"
        candidate = path.with_name(f"{path.name}.bak.{suffix}")
        try:
            with candidate.open("xb") as handle:
                handle.write(content)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"Could not create a unique backup for {path}")


def _write_text_atomic(path: Path, text: str) -> None:
    """Replace a file in one step so an interruption cannot leave it half written."""
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


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
        backup_path = _write_backup(preview.config_path)

    _write_text_atomic(preview.config_path, preview.rendered_config)
    if host == "cursor":
        rule_path = _cursor_rule_path(_cwd_path(cwd))
        rule_path.parent.mkdir(parents=True, exist_ok=True)
        if rule_path.exists():
            _write_backup(rule_path)
        _write_text_atomic(rule_path, build_antigravity_artifact("cursor", "cursor-rule"))
    elif host == "claude-code":
        agents_dir = _claude_agents_dir(_home_path(home))
        agents_dir.mkdir(parents=True, exist_ok=True)
        for agent_name in ("mnemos-recall.md", "mnemos-curator.md"):
            agent_path = agents_dir / agent_name
            rendered_agent = _render_claude_agent(_cwd_path(cwd), agent_name)
            # A hand-edited agent file is kept as a backup before replacement.
            if agent_path.exists() and _read_text(agent_path) != rendered_agent:
                _write_backup(agent_path)
            _write_text_atomic(agent_path, rendered_agent)
    elif host == "codex":
        agents_path = _codex_agents_path(_cwd_path(cwd))
        agents_path.parent.mkdir(parents=True, exist_ok=True)
        existing_agents_text = _read_text(agents_path)
        if agents_path.exists():
            _write_backup(agents_path)
        _write_text_atomic(
            agents_path,
            _merge_markdown_section(
                existing_agents_text,
                heading="Mnemos Memory",
                rendered_section=build_antigravity_artifact("codex", "codex-agents"),
            ),
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
