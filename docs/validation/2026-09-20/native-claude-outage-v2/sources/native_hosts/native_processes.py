"""Disposable Codex/Claude phase execution; lifecycle acceptance is separate.

These commands preserve the observed native v9 invocation modes. No function
in this module runs a host on import. Callers must verify package, source,
protocol, queue exit, and provider bindings before executing a phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping
from uuid import uuid4

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

CLAUDE_PREFIX = "mcp__plugin_mnemos-memory_mnemos__"
MCP_TOOLS = frozenset(
    {
        "mnemos_store",
        "mnemos_consolidate",
        "mnemos_retrieve",
        "mnemos_inspect",
        "mnemos_correct",
        "mnemos_forget",
        "mnemos_forget_preview",
        "mnemos_forget_related",
        "mnemos_job_status",
    }
)


@dataclass(frozen=True)
class PhaseEnvironment:
    root: Path
    project: Path
    registry: Path
    descriptor: Path
    settings: Path
    profile: str
    plugin_snapshot: Path

    def validate(self) -> None:
        root = self.root.resolve(strict=True)
        for value in (
            self.project,
            self.registry,
            self.descriptor,
            self.settings,
            self.plugin_snapshot,
        ):
            if not value.resolve().is_relative_to(root):
                raise ValueError("Native phase resource escapes disposable root")
        if not self.profile.strip() or not self.project.is_dir():
            raise ValueError("Native phase requires a project and profile")
        if not self.registry.is_file() or not self.settings.is_file():
            raise ValueError("Native phase registry/settings are missing")
        if not self.plugin_snapshot.is_dir():
            raise ValueError("Native phase requires a disposable plugin snapshot")


def clean_environment(overrides: Mapping[str, str]) -> dict[str, str]:
    """Strip inherited memory selectors, import overrides, and provider secrets."""
    sensitive = {
        "PYTHONPATH",
        "PYTHONHOME",
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
        "CLAUDECODE",
        "OPENAI_API_KEY",
        "OPENAI_ORG_ID",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "MCP_CONNECTION_NONBLOCKING",
        "CLAUDE_SECURESTORAGE_CONFIG_DIR",
        "ANTHROPIC_CONFIG_DIR",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_PROJECT_ID",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    }
    result = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in sensitive and not key.upper().startswith("MNEMOS_")
    }
    result.update(overrides)
    result["PYTHONIOENCODING"] = "utf-8"
    return result


def _private(path: Path) -> None:
    from mnemos.service import _protect_private_path

    _protect_private_path(path)


def _auth_copy(source: Path, home: Path, root: Path) -> Path:
    if not home.resolve().is_relative_to(root.resolve()):
        raise ValueError("Authentication destination escapes disposable root")
    if not source.is_file():
        raise ValueError("Native host authentication source is unavailable")
    target = home / source.name
    if target.exists() or target.is_symlink():
        raise FileExistsError("Refusing to replace existing authentication")
    created = False
    try:
        with target.open("xb") as output:
            created = True
            _private(target)
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return target


def _selectors(config: PhaseEnvironment, host: str, phase: str) -> dict[str, str]:
    return {
        "MNEMOS_CONFIG_PATH": str(config.settings),
        "MNEMOS_SCOPE_REGISTRY": str(config.registry),
        "MNEMOS_USER_PROFILE_ID": config.profile,
        "MNEMOS_RUNTIME_DESCRIPTOR": str(config.descriptor),
        "MNEMOS_RUNTIME_DIR": str(config.root.resolve()),
        "MNEMOS_STORE_TYPE": "sqlite",
        "MNEMOS_SQLITE_PATH": str(config.root.resolve() / "memory.sqlite"),
        "MNEMOS_PROJECT_ROOT": str(config.project),
        "MNEMOS_HOST": host,
        "MNEMOS_SESSION_ID": "native-three-host-" + phase,
        "MNEMOS_AUTO_START": "false",
        "MNEMOS_ROUTING_REVIEW_DIR": str(config.root / "routing-review"),
    }


def _phase_root(config: PhaseEnvironment, phase: str) -> Path:
    config.validate()
    if not phase or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in phase):
        raise ValueError("Phase name must be a simple unique identifier")
    path = config.root / phase
    path.mkdir(exist_ok=False)
    _private(path)
    return path


def terminate_owned_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            timeout=15,
            check=False,
        )
        if result.returncode and process.poll() is None:
            raise RuntimeError("Could not terminate the owned native process tree")
    else:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def _codex_config_home(config_path: Path, requested_home: Path) -> Path:
    expected = (requested_home / ".codex").resolve()
    if config_path.resolve().parent != expected:
        raise ValueError("Generated Codex configuration escapes its disposable home")
    return expected


def _save_result(path: Path, result: dict[str, Any]) -> None:
    (path / "process.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _codex_observed_models(home: Path) -> list[Any]:
    """Read model observations only from the fresh disposable Codex home."""
    sessions = home / "sessions"
    observations: list[Any] = []
    if not sessions.is_dir():
        return observations
    for path in sorted(sessions.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") != "turn_context":
                continue
            payload = record.get("payload")
            observations.append(payload.get("model") if isinstance(payload, dict) else None)
    return observations


def _codex_user_messages(home: Path) -> list[Any]:
    """Read user-message events only from the fresh disposable Codex home."""
    sessions = home / "sessions"
    messages: list[Any] = []
    if not sessions.is_dir():
        return messages
    for path in sorted(sessions.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") != "event_msg":
                continue
            payload = record.get("payload")
            if isinstance(payload, dict) and payload.get("type") == "user_message":
                messages.append(payload.get("message"))
    return messages


def _cleanup_phase(
    directory: Path,
    result: dict[str, Any],
    process: subprocess.Popen[Any] | None,
    auth: Path | None,
    started: float,
) -> None:
    errors = []
    if process is not None:
        try:
            terminate_owned_process(process)
        except Exception as exc:
            errors.append("process_termination:" + type(exc).__name__)
        if process.poll() is None:
            errors.append("owned_process_still_running")
    if auth is not None:
        try:
            auth.unlink(missing_ok=True)
        except OSError as exc:
            errors.append("authentication_removal:" + type(exc).__name__)
    result["auth_copy_removed"] = auth is None or not auth.exists()
    result["cleanup_errors"] = errors
    result["seconds"] = time.monotonic() - started
    if errors or not result["auth_copy_removed"]:
        result["status"] = "failed"
    _save_result(directory, result)
    if errors or not result["auth_copy_removed"]:
        raise RuntimeError("Native phase cleanup failed; inspect process.json")


def _prepare_codex(
    config: PhaseEnvironment, directory: Path, phase: str
) -> tuple[Path, dict[str, str]]:
    from mnemos.hosts import apply_host_integration

    home = directory / "codex-home"
    home.mkdir()
    _private(home)
    selectors = _selectors(config, "codex", phase)
    generated = apply_host_integration(
        "codex",
        mnemos_config_path=config.settings,
        cwd=config.project,
        home=home,
        integration_env={
            key: selectors[key]
            for key in (
                "MNEMOS_CONFIG_PATH",
                "MNEMOS_SCOPE_REGISTRY",
                "MNEMOS_USER_PROFILE_ID",
                "MNEMOS_RUNTIME_DESCRIPTOR",
            )
        },
    )
    codex_home = _codex_config_home(generated.config_path, home)
    text = generated.config_path.read_text(encoding="utf-8")
    if "[windows]" not in text:
        text = text.rstrip() + '\n\n[windows]\nsandbox = "unelevated"\n'
    else:
        text = text.replace('sandbox = "workspace-write"', 'sandbox = "unelevated"')
    generated.config_path.write_text(text, encoding="utf-8")
    _private(generated.config_path)
    parsed = tomllib.loads(text)
    if parsed["windows"]["sandbox"] != "unelevated":
        raise ValueError("Generated native sandbox differs from observed configuration")
    entry = parsed["mcp_servers"]["mnemos"]
    if Path(entry["command"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("Generated MCP entrypoint is not the verified installed interpreter")
    env = clean_environment(
        {
            **selectors,
            "CODEX_HOME": str(codex_home),
            "TERM": "xterm-256color",
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        }
    )
    return codex_home, env


def run_codex_phase(
    config: PhaseEnvironment,
    *,
    phase: str,
    prompt: str,
    executable: Path,
    auth_source: Path,
    max_seconds: int = 180,
    disable_code_mode_host: bool = False,
    model: str | None = None,
    migration_notice_target: str | None = None,
) -> dict[str, Any]:
    """Run an interactive Codex phase in the caller's terminal and return provenance.

    Retain native session files privately for strict trace parsing by the caller.
    A zero process exit does not establish any memory-operation acceptance.
    """
    if max_seconds != 180:
        raise ValueError("The frozen Codex phase deadline is 180 seconds")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError("Native Codex requires the observed terminal invocation mode")
    if model is not None and (
        not isinstance(model, str) or re.fullmatch(r"[A-Za-z0-9._-]{1,128}", model) is None
    ):
        raise ValueError("Codex model must be a nonempty safe model slug")
    if migration_notice_target is not None:
        if model is None:
            raise ValueError("Codex migration notice requires a pinned model")
        if (
            not isinstance(migration_notice_target, str)
            or re.fullmatch(r"[A-Za-z0-9._-]{1,128}", migration_notice_target) is None
        ):
            raise ValueError("Codex migration notice target must be a nonempty safe model slug")
    directory = _phase_root(config, phase)
    command = [
        str(executable),
        *(["--disable", "code_mode_host"] if disable_code_mode_host else []),
        "--sandbox",
        "read-only",
        "--ask-for-approval",
        "on-request",
        "--cd",
        str(config.project),
        "--no-alt-screen",
        *(
            [
                "-c",
                "notice.model_migrations="
                + "{"
                + json.dumps(model)
                + "="
                + json.dumps(migration_notice_target)
                + "}",
            ]
            if migration_notice_target is not None
            else []
        ),
        *(["--model", model] if model is not None else []),
        prompt,
    ]
    result: dict[str, Any] = {
        "host": "codex",
        "phase": phase,
        "status": "running",
        "terminal_exit": "Controller must exit the TUI after the completed native turn, within the deadline",
        "command": command[:-1] + ["<synthetic phase prompt>"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "max_seconds": max_seconds,
        "code_mode_host_disabled": disable_code_mode_host,
        "requested_model": model,
        "migration_notice_pair": (
            {"from": model, "to": migration_notice_target}
            if migration_notice_target is not None
            else None
        ),
    }
    auth = None
    process = None
    started = time.monotonic()
    try:
        codex_home, env = _prepare_codex(config, directory, phase)
        result["home"] = str(codex_home)
        auth = _auth_copy(auth_source, codex_home, config.root)
        process = subprocess.Popen(
            command,
            cwd=config.project,
            env=env,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        result["pid"] = process.pid
        _save_result(directory, result)
        try:
            process.wait(timeout=max_seconds)
            result["timed_out"] = False
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            terminate_owned_process(process)
        result["returncode"] = process.returncode
        result["status"] = (
            "completed" if process.returncode == 0 and not result["timed_out"] else "failed"
        )
        if model is not None and result["status"] == "completed":
            observed = _codex_observed_models(codex_home)
            result["observed_models"] = observed
            result["model_verified"] = bool(observed) and all(item == model for item in observed)
            if not result["model_verified"]:
                raise ValueError("Codex model observation is missing or mismatched")
            user_messages = _codex_user_messages(codex_home)
            result["user_prompt_count"] = len(user_messages)
            result["user_prompt_hashes"] = [
                hashlib.sha256(message.encode()).hexdigest()
                for message in user_messages
                if isinstance(message, str)
            ]
            result["user_prompt_verified"] = len(user_messages) == 1 and user_messages[0] == prompt
            if not result["user_prompt_verified"]:
                raise ValueError("Codex user prompt observation is missing or mismatched")
    except BaseException as exc:
        result["status"] = "failed"
        result["exception_type"] = type(exc).__name__
        raise
    finally:
        _cleanup_phase(directory, result, process, auth, started)
    return result


def run_claude_phase(
    config: PhaseEnvironment,
    *,
    phase: str,
    prompt: str,
    executable: Path,
    auth_source: Path,
    allowed_tools: list[str],
    max_budget_usd: float = 0.50,
    max_seconds: int = 180,
) -> dict[str, Any]:
    """Run a fresh Claude conversation, retaining raw stream and exit evidence."""
    if max_seconds != 180 or not 0 < max_budget_usd <= 0.50:
        raise ValueError("Native Claude phase exceeds the frozen resource bounds")
    if (
        not allowed_tools
        or len(set(allowed_tools)) != len(allowed_tools)
        or any(name not in {CLAUDE_PREFIX + tool for tool in MCP_TOOLS} for name in allowed_tools)
    ):
        raise ValueError("Only named Mnemos tools may be enabled")
    directory = _phase_root(config, phase)
    home = directory / "claude-home"
    home.mkdir()
    _private(home)
    checkpoint = directory / "checkpoints"
    checkpoint.mkdir()
    _private(checkpoint)
    session_id = str(uuid4())
    env = clean_environment(
        {
            **_selectors(config, "claude-code", phase),
            "CLAUDE_CONFIG_DIR": str(home),
            "MNEMOS_PLUGIN_PYTHON": sys.executable,
            "MNEMOS_CLAUDE_CHECKPOINT_DIR": str(checkpoint),
            "DISABLE_AUTOUPDATER": "1",
            "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        }
    )
    command = [
        str(executable),
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--include-hook-events",
        "--setting-sources",
        "project",
        "--plugin-dir",
        str(config.plugin_snapshot),
        "--tools",
        "",
        "--allowed-tools",
        ",".join(allowed_tools),
        "--disable-slash-commands",
        "--permission-mode",
        "default",
        "--session-id",
        session_id,
        "--model",
        "claude-sonnet-4-6",
        "--max-budget-usd",
        str(max_budget_usd),
        "--system-prompt",
        "Use only the explicitly allowed Mnemos tools. Complete the requested lifecycle in order.",
        prompt,
    ]
    result: dict[str, Any] = {
        "host": "claude-code",
        "phase": phase,
        "session_id": session_id,
        "status": "running",
        "home": str(home),
        "command": command[:-1] + ["<synthetic phase prompt>"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "allowed_tools": allowed_tools,
        "max_seconds": max_seconds,
        "max_budget_usd": max_budget_usd,
    }
    auth = None
    process = None
    started = time.monotonic()
    try:
        auth = _auth_copy(auth_source, home, config.root)
        with (
            (directory / "stdout.jsonl").open("wb") as stdout,
            (directory / "stderr.log").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=config.project,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            result["pid"] = process.pid
            _save_result(directory, result)
            try:
                process.wait(timeout=max_seconds)
                result["timed_out"] = False
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                terminate_owned_process(process)
        result["returncode"] = process.returncode
        result["status"] = (
            "completed" if process.returncode == 0 and not result["timed_out"] else "failed"
        )
        result["stdout_sha256"] = hashlib.sha256(
            (directory / "stdout.jsonl").read_bytes()
        ).hexdigest()
        result["stderr_sha256"] = hashlib.sha256(
            (directory / "stderr.log").read_bytes()
        ).hexdigest()
    except BaseException as exc:
        result["status"] = "failed"
        result["exception_type"] = type(exc).__name__
        raise
    finally:
        _cleanup_phase(directory, result, process, auth, started)
    return result
