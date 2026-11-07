"""Observe normal Codex hook trust without granting trust or starting a turn."""

from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
from threading import Thread
import time
from typing import Any

from .native_processes import clean_environment, terminate_owned_process

EVENT_NAMES = {
    "sessionStart": "SessionStart",
    "userPromptSubmit": "UserPromptSubmit",
    "preCompact": "PreCompact",
    "postCompact": "PostCompact",
    "stop": "Stop",
    "sessionEnd": "SessionEnd",
}


def validate_inventory(
    result: dict[str, Any],
    *,
    project: Path,
    config_path: Path,
    expected_hooks: dict[str, Any],
    require_trusted: bool = False,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Require the exact six user hooks; an untrusted observation is not a pass."""
    entries = result.get("data")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise ValueError("Expected exactly one hook inventory")
    entry = entries[0]
    if Path(entry.get("cwd", "")).resolve() != project.resolve() or entry.get("errors") != []:
        raise ValueError("Hook inventory has errors or a different project")
    hooks = entry.get("hooks")
    if not isinstance(hooks, list) or len(hooks) != len(expected_hooks) or len(hooks) != 6:
        raise ValueError("Hook inventory must contain exactly six reviewed hooks")
    observed: dict[str, dict[str, Any]] = {}
    keys: set[str] = set()
    for hook in hooks:
        if not isinstance(hook, dict):
            raise ValueError("Malformed hook metadata")
        native_event = hook.get("eventName")
        event = EVENT_NAMES.get(native_event) if isinstance(native_event, str) else None
        if not isinstance(event, str) or event not in expected_hooks or event in observed:
            raise ValueError("Unexpected or duplicate hook event")
        groups = expected_hooks[event]
        if len(groups) != 1 or len(groups[0]["hooks"]) != 1:
            raise ValueError("Expected configuration is not the reviewed single-hook shape")
        group, handler = groups[0], groups[0]["hooks"][0]
        expected = {
            "handlerType": "command",
            "command": handler["command"],
            "matcher": group.get("matcher"),
            "timeoutSec": handler.get(
                "timeout", 1 if event in {"SessionEnd", "Interrupt"} else 600
            ),
            "additionalContextLimit": handler.get("additionalContextLimit"),
            "source": "user",
            "isManaged": False,
            "enabled": True,
        }
        if any(hook.get(key) != value for key, value in expected.items()):
            raise ValueError("Hook command, source or execution settings differ: " + event)
        if hook.get("enabled") is not True or hook.get("isManaged") is not False:
            raise ValueError("Hook enablement and ownership must be native booleans")
        if hook.get("async", False) is not False:
            raise ValueError("Unexpected asynchronous hook")
        source = hook.get("sourcePath")
        if not isinstance(source, str) or Path(source).resolve() != config_path.resolve():
            raise ValueError("Hook came from a different configuration file")
        key, digest, status = hook.get("key"), hook.get("currentHash"), hook.get("trustStatus")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError("Missing or duplicate hook key")
        if not isinstance(digest, str) or not digest:
            raise ValueError("Missing current hook hash")
        if status not in {"untrusted", "trusted", "modified"}:
            raise ValueError("Unexpected hook trust state")
        if require_trusted and status != "trusted":
            raise ValueError("Normal hook review is still required: " + event)
        keys.add(key)
        observed[event] = {"key": key, "current_hash": digest, "trust_status": status}
    hashes = {name: value["current_hash"] for name, value in observed.items()}
    if expected_hashes is not None and hashes != expected_hashes:
        raise ValueError("Hook definitions changed since review preparation")
    return {
        "status": (
            "trusted"
            if all(row["trust_status"] == "trusted" for row in observed.values())
            else "review_required"
        ),
        "hooks": observed,
        "current_hashes": hashes,
        "warnings": entry.get("warnings", []),
    }


def read_inventory(
    executable: Path, *, home: Path, project: Path, output: Path, timeout: float = 30
) -> dict[str, Any]:
    """Use only initialize and hooks/list on an owned stdio app server."""
    if not home.is_dir() or not project.is_dir() or output.exists():
        raise ValueError("Use existing isolated home/project and a fresh output directory")
    output.mkdir(parents=True)
    messages: Queue[bytes | None] = Queue()
    command = [str(executable), "app-server", "--listen", "stdio://", "--enable", "hooks"]
    record: dict[str, Any] = {"command": command, "methods": [], "model_turns": 0}
    process: subprocess.Popen[bytes] | None = None
    try:
        with (output / "stderr.log").open("wb") as errors:
            process = subprocess.Popen(
                command,
                cwd=project,
                env=clean_environment({"CODEX_HOME": str(home)}),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errors,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            record["pid"] = process.pid
            assert process.stdout is not None and process.stdin is not None

            def collect() -> None:
                assert process is not None and process.stdout is not None
                for line in process.stdout:
                    messages.put(line)
                messages.put(None)

            reader = Thread(target=collect, daemon=True)
            reader.start()
            transcript: list[Any] = []
            deadline = time.monotonic() + timeout

            def request(method: str, params: dict[str, Any], identity: int | None) -> Any:
                assert process is not None and process.stdin is not None
                payload: dict[str, Any] = {"method": method, "params": params}
                if identity is not None:
                    payload["id"] = identity
                record["methods"].append(method)
                process.stdin.write(json.dumps(payload).encode() + b"\n")
                process.stdin.flush()
                if identity is None:
                    return None
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Native hook inventory timed out")
                    try:
                        raw = messages.get(timeout=remaining)
                    except Empty as error:
                        raise TimeoutError("Native hook inventory timed out") from error
                    if raw is None:
                        raise ValueError("App server ended before its inventory response")
                    if len(raw) > 2_000_000:
                        raise ValueError("Unexpectedly large app-server reply")
                    row = json.loads(raw)
                    transcript.append(row)
                    if row.get("id") == identity:
                        if "error" in row or "result" not in row:
                            raise ValueError("Native inventory request failed: " + method)
                        return row["result"]

            request(
                "initialize",
                {
                    "clientInfo": {"name": "mnemos_hook_review", "version": "1"},
                    "capabilities": {"experimentalApi": True},
                },
                1,
            )
            request("initialized", {}, None)
            result = request("hooks/list", {"cwds": [str(project)]}, 2)
            (output / "responses.json").write_text(
                json.dumps(transcript, indent=2) + "\n", encoding="utf-8"
            )
            (output / "inventory.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            process.stdin.close()
            process.wait(timeout=10)
            reader.join(timeout=1)
            record.update(returncode=process.returncode, status="completed")
            if process.returncode != 0:
                raise ValueError("Inventory process did not exit cleanly")
            return dict(result)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if process is not None:
            terminate_owned_process(process)
            record["process_exited"] = process.poll() is not None
        (output / "process.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
