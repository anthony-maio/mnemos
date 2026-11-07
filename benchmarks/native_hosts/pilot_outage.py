"""Acceptance and controlled recovery for an isolated native pilot service outage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any, Callable

PROMPT = (
    "For this disposable project, the outage test marker is birch-window-842. "
    "Do not call any tools or claim that memory storage succeeded. "
    "Reply with exactly: Acknowledged birch-window-842."
)
ACK = "Acknowledged birch-window-842."


def snapshot(database: Path) -> dict[str, Any]:
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            "events": "episodic_events",
            "jobs": "memory_jobs",
            "chunks": "memory_chunks",
            "evidence": "fact_evidence",
            "revisions": "fact_revisions",
        }
        result = {
            name: sorted(
                (dict(row) for row in connection.execute(f"SELECT * FROM {table}")),
                key=lambda row: json.dumps(row, sort_keys=True, default=str),
            )
            for name, table in tables.items()
        }
        if (
            connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
            or connection.execute("PRAGMA foreign_key_check").fetchall()
        ):
            raise ValueError("Recovered database integrity failed")
    return result


def require_no_tools(rows: list[dict[str, Any]], final: str) -> None:
    if final != ACK:
        raise ValueError("Native outage acknowledgement differs")
    forbidden = {
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "tool_search_call",
        "tool_search_output",
        "tool_use",
        "tool_result",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") in forbidden:
                raise ValueError("Native outage turn used a tool")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(rows)


def require_buffered(
    entries: list[dict[str, Any]], *, session_id: str, turn_id: str, project_id: str, profile: str
) -> None:
    if len(entries) != 2 or len({entry.get("event_id") for entry in entries}) != 2:
        raise ValueError("Require exactly two distinct buffered messages")
    expected = {"user": (PROMPT, "UserPromptSubmit"), "assistant": (ACK, "Stop")}
    if {entry.get("payload", {}).get("role") for entry in entries} != set(expected):
        raise ValueError("Buffered message roles differ")
    for entry in entries:
        payload = entry["payload"]
        content, hook = expected[payload["role"]]
        context = payload.get("context", {})
        if (
            entry.get("operation") != "capture"
            or payload.get("content") != content
            or payload.get("source_ref") != hook + ":" + turn_id
            or payload.get("source_kind") != "codex_hook"
            or payload.get("identity_status", "normal") != "normal"
            or context.get("host") != "codex"
            or context.get("session_id") != session_id
            or context.get("project_id") != project_id
            or context.get("user_profile_id") != profile
            or context.get("scope", "project") != "project"
            or context.get("scope_id", project_id) != project_id
        ):
            raise ValueError("Buffered native message identity or content differs")


def require_recovered(
    state: dict[str, Any], before: dict[str, Any], entries: list[dict[str, Any]], *, completed: bool
) -> None:
    prior_events = {row["id"]: row for row in before["events"]}
    events = {row["id"]: row for row in state["events"]}
    jobs = {row["id"]: row for row in state["jobs"]}
    if len(events) != len(prior_events) + 2 or len(jobs) != len(before["jobs"]) + 2:
        raise ValueError("Recovered event or job count differs")
    if any(events.get(key) != value for key, value in prior_events.items()):
        raise ValueError("Retained source control changed")
    for entry in entries:
        payload = entry["payload"]
        event = events.get(entry["event_id"], {})
        job = jobs.get("event:" + entry["event_id"], {})
        metadata = json.loads(event.get("metadata", "{}"))
        context = payload["context"]
        if (
            event.get("content") != payload["content"]
            or event.get("role") != payload["role"]
            or event.get("scope") != "project"
            or event.get("scope_id") != context["project_id"]
            or event.get("source") != "codex"
            or event.get("user_profile_id") != context["user_profile_id"]
            or event.get("session_id") != context["session_id"]
            or metadata.get("source_event_id") != payload["event_id"]
            or metadata.get("source_ref") != payload["source_ref"]
            or metadata.get("identity_status") != "normal"
            or job.get("kind") != "extract"
            or json.loads(job.get("payload", "{}")).get("event_id") != entry["event_id"]
        ):
            raise ValueError("Recovered physical source identity or job differs")
        if completed and (
            event.get("status") != "consolidated" or job.get("status") != "completed"
        ):
            raise ValueError("Recovered processing did not complete")
    prior_chunks = {row["id"]: row for row in before["chunks"]}
    chunks = {row["id"]: row for row in state["chunks"]}
    if any(chunks.get(key) != value for key, value in prior_chunks.items()):
        raise ValueError("Retained fact control changed")
    new_chunks = [row for key, row in chunks.items() if key not in prior_chunks]
    if completed and len(new_chunks) != 2:
        raise ValueError("Completed capture did not produce two raw chunks")
    for chunk in new_chunks:
        metadata = json.loads(chunk["metadata"])
        if (
            metadata.get("source") != "captured_passthrough"
            or metadata.get("assertion_status") != "unknown"
        ):
            raise ValueError("Raw recovery was promoted to accepted facts")


def remaining_service_pids(descriptor: Path) -> list[int]:
    """Inspect only Python service commands naming this disposable descriptor."""
    literal = "'" + str(descriptor).replace("'", "''") + "'"
    command = (
        "$owned = @(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
        "Where-Object { $_.CommandLine -and $_.CommandLine.Contains('mnemos.service') -and "
        "$_.CommandLine.Contains(" + literal + ") } | Select-Object -ExpandProperty ProcessId); "
        "ConvertTo-Json -InputObject $owned -Compress"
    )
    result = subprocess.check_output(
        ["pwsh", "-NoProfile", "-Command", command],
        text=True,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return [int(value) for value in json.loads(result)]


async def recover(
    root: Path,
    project: Path,
    environment: dict[str, str],
    before: dict[str, Any],
    entries: list[dict[str, Any]],
    stop_service: Callable[[], None],
    remember_owner: Callable[[], None],
) -> dict[str, Any]:
    from mnemos.client import MnemosClient, Outbox

    def save(name: str, value: Any) -> None:
        (root / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def health() -> dict[str, Any]:
        result = subprocess.run(
            [sys.executable, "-I", str(root / "launch.py"), "cli", "codex", "stats"],
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode:
            raise ValueError("Ordinary recovery health request failed")
        value: dict[str, Any] = json.loads(result.stdout)
        runtime = value.get("runtime", {})
        if (
            value.get("status") != "running"
            or Path(runtime.get("database_path", "")).resolve() != root / "memory.sqlite"
            or runtime.get("cognition_provider") != "UnavailableLLMProvider"
            or runtime.get("embedding_provider") != "FastembedProvider"
        ):
            raise ValueError("Recovered service configuration differs")
        remember_owner()
        return value

    first_health = await asyncio.to_thread(health)
    outbox = Outbox(root / "runtime/outbox.json")
    if outbox.pending():
        raise ValueError("Ordinary health request did not drain buffered captures")
    drained = snapshot(root / "memory.sqlite")
    save("state-after-drain.json", drained)
    require_recovered(drained, before, entries, completed=False)
    client = MnemosClient(
        descriptor_path=root / "runtime/runtime.json", outbox=outbox, auto_start=False
    )
    try:
        maintenance = await client.request("maintenance", {})
        save("recovery-maintenance.json", maintenance)
        if maintenance.get("degraded") or maintenance.get("status") in {"degraded", "error"}:
            raise ValueError("Recovered maintenance failed")
        deadline = time.monotonic() + 120
        while True:
            processed = snapshot(root / "memory.sqlite")
            if (
                all(row["status"] == "completed" for row in processed["jobs"])
                or time.monotonic() >= deadline
            ):
                break
            await asyncio.sleep(0.25)
        require_recovered(processed, before, entries, completed=True)
        save("state-before-replay.json", processed)
        replay = []
        for entry in entries:
            result = await client.request("capture", entry["payload"], queue_on_failure=False)
            if result.get("status") != "committed":
                raise ValueError("Exact source replay did not commit idempotently")
            replay.append(result)
        save("exact-source-replay.json", replay)
        if snapshot(root / "memory.sqlite") != processed or outbox.pending():
            raise ValueError("Exact source replay changed recovered records")
    finally:
        await client.close()
    await asyncio.to_thread(stop_service)
    second_health = await asyncio.to_thread(health)
    restarted = snapshot(root / "memory.sqlite")
    save("state-after-second-restart.json", restarted)
    if restarted != processed or outbox.pending():
        raise ValueError("Second restart changed recovered records")
    require_recovered(restarted, before, entries, completed=True)
    save("recovery-health.json", [first_health, second_health])
    return {
        "status": "passed",
        "buffered_native_messages": len(entries),
        "ordinary_health_drained_outbox": True,
        "completed_capture_jobs": 2,
        "raw_unasserted_chunks": 2,
        "automatic_accepted_facts": 0,
        "exact_replays": len(replay),
        "second_restart_preserved_rows": True,
        "retained_seed_unchanged": True,
    }
