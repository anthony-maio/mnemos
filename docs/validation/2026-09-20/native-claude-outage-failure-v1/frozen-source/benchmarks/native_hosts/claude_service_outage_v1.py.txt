"""One frozen native Claude attempt while its disposable memory service is down."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
from typing import Any
import zipfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "native_hosts"

from .capture_coverage import expected_messages, reconcile_captures
from .native_processes import CLAUDE_PREFIX, PhaseEnvironment, run_claude_phase

WHEEL_SHA = "3d37b864d35916dbed0c22bcb0428a4e091d6b6c31a0177491130fa18da34fa1"
PROTOCOL = "docs/plans/2026-09-20-native-claude-service-outage-v1.md"
PROMPT = (
    "For this disposable project, the outage test marker is birch-window-618. "
    "Do not call any tools or claim that memory storage succeeded. "
    "Reply with exactly: Acknowledged birch-window-618."
)


def save(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode())


def rows(database: Path) -> dict[str, Any]:
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return {
            "events": [
                dict(row) for row in connection.execute("SELECT * FROM episodic_events ORDER BY id")
            ],
            "jobs": [
                dict(row) for row in connection.execute("SELECT * FROM memory_jobs ORDER BY id")
            ],
            "fact_count": connection.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0],
            "integrity": connection.execute("PRAGMA integrity_check").fetchone()[0],
        }
    finally:
        connection.close()


def stable(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "events": [
            {key: row[key] for key in ("id", "role", "content", "metadata", "status")}
            for row in state["events"]
        ],
        "jobs": [
            {key: row[key] for key in ("id", "kind", "status", "payload")} for row in state["jobs"]
        ],
        "fact_count": state["fact_count"],
    }


def require_pending_sources(state: dict[str, Any], pending: list[dict[str, Any]]) -> None:
    """Verify persisted raw sources and pending jobs, independently of receipts."""
    events = {row["id"]: row for row in state["events"]}
    jobs = {row["id"]: row for row in state["jobs"]}
    if (
        not pending
        or len({row["event_id"] for row in pending}) != len(pending)
        or len(events) != len(pending)
        or len(jobs) != len(pending)
        or state["fact_count"] != 0
        or state["integrity"] != "ok"
    ):
        raise ValueError("raw recovery counts or database integrity differ")
    for entry in pending:
        payload = entry["payload"]
        event = events.get(entry["event_id"], {})
        job = jobs.get("event:" + entry["event_id"], {})
        metadata = json.loads(event.get("metadata", "{}"))
        identity_status = payload.get("identity_status", "normal")
        provisional = identity_status == "provisional_unassociated"
        expected_metadata = {
            **payload["context"],
            "source_event_id": payload["event_id"],
            "source_kind": payload["source_kind"],
            "source_ref": payload["source_ref"],
            "identity_status": identity_status,
        }
        if (
            entry["operation"] != "capture"
            or event.get("role") != payload["role"]
            or event.get("content") != payload["content"]
            or event.get("status") != ("deferred" if provisional else "pending")
            or any(metadata.get(key) != value for key, value in expected_metadata.items())
            or job.get("kind") != ("capture_only" if provisional else "extract")
            or job.get("status") != "pending"
            or json.loads(job.get("payload", "{}")).get("event_id") != entry["event_id"]
        ):
            raise ValueError("raw source or pending job differs from the native outbox")


def main() -> int:
    if len(sys.argv) not in {2, 3} or (len(sys.argv) == 3 and sys.argv[2] != "--preflight"):
        raise ValueError("Pass one fresh absolute disposable output directory")
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    if root.exists() or root.is_relative_to(repo):
        raise ValueError("output must be fresh and outside the repository")
    from mnemos.client import MnemosClient
    from mnemos.config import MnemosConfig, ModuleConfig
    from mnemos.engine import MnemosEngine
    from mnemos.memory_service import LocalMemoryService
    from mnemos.scopes import ScopeRegistry
    from mnemos.service import MnemosService, _protect_private_path
    from mnemos.utils.embeddings import OllamaEmbeddingProvider
    from mnemos.utils.llm import UnavailableLLMProvider
    from mnemos.utils.storage import SQLiteStore
    import mnemos

    old = json.loads(
        (repo / "docs/validation/2026-09-20/three-host-native-v7/native/report.json").read_bytes()
    )
    wheel = Path(old["bindings"]["wheel"])
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA:
        raise ValueError("verified product wheel changed")
    package = Path(mnemos.__file__).resolve().parent
    if package.is_relative_to(repo):
        raise ValueError("isolated installed interpreter is required")
    with zipfile.ZipFile(wheel) as zipped:
        for name in zipped.namelist():
            if name.startswith("mnemos/") and name.endswith(".py"):
                if (package.parent / name).read_bytes() != zipped.read(name):
                    raise ValueError("installed package differs from verified wheel")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    inputs = [
        PROTOCOL,
        "benchmarks/native_hosts/claude_service_outage_v1.py",
        "benchmarks/native_hosts/native_processes.py",
        "benchmarks/native_hosts/capture_coverage.py",
    ]
    hashes = {}
    for name in inputs:
        raw = (repo / name).read_bytes()
        committed = subprocess.check_output(["git", "-C", str(repo), "show", commit + ":" + name])
        if raw.replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
            raise ValueError("runner inputs must be committed before execution")
        hashes[name] = hashlib.sha256(raw).hexdigest()
    executable = Path.home() / ".local/bin/claude.exe"
    version = subprocess.check_output([str(executable), "--version"], text=True, timeout=15).strip()
    if "2.1.191" not in version.split():
        raise ValueError("Claude version changed")
    plugin_source = Path(
        "C:/Users/Ant/AppData/Local/Temp/mnemos-three-host-native-v7/plugin-snapshot"
    )
    plugin_bytes = {}
    for name, expected in old["bindings"]["plugin_files"].items():
        raw = (plugin_source / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("bound plugin bytes changed")
        plugin_bytes[name] = raw
    if len(sys.argv) == 3:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "runner_commit": commit,
                    "input_hashes": hashes,
                    "installed_package": str(package),
                    "claude_version": version,
                    "native_inference": False,
                }
            )
        )
        return 0
    root.mkdir(parents=True)
    _protect_private_path(root)
    project = root / "project"
    project.mkdir()
    profile = "native-claude-outage-v1"
    registry = ScopeRegistry()
    project_entry = registry.register_project(project, user_profile_id=profile, enabled=True)
    registry.save(root / "scopes.json")
    settings = root / "settings.toml"
    settings.write_text(
        "[modules]\nsurprisal=false\nmutable_rag=false\naffective=false\nsleep=false\nspreading=false\n",
        encoding="utf-8",
    )
    for name, raw in plugin_bytes.items():
        destination = root / "plugin" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    descriptor = root / "runtime.json"
    database = root / "memory.sqlite"
    dispatches: list[dict[str, Any]] = []
    embedding_responses: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "runner_commit": commit,
        "input_hashes": hashes,
        "wheel_sha256": WHEEL_SHA,
        "installed_package": str(package),
        "claude_version": version,
        "claude_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "model_requested": "claude-sonnet-4-6",
        "background_processing_enabled": False,
        "local_provider_calls_expected": 0,
    }
    save(root / "report.json", report)

    class TracedService(LocalMemoryService):
        async def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
            result = await super().dispatch(operation, payload)
            dispatches.append(
                {
                    "phase": "claude-offline",
                    "operation": operation,
                    "payload": deepcopy(payload),
                    "result": deepcopy(result),
                }
            )
            return result

    def factory() -> LocalMemoryService:
        config = MnemosConfig(
            modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
        )
        config.surprisal.embedding_dim = 768
        return TracedService(
            MnemosEngine(
                config=config,
                llm=UnavailableLLMProvider(),
                embedder=OllamaEmbeddingProvider(
                    base_url="http://127.0.0.1:11434",
                    timeout=30,
                    response_observer=embedding_responses.append,
                ),
                store=SQLiteStore(str(database)),
                scope_registry=registry,
                require_scope_context=True,
                owns_providers=True,
            ),
            registry,
        )

    def start() -> MnemosService:
        owner = MnemosService(
            dispatcher_factory=factory,
            descriptor_path=descriptor,
            service_log_path=root / "service.log",
        )
        owner.start()
        return owner

    async def health() -> None:
        client = MnemosClient(descriptor_path=descriptor, auto_start=False)
        try:
            await client.request("health", {})
        finally:
            await client.close()

    service = None
    try:
        service = start()
        asyncio.run(health())
        port = service.descriptor["port"] if service.descriptor else None
        assert isinstance(port, int)
        service.stop()
        service = None
        assert not descriptor.exists()
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(("127.0.0.1", port)) != 0
        report["outage_verified"] = {"descriptor_absent": True, "former_port_closed": True}
        config = PhaseEnvironment(
            root, project, root / "scopes.json", descriptor, settings, profile, root / "plugin"
        )
        process = run_claude_phase(
            config,
            phase="claude-offline",
            prompt=PROMPT,
            executable=executable,
            auth_source=Path.home() / ".claude/.credentials.json",
            allowed_tools=[CLAUDE_PREFIX + "mnemos_retrieve"],
        )
        report["native_process"] = process
        assert process["status"] == "completed" and process["auth_copy_removed"]
        assert not descriptor.exists()
        paths = list((root / "claude-offline/claude-home/projects").rglob("*.jsonl"))
        assert len(paths) == 1
        native = [json.loads(line) for line in paths[0].read_bytes().splitlines()]
        observed_models = sorted(
            {
                row["message"]["model"]
                for row in native
                if row.get("type") == "assistant"
                and isinstance(row.get("message"), dict)
                and isinstance(row["message"].get("model"), str)
            }
        )
        report["models_observed"] = observed_models
        assert observed_models == ["claude-sonnet-4-6"]
        assert not any(
            part.get("type") == "tool_use"
            for row in native
            for part in (row.get("message", {}).get("content") or [])
            if isinstance(part, dict)
        )
        expected = expected_messages("claude-code", "claude-offline", native)
        pending = json.loads((root / "outbox.json").read_bytes())
        assert pending
        save(root / "outbox-before-recovery.json", pending)
        report["native_transcript"] = {
            "path": str(paths[0]),
            "sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            "expected_text_messages": len(expected),
        }
        service = start()
        asyncio.run(health())
        assert not (root / "outbox.json").exists()
        before_replay = rows(database)
        save(root / "state-after-recovery.json", before_replay)
        require_pending_sources(before_replay, pending)
        coverage = reconcile_captures(
            expected,
            dispatches,
            scope={
                "user_profile_id": profile,
                "project_id": project_entry.id,
                "workspace_id": None,
            },
            jobs=before_replay["jobs"],
        )
        save(root / "capture-coverage.json", coverage)
        assert coverage["status"] == "bounded_text_receipts_matched"

        async def replay() -> None:
            client = MnemosClient(descriptor_path=descriptor, auto_start=False)
            try:
                for entry in pending:
                    assert entry["operation"] == "capture"
                    result = await client.capture(entry["payload"])
                    assert result["status"] == "committed"
            finally:
                await client.close()

        asyncio.run(replay())
        assert stable(rows(database)) == stable(before_replay)
        service.stop()
        service = start()
        asyncio.run(health())
        final = rows(database)
        save(root / "final-state.json", final)
        assert stable(final) == stable(before_replay)
        assert final["fact_count"] == 0 and final["integrity"] == "ok"
        assert all(job["status"] == "pending" for job in final["jobs"])
        assert not (root / "outbox.json").exists()
        assert not embedding_responses
        report.update(
            status="native_service_outage_raw_capture_observed",
            queued_entries=len(pending),
            covered_messages=coverage["covered_messages"],
            repeat_replay_and_restart_preserved=True,
        )
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        cleanup_errors = []
        if service is not None:
            try:
                service.stop()
            except Exception as exc:
                cleanup_errors.append(type(exc).__name__)
        report["cleanup_errors"] = cleanup_errors
        report["embedding_responses_observed"] = len(embedding_responses)
        report["cleanup_established"] = (
            not cleanup_errors
            and not descriptor.exists()
            and not any(root.rglob(".credentials.json"))
        )
        if not report["cleanup_established"]:
            report["status"] = "failed"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(root / "dispatch-records.json", dispatches)
        save(root / "report.json", report)
    return 0 if report["status"] == "native_service_outage_raw_capture_observed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
