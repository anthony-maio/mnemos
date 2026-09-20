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
import subprocess
import sys
from typing import Any
import zipfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "native_hosts"

from .claude_service_outage_v1 import rows, stable
from .outage_acceptance import (
    require_health_gate,
    require_native_transcript,
    require_pending_sources_v2,
)
from .capture_coverage import expected_messages, reconcile_captures
from .native_processes import (
    CLAUDE_PREFIX,
    PhaseEnvironment,
    run_claude_phase,
    _selectors,
    clean_environment,
)

PRODUCT_COMMIT = "a64afb73d1b99510cfa32e96164bf6c11458b51e"
CHECKS = Path(
    "C:/Users/Ant/AppData/Local/Temp/mnemos-package-checks-autostart-env-v1/verification.json"
)
INSTALLED = Path(
    "C:/Users/Ant/AppData/Local/Temp/mnemos-installed-checks-autostart-env-v1/installation.json"
)
PROTOCOL = "docs/plans/2026-09-20-native-claude-service-outage-v2.md"
PROMPT = (
    "For this disposable project, the outage test marker is birch-window-618. "
    "Do not call any tools or claim that memory storage succeeded. "
    "Reply with exactly: Acknowledged birch-window-618."
)


def save(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode())


def main() -> int:
    if len(sys.argv) not in {2, 3} or (len(sys.argv) == 3 and sys.argv[2] != "--preflight"):
        raise ValueError("Pass one fresh absolute disposable output directory")
    supplied_root = Path(sys.argv[1])
    expected_root = Path("C:/Users/Ant/AppData/Local/Temp/mnemos-native-claude-outage-v2")
    if not supplied_root.is_absolute() or supplied_root.resolve() != expected_root.resolve():
        raise ValueError("Use the protocol's exact disposable output path")
    root = supplied_root.resolve()
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
    verification = json.loads(CHECKS.read_bytes())
    installed = json.loads(INSTALLED.read_bytes())
    required = {"pytest", "black", "mypy", "javascript", "build", "twine"}
    if (
        verification.get("status") != "passed"
        or verification.get("source_commit") != PRODUCT_COMMIT
        or {
            row["name"]
            for row in verification["checks"]
            if row.get("status") == "passed" and row.get("exit_code") == 0
        }
        != required
        or installed.get("status") != "passed"
        or installed.get("commit") != PRODUCT_COMMIT
        or Path(installed["checks"]).resolve() != CHECKS.resolve()
        or Path(sys.executable).resolve()
        != (Path(installed["environment"]) / "Scripts/python.exe").resolve()
    ):
        raise ValueError("Source and installed package checks must pass on the pinned product")
    artifacts = [item for item in verification["artifacts"] if item["file"].endswith(".whl")]
    if len(artifacts) != 1:
        raise ValueError("One verified wheel is required")
    wheel = Path(artifacts[0]["file"])
    wheel_sha = artifacts[0]["sha256"]
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != wheel_sha:
        raise ValueError("verified product wheel changed")
    package_checks = {
        "source_commit": PRODUCT_COMMIT,
        "verification_sha256": hashlib.sha256(CHECKS.read_bytes()).hexdigest(),
        "installation_sha256": hashlib.sha256(INSTALLED.read_bytes()).hexdigest(),
    }
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
        "benchmarks/native_hosts/claude_service_outage_v2.py",
        "benchmarks/native_hosts/native_processes.py",
        "benchmarks/native_hosts/capture_coverage.py",
        "benchmarks/native_hosts/outage_acceptance.py",
        "benchmarks/native_hosts/claude_service_outage_v1.py",
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
                    "package_checks": package_checks,
                    "wheel_sha256": wheel_sha,
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
    profile = "native-claude-outage-v2"
    registry = ScopeRegistry()
    project_entry = registry.register_project(project, user_profile_id=profile, enabled=True)
    registry.save(root / "scopes.json")
    settings = root / "settings.toml"
    settings.write_text(
        "[modules]\nsurprisal=false\nmutable_rag=false\naffective=false\nsleep=false\nspreading=false\n",
        encoding="utf-8",
    )
    with settings.open("a", encoding="utf-8") as handle:
        handle.write(
            '\n[storage]\ntype="sqlite"\nsqlite_path='
            + json.dumps(str(root / "memory.sqlite"))
            + "\n"
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
        "wheel_sha256": wheel_sha,
        "package_checks": package_checks,
        "installed_package": str(package),
        "claude_version": version,
        "claude_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "model_requested": "claude-sonnet-4-6",
        "background_processing_enabled": False,
        "local_provider_calls_expected": 0,
    }
    config = PhaseEnvironment(
        root, project, root / "scopes.json", descriptor, settings, profile, root / "plugin"
    )
    config.validate()
    selectors = _selectors(config, "claude-code", "claude-offline")
    from mnemos.settings import load_settings

    resolved = load_settings(env=clean_environment(selectors), cwd=project)
    if (
        resolved.settings.storage.type != "sqlite"
        or Path(resolved.settings.storage.sqlite_path).resolve() != database
    ):
        raise ValueError("Disposable storage selector did not resolve correctly")
    for key in list(os.environ):
        if key.upper().startswith("MNEMOS_"):
            del os.environ[key]
    os.environ.update(selectors)
    if MnemosClient().auto_start is not False:
        raise ValueError("Installed client ignored autostart opt-out")
    report["isolation"] = {
        "selectors": selectors,
        "resolved_database": str(database),
        "autostart_disabled": True,
    }
    observed_modules: list[dict[str, bool]] = []
    health_responses: list[dict[str, Any]] = []
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
        observed_modules.append(config.modules.model_dump())
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
            response = await client.request("health", {})
            health_responses.append(response)
            require_health_gate(
                response, expected_database=database, module_config=observed_modules[-1]
            )
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
        if any(
            row.get("sessionId") != process["session_id"]
            for row in native
            if row.get("type") in {"user", "assistant"}
        ):
            raise ValueError("Native transcript session differs from the launched conversation")
        require_native_transcript(
            native,
            submitted_marker=PROMPT,
            expected_ack="Acknowledged birch-window-618.",
            expected_model="claude-sonnet-4-6",
        )
        report["models_observed"] = ["claude-sonnet-4-6"]
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
        require_pending_sources_v2(before_replay, pending)
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
        require_pending_sources_v2(final, pending)
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
        save(root / "health-responses.json", health_responses)
        save(root / "injected-module-configs.json", observed_modules)
        save(root / "report.json", report)
    return 0 if report["status"] == "native_service_outage_raw_capture_observed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
