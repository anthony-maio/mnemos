"""Installed pilot recovery from one explicitly injected embedding exception."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WHEEL_SHA = "ecd11eb7ecbb9cc4f14552d9a7253dead7a8348879c35063411bb850abe06036"
LAUNCHER_SHA = "ab2818b0935320e70dbfd35401735ed11b49eb82d4d36668c6e54f4a547ee459"
PROFILE = "private-pilot-20261106"
INPUTS = (
    "scripts/check-pilot-embedding-recovery.py",
    "docs/plans/2026-11-07-pilot-embedding-recovery-v1.md",
)
TABLES = {
    "events": "episodic_events",
    "jobs": "memory_jobs",
    "chunks": "memory_chunks",
    "evidence": "fact_evidence",
    "revisions": "fact_revisions",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def snapshot(database: Path) -> dict[str, Any]:
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        return {
            name: sorted(
                (dict(row) for row in connection.execute("SELECT * FROM " + table)),
                key=lambda row: json.dumps(row, sort_keys=True),
            )
            for name, table in TABLES.items()
        }
    finally:
        connection.close()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import mnemos
    from mnemos.client import MnemosClient
    from mnemos.scopes import ScopeRegistry
    from mnemos.service import MnemosService, _protect_private_path
    from mnemos.utils.embeddings import FastembedProvider
    from mnemos.utils.reliability import MnemosTransientError

    pilot, root = args.pilot_root.resolve(), args.output.resolve()
    assert not root.exists() and not root.is_relative_to(ROOT) and not root.is_relative_to(pilot)
    assert Path(sys.executable).resolve() == pilot / "venv/Scripts/python.exe"
    assert Path(mnemos.__file__).resolve().is_relative_to(pilot / "venv")
    installation = json.loads((pilot / "installation-state.json").read_bytes())
    assert installation["status"] == "PREPARED_INACTIVE"
    wheel = pilot / "artifacts/mnemos_memory-0.7.0b1-py3-none-any.whl"
    assert installation["wheel_sha256"] == sha(wheel) == WHEEL_SHA
    assert sha(pilot / "launch.py") == LAUNCHER_SHA
    assert not (pilot / "runtime").exists() and not (pilot / "memory.sqlite").exists()
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("mnemos/") and not name.endswith("/"):
                assert (Path(mnemos.__file__).parent.parent / name).read_bytes() == archive.read(
                    name
                )
    watched = {
        pilot / name: sha(pilot / name)
        for name in ("launch.py", "settings.toml", "scopes.json", "installation-state.json")
    }
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name in INPUTS:
        committed = subprocess.check_output(["git", "show", f"{head}:{name}"], cwd=ROOT)
        assert committed.replace(b"\r\n", b"\n") == (ROOT / name).read_bytes().replace(
            b"\r\n", b"\n"
        )
    root.mkdir(parents=True)
    _protect_private_path(root)
    for key in tuple(os.environ):
        if key.upper().startswith("GIT_"):
            del os.environ[key]
    registry = ScopeRegistry()
    contexts = []
    for name in ("project", "foreign"):
        project = root / name
        project.mkdir()
        subprocess.run(
            ["git", "init", "--quiet"],
            cwd=project,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        (project / ".mnemos").mkdir()
        (project / ".mnemos/mnemos.toml").write_text("", encoding="utf-8")
        registry.register_project(project, user_profile_id=PROFILE, enabled=True)
        contexts.append(
            registry.resolve(
                project, user_profile_id=PROFILE, session_id=name, host="codex"
            ).model_dump(mode="json")
        )
    assert contexts[0]["project_id"] != contexts[1]["project_id"]
    registry.save(root / "scopes.json")
    (root / ".mnemos").mkdir()
    (root / ".mnemos/mnemos.toml").write_text("", encoding="utf-8")
    (root / "launch.py").write_bytes((pilot / "launch.py").read_bytes())
    (root / "settings.toml").write_text(
        (pilot / "settings.toml")
        .read_text(encoding="utf-8")
        .replace(pilot.as_posix(), root.as_posix()),
        encoding="utf-8",
    )
    os.chdir(root / "project")
    runpy.run_path(str(root / "launch.py"), run_name="pilot_configuration")["prepare"]("codex")
    descriptor = root / "runtime/runtime.json"
    database = root / "memory.sqlite"
    fault = threading.Event()
    calls: list[dict[str, Any]] = []
    calls_lock = threading.Lock()
    record: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "product_commit": "6d0e5ba69f39d55df8623830cb682e14a5b2d607",
        "runner_commit": head,
        "wheel_sha256": WHEEL_SHA,
        "launcher_sha256": LAUNCHER_SHA,
        "input_hashes": {name: sha(ROOT / name) for name in INPUTS},
        "configuration_hashes": {
            name: sha(root / name) for name in ("launch.py", "settings.toml", "scopes.json")
        },
        "watched_file_hashes_before": {str(path): digest for path, digest in watched.items()},
        "fault": "MnemosTransientError injected at the real FastembedProvider._vectors boundary",
        "native_host_run": False,
        "natural_provider_failure": False,
        "background_scheduler_disabled": False,
        "synthetic_vectors": False,
        "clock_or_job_rows_changed": False,
        "real_pilot_activated": False,
        "pilot_days": 0,
    }

    class FaultService(MnemosService):
        def _make_default_dispatcher(self) -> Any:
            dispatcher = super()._make_default_dispatcher()
            provider = dispatcher.engine.embedder
            assert type(provider) is FastembedProvider
            assert not any(dispatcher.engine.config.modules.model_dump().values())
            original = provider._vectors

            def vectors(texts: list[str]) -> list[list[float]]:
                failed = fault.is_set()
                item: dict[str, Any] = {"failed": failed, "texts": texts}
                if not failed:
                    result = original(texts)
                    assert all(len(vector) == 384 for vector in result)
                    item["vectors"] = result
                with calls_lock:
                    calls.append(item)
                    save(root / "provider-calls.json", calls)
                if failed:
                    raise MnemosTransientError("controlled-pilot-embedding-fault")
                return result

            setattr(provider, "_vectors", vectors)
            return dispatcher

    runtime = FaultService(descriptor_path=descriptor)
    client = MnemosClient(descriptor_path=descriptor, auto_start=False, request_timeout=60)
    external_processes: list[subprocess.Popen[bytes]] = []

    async def request(name: str, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await client.request(operation, payload)
        finally:
            if client._child is not None and client._child not in external_processes:
                external_processes.append(client._child)
        save(
            root / (name + ".json"),
            {"operation": operation, "payload": payload, "response": response},
        )
        with (root / "request-history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"name": name, "operation": operation, "payload": payload, "response": response}
                )
                + "\n"
            )
        return response

    async def stop_external() -> None:
        if descriptor.exists():
            await client.request("shutdown", {})
        for child in external_processes:
            if child.poll() is None:
                await asyncio.to_thread(child.wait, 30)
        assert not descriptor.exists()

    def source(context: dict[str, Any], event_id: str, content: str) -> dict[str, Any]:
        return {
            "context": context,
            "event_id": event_id,
            "content": content,
            "role": "user",
            "source_kind": "pilot_recovery_fixture",
            "source_ref": event_id,
            "occurred_at": "2026-11-07T00:00:00Z",
        }

    save(root / "report.json", record)
    try:
        runtime.start()
        healthy = await request("initial-health", "health", {})
        assert healthy["capabilities"]["background_maintenance"]
        seeds = []
        for index, marker in enumerate(("willow-orbit-319", "copper-foreign-627")):
            seeds.append(
                await request(
                    f"seed-{index}",
                    "remember",
                    source(
                        contexts[index], f"seed-{index}", "The release marker is " + marker + "."
                    ),
                )
            )
        await request("seed-maintenance", "maintenance", {})
        baseline = snapshot(database)
        assert len(baseline["chunks"]) == 2
        assert all(row["status"] == "completed" for row in baseline["jobs"])
        save(root / "baseline.json", baseline)
        fault.set()
        operations = [
            (
                "remember",
                source(contexts[0], "fault-explicit", "The backup marker is fir-window-928."),
            ),
            (
                "capture",
                source(contexts[0], "fault-capture", "A temporary note mentions ash-table-516."),
            ),
        ]
        writes = [
            await request(f"outage-write-{i}", operation, payload)
            for i, (operation, payload) in enumerate(operations)
        ]
        assert all(row["status"] == "committed" for row in writes)
        failed = await request("outage-maintenance", "maintenance", {})
        assert failed["completed"] == 0
        state = snapshot(database)
        pending = [
            row for row in state["jobs"] if row["id"] in {write["job_id"] for write in writes}
        ]
        assert len(pending) == 2 and len(state["events"]) == 4 and len(state["chunks"]) == 2
        assert all(
            row["status"] == "pending"
            and row["attempt_count"] == 0
            and row["provider_failures"] == 1
            and row["last_error"] == "MnemosTransientError"
            for row in pending
        )
        assert all(
            row["lease_owner"] is None and row["lease_token"] is None and row["lease_until"] is None
            for row in pending
        )
        assert all(
            datetime.fromisoformat(row["available_at"]) > datetime.fromisoformat(row["updated_at"])
            for row in pending
        )
        for i, write in enumerate(writes):
            job = await request(
                f"outage-job-{i}", "job", {"context": contexts[0], "job_id": write["job_id"]}
            )
            assert (
                job["job"]["status"] == "pending"
                and job["job"]["last_error"] == "MnemosTransientError"
            )
        for name, query in (
            ("retained", "willow-orbit-319"),
            ("foreign", "copper-foreign-627"),
            ("unprocessed", "fir-window-928"),
        ):
            found = await request(
                "outage-recall-" + name,
                "retrieve",
                {"context": contexts[0], "query": query, "allowed_scopes": ["project"]},
            )
            assert found["degraded"] and found["retrieval"]["mode"] == "lexical_only"
            assert found["retrieval"]["reason"] == "embedding_unavailable"
            assert all(row["scores"]["semantic"] is None for row in found["facts"])
            if name == "retained":
                assert (
                    len(found["facts"]) == 1 and "willow-orbit-319" in found["facts"][0]["content"]
                )
            else:
                assert not found["facts"]
        save(root / "outage-state.json", snapshot(database))
        await client.close()
        runtime.stop()
        assert not descriptor.exists()
        fault.clear()
        client = MnemosClient(
            descriptor_path=descriptor, auto_start=True, startup_timeout=60, request_timeout=60
        )
        healthy = await request("recovery-health", "health", {})
        assert healthy["runtime"]["embedding_provider"] == "FastembedProvider"
        assert healthy["runtime"]["cognition_provider"] == "UnavailableLLMProvider"
        assert healthy["capabilities"]["background_maintenance"]
        deadline = time.monotonic() + 60
        while True:
            await request("recovery-maintenance", "maintenance", {})
            state = snapshot(database)
            jobs = {row["id"]: row for row in state["jobs"]}
            if all(jobs[write["job_id"]]["status"] == "completed" for write in writes):
                break
            assert time.monotonic() < deadline
            await asyncio.sleep(0.5)
        assert len(state["events"]) == len(state["jobs"]) == len(state["chunks"]) == 4
        for write in writes:
            job = jobs[write["job_id"]]
            assert (
                job["attempt_count"] == 1
                and job["provider_failures"] == 1
                and job["last_error"] == "MnemosTransientError"
            )
        target = next(
            row for row in state["chunks"] if row["content"] == operations[0][1]["content"]
        )
        raw = next(row for row in state["chunks"] if row["content"] == operations[1][1]["content"])
        assert json.loads(target["metadata"])["assertion_status"] == "asserted"
        assert json.loads(raw["metadata"])["source"] == "captured_passthrough"
        assert json.loads(raw["metadata"])["assertion_status"] == "unknown"
        for original in baseline["chunks"]:
            current = next(row for row in state["chunks"] if row["id"] == original["id"])
            for key in ("content", "embedding", "metadata", "version"):
                assert current[key] == original[key]
        found = await request(
            "recovered-semantic-recall",
            "retrieve",
            {"context": contexts[0], "query": "fir-window-928", "allowed_scopes": ["project"]},
        )
        assert not found.get("degraded", False)
        assert any(
            row["id"] == target["id"] and row["scores"]["semantic"] is not None
            for row in found["facts"]
        )
        assert all(row["id"] != raw["id"] for row in found["facts"])
        finished = snapshot(database)
        save(root / "completed-state.json", finished)
        for i, (operation, payload) in enumerate(operations):
            replay = await request(f"exact-replay-{i}", operation, payload)
            assert (
                replay["event_id"] == writes[i]["event_id"]
                and replay["job_id"] == writes[i]["job_id"]
            )
        assert snapshot(database) == finished
        await stop_external()
        await client.close()
        client = MnemosClient(
            descriptor_path=descriptor, auto_start=True, startup_timeout=60, request_timeout=60
        )
        await request("second-restart-health", "health", {})
        restarted = snapshot(database)
        save(root / "second-restart-state.json", restarted)
        assert restarted == finished
        record.update(
            status="passed",
            pending_jobs_during_fault=2,
            completed_jobs_after_recovery=2,
            recovered_explicit_facts=1,
            recovered_unasserted_chunks=1,
            exact_replays=2,
            second_restart_rows_identical=True,
            retained_control_and_foreign_fact_preserved=True,
        )
    except BaseException as error:
        record.update(status="failed", error=type(error).__name__)
        raise
    finally:
        fault.clear()
        cleanup_errors = []
        try:
            runtime.stop()
            await stop_external()
        except Exception as error:
            cleanup_errors.append(type(error).__name__)
        await client.close()
        record["external_processes"] = [
            {"pid": child.pid, "returncode": child.poll()} for child in external_processes
        ]
        if any(item["returncode"] != 0 for item in record["external_processes"]):
            cleanup_errors.append("ExternalServiceDidNotExitCleanly")
        record["watched_file_hashes_after"] = {str(path): sha(path) for path in watched}
        if record["watched_file_hashes_after"] != record["watched_file_hashes_before"]:
            cleanup_errors.append("PilotFilesChanged")
        if (
            descriptor.exists()
            or (pilot / "runtime").exists()
            or (pilot / "memory.sqlite").exists()
        ):
            cleanup_errors.append("RuntimeRemains")
        record["cleanup_errors"] = cleanup_errors
        record["cleanup"] = "failed" if cleanup_errors else "passed"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(root / "report.json", record)
        if cleanup_errors:
            raise RuntimeError("Disposable recovery cleanup failed")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(asyncio.run(run(parser.parse_args())), indent=2))
