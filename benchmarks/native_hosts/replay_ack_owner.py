"""Disposable comparison owner. Only invoked with a frozen explicit run root."""

import asyncio
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time


def read_control(path):
    """Tolerate transient Windows sharing violations, never malformed input."""
    deadline = time.monotonic() + 2
    while True:
        try:
            return json.loads(path.read_bytes())
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def close_resources(worker, memory, stores):
    """Attempt every owned cleanup even if an earlier resource fails."""
    errors = []
    for name, close in [("worker", worker.stop), ("memory", memory.stop)]:
        try:
            close()
        except BaseException as exc:
            errors.append({"resource": name, "error_type": type(exc).__name__})
    owner = getattr(memory, "_owner_thread", None)
    if owner is not None and owner.is_alive():
        errors.append({"resource": "stores", "error_type": "OwnerStillAlive"})
        return errors
    for i, store in enumerate(stores):
        try:
            store.close()
        except BaseException as exc:
            errors.append({"resource": f"store-{i}", "error_type": type(exc).__name__})
    return errors


def validate_paths(root, arm, descriptor, registry):
    """Reject fallback to any normal runtime before constructing a service."""
    folder = root / arm
    expected = {
        "MNEMOS_CONFIG_PATH": root / "project/settings.toml",
        "MNEMOS_SCOPE_REGISTRY": root / "scopes.json",
        "MNEMOS_RUNTIME_DIR": folder,
        "MNEMOS_RUNTIME_DESCRIPTOR": folder / "memory.json",
        "MNEMOS_HOOK_WORKER_DESCRIPTOR": folder / "worker.json",
    }
    for key, path in expected.items():
        if not os.environ.get(key) or Path(os.environ[key]).resolve() != path.resolve():
            raise ValueError(f"Disposable owner requires explicit {key}")
    if descriptor.resolve() != (folder / "memory.json").resolve():
        raise ValueError("Memory descriptor is outside the selected arm")
    if registry.resolve() != (root / "scopes.json").resolve():
        raise ValueError("Registry is outside the comparison")
    for suffix in ("json", "sqlite"):
        if not descriptor.with_name(f"outbox.{suffix}").resolve().is_relative_to(folder.resolve()):
            raise ValueError("Outbox is outside the selected arm")
    if os.environ.get("MNEMOS_AUTO_START") != "false":
        raise ValueError("Disposable owner requires autostart disabled")


def main():
    root = Path(os.environ["MNEMOS_COMPARISON_ROOT"]).resolve(strict=True)
    arm = os.environ["MNEMOS_COMPARISON_ARM"]
    assert arm in {"baseline", "candidate"}
    freeze = json.loads((root / "freeze.json").read_text())
    target = Path(freeze["arms"][arm]["installed_target"])
    sys.path.insert(0, str(target))
    import mnemos
    from mnemos.client import MnemosClient
    from mnemos.engine import MnemosEngine
    from mnemos.hook_worker import HookWorker
    from mnemos.memory_service import LocalMemoryService
    from mnemos.scopes import ScopeRegistry, default_registry_path
    from mnemos.service import MnemosService, default_descriptor_path
    from mnemos.sqlite_outbox import SQLiteOutbox
    from mnemos.utils.embeddings import SimpleEmbeddingProvider
    from mnemos.utils.llm import UnavailableLLMProvider
    from mnemos.utils.storage import SQLiteStore

    assert Path(mnemos.__file__).resolve().is_relative_to(target.resolve())
    validate_paths(root, arm, default_descriptor_path(), default_registry_path())
    folder = root / arm
    lock = threading.Lock()
    activity = {"ack": 0, "replay": 0}
    intervals = []
    responses = []
    original_ack = SQLiteOutbox.acknowledge_many
    original_replay = MnemosClient.replay_outbox
    original_submit = MnemosService.submit

    def observed_ack(self, event_ids):
        row = {"start_ns": time.perf_counter_ns(), "event_ids": list(event_ids)}
        with lock:
            activity["ack"] += 1
        try:
            row["acknowledged"] = original_ack(self, event_ids)
            return row["acknowledged"]
        except BaseException as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            row["end_ns"] = time.perf_counter_ns()
            with lock:
                intervals.append(row)
                activity["ack"] -= 1

    async def observed_replay(self):
        with lock:
            activity["replay"] += 1
        try:
            return await original_replay(self)
        finally:
            with lock:
                activity["replay"] -= 1

    def observed_submit(self, operation, payload):
        result = original_submit(self, operation, payload)
        if operation == "hook":
            with lock:
                responses.append(result)
        return result

    SQLiteOutbox.acknowledge_many = observed_ack
    MnemosClient.replay_outbox = observed_replay
    MnemosService.submit = observed_submit

    def save(name, data):
        path = folder / name
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)

    stores = []
    registry_path = root / "scopes.json"

    def load_registry():
        return ScopeRegistry.load(registry_path)

    def factory():
        store = SQLiteStore(str(folder / "memory.sqlite"))
        stores.append(store)
        engine = MnemosEngine(
            llm=UnavailableLLMProvider(), embedder=SimpleEmbeddingProvider(), store=store
        )
        for flag in type(engine.config.modules).model_fields:
            setattr(engine.config.modules, flag, False)
        return LocalMemoryService(engine, load_registry(), registry_loader=load_registry)

    memory = MnemosService(dispatcher_factory=factory, descriptor_path=folder / "memory.json")
    worker = HookWorker(
        "codex",
        descriptor_path=folder / "worker.json",
        cwd=root / "project",
        queue_captures=True,
        outbox_backend="sqlite",
    )

    async def loop_state():
        dispatcher = worker._dispatcher
        pumps = [p for p in (dispatcher.replay, dispatcher.compatibility_replay) if p is not None]
        with lock:
            state = dict(activity)
        state["pumps_idle"] = all(
            p.last_result is not None
            and p.last_error_type is None
            and p.last_result["pending"] == 0
            and not p._wake.is_set()
            for p in pumps
        )
        return state

    def drain(expected):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            memory.submit("maintenance", {})
            state = asyncio.run_coroutine_threadsafe(loop_state(), worker._loop).result(timeout=5)
            with closing(
                sqlite3.connect((folder / "memory.sqlite").as_uri() + "?mode=ro", uri=True)
            ) as db:
                count = db.execute("SELECT count(*) FROM episodic_events").fetchone()[0]
                jobs = db.execute("SELECT status FROM memory_jobs").fetchall()
            queue = worker._dispatcher._owned_outbox
            pending = queue.pending()
            if (
                count == expected
                and len(jobs) == expected
                and all(j[0] == "completed" for j in jobs)
                and not pending
                and state == {"ack": 0, "replay": 0, "pumps_idle": True}
            ):
                return {
                    "events": count,
                    "jobs": len(jobs),
                    "pending": 0,
                    "observed_ns": time.perf_counter_ns(),
                    **state,
                }
            time.sleep(0.01)
        raise TimeoutError("Drain barrier did not complete")

    try:
        md, wd = memory.start(), worker.start()
        drain(0)
        queue = worker._dispatcher._owned_outbox
        with queue._lock:
            connection = queue._connect()
            assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        save(
            "ready.json",
            {
                "pid": os.getpid(),
                "memory_port": md["port"],
                "worker_port": wd["port"],
                "environment_sha256": wd["environment_sha256"],
                "full_sync": True,
                "integrity_check": "ok",
            },
        )
        sequence = 0
        while True:
            command = folder / f"control-{sequence}.json"
            if not command.exists():
                time.sleep(0.01)
                continue
            request = read_control(command)
            sequence += 1
            if request["operation"] == "stop":
                break
            assert request["operation"] == "drain"
            state = drain(request["expected"])
            with lock:
                state.update(responses=list(responses), ack_intervals=list(intervals))
            save(f"barrier-{request['round']}.json", state)
    finally:
        cleanup_errors = close_resources(worker, memory, stores)
        with lock:
            save(
                "final-trace.json",
                {"responses": responses, "ack_intervals": intervals, "activity": activity},
            )
        save(
            "stopped.json",
            {"pid": os.getpid(), "stopped": not cleanup_errors, "cleanup_errors": cleanup_errors},
        )
        if cleanup_errors:
            raise RuntimeError("Owner cleanup failed; see stopped.json")


if __name__ == "__main__":
    main()
