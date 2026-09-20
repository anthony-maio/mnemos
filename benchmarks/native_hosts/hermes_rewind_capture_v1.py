"""Exercise actual Hermes undo/retry through MemoryManager and the Mnemos outbox.

Run in the isolated WSL host test interpreter. This is an API fixture, not a
native model conversation, a transport test, or an identity migration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import zipfile

HOST_COMMIT = "14a346345486418aa85ed6526549412ee2665ade"
WHEEL_SHA = "9af6667d86933570beecc940e55e637f70a9258e5339e948dc48ba9692782bf6"
HOST_FILES = (
    "hermes_state.py",
    "agent/memory_manager.py",
    "agent/conversation_compression.py",
    "hermes_cli/cli_agent_setup_mixin.py",
    "hermes_cli/cli_session_mixin.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    host, wheel, root = args.host.resolve(), args.wheel.resolve(), args.output.resolve()
    assert sha(wheel) == WHEEL_SHA, "wheel binding mismatch"
    assert (
        subprocess.check_output(["git", "-C", str(host), "rev-parse", "HEAD"], text=True).strip()
        == HOST_COMMIT
    )
    root.mkdir(parents=True, exist_ok=False)
    original_env = os.environ.copy()
    os.environ.clear()
    os.environ.update(
        {k: v for k, v in original_env.items() if k in {"PATH", "LANG", "WSL_INTEROP"}}
    )
    os.environ.update(
        HOME=str(root / "home"),
        HERMES_HOME=str(root / "home"),
        MNEMOS_AUTO_START="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    (root / "home").mkdir()
    os.chdir(root)
    sys.dont_write_bytecode = True
    extracted = root / "wheel"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(extracted)
    sys.path[:0] = [str(extracted), str(host)]
    before = {name: sha(host / name) for name in HOST_FILES}
    network_attempts: list[str] = []

    def deny_network(*unused: Any, **kwargs: Any) -> Any:
        network_attempts.append("blocked")
        raise RuntimeError("network prohibited in API fixture")

    socket.socket.connect = deny_network  # type: ignore[method-assign]
    socket.socket.connect_ex = deny_network  # type: ignore[method-assign]
    socket.create_connection = deny_network

    from agent.memory_manager import MemoryManager
    from hermes_cli.cli_session_mixin import CLISessionMixin
    from hermes_state import SessionDB
    from mnemos.adapters.hermes import MnemosHermesProvider
    from mnemos.client import Outbox
    from mnemos.scopes import ScopeRegistry

    names = ["agent.memory_manager", "hermes_cli.cli_session_mixin", "hermes_state"]
    origins = {name: importlib.import_module(name).__file__ for name in names}
    assert all(value and Path(value).resolve().is_relative_to(host) for value in origins.values())
    adapter_file = importlib.import_module("mnemos.adapters.hermes").__file__
    assert adapter_file
    adapter_origin = Path(adapter_file).resolve()
    assert adapter_origin.is_relative_to(extracted)
    report: dict[str, Any] = {
        "status": "running",
        "scope": "actual CLI methods and MemoryManager; local queued outbox only",
        "host_commit": HOST_COMMIT,
        "host": str(host),
        "host_files_before": before,
        "host_origins": origins,
        "adapter_origin": str(adapter_origin),
        "wheel_sha256": WHEEL_SHA,
        "runner_sha256": sha(Path(__file__)),
        "cases": [],
    }

    class QueueClient:
        def __init__(self, path: Path) -> None:
            self.outbox = Outbox(path)

        async def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.outbox.append(payload, event_id=payload["event_id"])
            return {"status": "queued", "committed": False}

    for operation in ("undo", "retry"):
        case_root = root / operation
        case_root.mkdir()
        case: dict[str, Any] = {"operation": operation, "status": "running", "phases": []}
        report["cases"].append(case)
        db = SessionDB(case_root / "state.db")
        sid = "rewind-capture-fixture"
        db.create_session(sid, "cli", cwd=str(case_root), model="fixture-no-model")
        registry = ScopeRegistry()
        registry.register_project(case_root, user_profile_id="rewind-fixture", enabled=True)
        client = QueueClient(case_root / "outbox.json")

        def start() -> tuple[Any, Any]:
            provider = MnemosHermesProvider(
                client=client, registry=registry, user_profile_id="rewind-fixture"
            )
            manager = MemoryManager()
            manager.add_provider(provider)
            manager.initialize_all(sid, cwd=str(case_root), hermes_home=str(case_root / "home"))
            return manager, provider

        manager, provider = start()

        def capture(label: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
            manager.sync_all("Use maple.", "Maple recorded.", session_id=sid, messages=messages)
            assert manager.flush_pending(timeout=10), "manager flush failed"
            assert provider._flush(timeout=10), "provider flush failed"
            pending = client.outbox.pending()
            snapshot = {
                "label": label,
                "input_row_ids": [m.get("_row_id") for m in messages],
                "pending": [row["payload"] for row in pending],
                "diagnostics": provider.get_lifecycle_diagnostics(),
            }
            case["phases"].append(snapshot)
            return {row["payload"]["event_id"]: row["payload"] for row in pending}

        try:
            messages = [
                {"role": "user", "content": "Use cedar."},
                {"role": "assistant", "content": "Cedar recorded."},
                {"role": "user", "content": "Use maple."},
                {"role": "assistant", "content": "Maple recorded."},
            ]
            assert db.append_messages_batch(sid, messages) == 4
            initial_rows = db.get_messages_as_conversation(sid, include_row_ids=True)
            initial = capture("initial_native_capture", initial_rows)
            assert len(initial) == 4
            assert capture("duplicate_replay", initial_rows) == initial

            cli = CLISessionMixin.__new__(CLISessionMixin)
            cli._session_db, cli.session_id = db, sid
            cli.conversation_history = [
                {k: v for k, v in row.items() if k != "_row_id"} for row in initial_rows
            ]
            cli.agent = SimpleNamespace(_memory_manager=manager, _session_messages=[])
            if operation == "undo":
                cli.undo_last(1, prefill=False)
            else:
                assert cli.retry_last() == "Use maple."
            retained = cli.conversation_history
            assert len(retained) == 2
            # Fixture models persistence of a new completed turn, using actual
            # SessionDB IDs. No inference or run_conversation call is simulated.
            assert db.append_messages_batch(sid, messages[2:]) == 2
            current_rows = db.get_messages_as_conversation(sid, include_row_ids=True)
            assert len(current_rows) == 4
            assert [m["_row_id"] for m in current_rows[2:]] != [
                m["_row_id"] for m in initial_rows[2:]
            ]
            after = capture("identical_regenerated_suffix", retained + current_rows[2:])
            assert len(after) == 6, f"expected six observations, got {len(after)}"
            assert all(after.get(key) == value for key, value in initial.items())
            assert all(
                p.get("identity_status") != "provisional_unassociated" for p in after.values()
            )
            assert [m["_row_id"] for m in retained] == [m["_row_id"] for m in initial_rows[:2]]

            manager.shutdown_all()
            manager, provider = start()
            assert capture("provider_restart_durable_reload", current_rows) == after
            assert capture("shortened_retained_view", current_rows[:2]) == after
            case["status"] = "passed"
        except Exception as exc:
            case.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            manager.shutdown_all()
            db.close()

    report["host_files_after"] = {name: sha(host / name) for name in HOST_FILES}
    report["network_attempts"] = len(network_attempts)
    report["host_unchanged"] = report["host_files_after"] == before
    passed = (
        all(case["status"] == "passed" for case in report["cases"])
        and report["host_unchanged"]
        and not network_attempts
    )
    report["status"] = "passed" if passed else "failed"
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(root / "report.json")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
