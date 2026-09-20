"""CLI retrieval view forwarding against an isolated authenticated service."""

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from mnemos.contracts import CaptureEvent
from mnemos.service import MnemosService
from tests.test_memory_service import service


def test_cli_retrieve_forwards_memory_and_captured_evidence_views(tmp_path: Path) -> None:
    memory, context = service(tmp_path)
    for name in type(memory.engine.config.modules).model_fields:
        setattr(memory.engine.config.modules, name, False)
    asyncio.run(
        memory.dispatch(
            "remember",
            {
                "context": context.model_dump(),
                "content": "The CLI note is maple-cli-552.",
            },
        )
    )
    registry_path = tmp_path / "registry.json"
    memory.registry.save(registry_path)
    runtime = MnemosService(
        dispatcher=memory,
        descriptor_path=tmp_path / "runtime.json",
        lock_path=tmp_path / "runtime.lock",
    )
    runtime.start()
    captured = memory.capture(
        CaptureEvent(
            event_id="cli-captured-input",
            context=context,
            occurred_at=datetime.now(timezone.utc),
            role="assistant",
            content="Earlier captured note says cedar-cli-441.",
            source_kind="conversation",
        )
    )
    asyncio.run(memory.maintenance.run_once(force=True))
    env = {
        **os.environ,
        "MNEMOS_RUNTIME_DESCRIPTOR": str(runtime.descriptor_path),
        "MNEMOS_RUNTIME_DIR": str(tmp_path),
        "MNEMOS_SCOPE_REGISTRY": str(registry_path),
        "MNEMOS_USER_PROFILE_ID": context.user_profile_id,
        "MNEMOS_PROJECT_ROOT": str(tmp_path),
        "MNEMOS_SESSION_ID": "cli-retrieval-views",
    }

    def cli(view: str) -> dict:
        result = subprocess.run(
            [sys.executable, "-m", "mnemos.cli", "retrieve", "cli note", "--view", view],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    try:
        ordinary = cli("memory")
        evidence = cli("captured_evidence")
        assert ordinary["view"] == "memory"
        assert ordinary["facts"]
        assert ordinary["captured_evidence"] == []
        assert evidence["view"] == "captured_evidence"
        assert evidence["facts"] == []
        assert evidence["captured_evidence"]
        assert "cedar-cli-441" in evidence["evidence_context"]
        assert memory.engine.store.get_episodic_event(captured["event_id"]) is not None
    finally:
        runtime.stop()
        memory.engine.store.close()


def test_cli_retrieve_rejects_unknown_view_without_starting_service(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mnemos.cli", "retrieve", "anything", "--view", "invalid"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "MNEMOS_RUNTIME_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
