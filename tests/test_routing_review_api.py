from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from mnemos.control_plane import ControlPlaneService
from mnemos.routing_review import enqueue_routing_failure
from mnemos.ui_server import MnemosUiRouter


def test_cli_routing_review_needs_no_registry_or_service(tmp_path: Path) -> None:
    queue_dir = tmp_path / "routing"
    record = enqueue_routing_failure(
        host="codex",
        event="UserPromptSubmit",
        reason="registry_missing",
        session_id="private-session",
        project_hint=str(tmp_path / "project"),
        configured_profile="alice",
        directory=queue_dir,
    )
    env = {
        **os.environ,
        "MNEMOS_ROUTING_REVIEW_DIR": str(queue_dir),
        "MNEMOS_USER_PROFILE_ID": "alice",
        "MNEMOS_SCOPE_REGISTRY": str(tmp_path / "does-not-exist.json"),
        "MNEMOS_RUNTIME_DESCRIPTOR": str(tmp_path / "no-service.json"),
    }
    root = Path(__file__).resolve().parents[1]
    reviewed = subprocess.run(
        [sys.executable, "-m", "mnemos.cli", "routing-review"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert reviewed.returncode == 0, reviewed.stderr
    payload = json.loads(reviewed.stdout)
    assert payload["records"] == [record]
    assert payload["records"][0]["host"] == "codex"
    assert "private-session" not in reviewed.stdout

    acknowledged = subprocess.run(
        [sys.executable, "-m", "mnemos.cli", "routing-review-ack", record["record_id"]],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert acknowledged.returncode == 0, acknowledged.stderr
    assert json.loads(acknowledged.stdout)["acknowledged"] is True


def test_cli_routing_review_requires_explicit_unconfigured_mode(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "MNEMOS_ROUTING_REVIEW_DIR": str(tmp_path),
        "MNEMOS_USER_PROFILE_ID": "",
    }
    result = subprocess.run(
        [sys.executable, "-m", "mnemos.cli", "routing-review"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "--unconfigured" in result.stderr


def test_ui_routing_review_filters_profiles_and_ack_requires_csrf(tmp_path: Path) -> None:
    queue_dir = tmp_path / "routing"
    alice = enqueue_routing_failure(
        host="claude-code",
        event="SessionStart",
        reason="unknown_user_profile",
        configured_profile="alice",
        directory=queue_dir,
    )
    enqueue_routing_failure(
        host="hermes",
        event="ContextResolve",
        reason="missing_user_profile",
        configured_profile="bob",
        directory=queue_dir,
    )
    unconfigured = enqueue_routing_failure(
        host="codex",
        event="Stop",
        reason="registry_missing",
        configured_profile=None,
        directory=queue_dir,
    )
    service = ControlPlaneService(
        cwd=tmp_path,
        env={
            "MNEMOS_USER_PROFILE_ID": "alice",
            "MNEMOS_ROUTING_REVIEW_DIR": str(queue_dir),
        },
        client=object(),
    )
    router = MnemosUiRouter(service)

    response = router.handle("GET", "/api/routing-review", None)
    assert response.status == 200
    payload = json.loads(response.body)
    assert [item["record_id"] for item in payload["records"]] == [alice["record_id"]]
    assert payload["records"][0]["configured_profile_sha256"]
    assert "alice" not in response.body.decode()

    body = json.dumps({"record_id": alice["record_id"]}).encode()
    assert (
        router.handle("POST", "/api/routing-review/ack", body, {"Host": "127.0.0.1"}).status == 403
    )
    acknowledged = router.handle(
        "POST",
        "/api/routing-review/ack",
        body,
        {"Host": "127.0.0.1", "X-Mnemos-CSRF": router.csrf_token},
    )
    assert acknowledged.status == 200
    assert json.loads(acknowledged.body)["acknowledged"] is True

    unconfigured_response = router.handle("GET", "/api/routing-review?unconfigured=1", None)
    assert [item["record_id"] for item in json.loads(unconfigured_response.body)["records"]] == [
        unconfigured["record_id"]
    ]


def test_ui_routing_review_missing_profile_is_explicit(tmp_path: Path) -> None:
    service = ControlPlaneService(
        cwd=tmp_path,
        env={"MNEMOS_ROUTING_REVIEW_DIR": str(tmp_path / "routing")},
        global_config_path=tmp_path / "missing.toml",
        client=object(),
    )
    response = MnemosUiRouter(service).handle("GET", "/api/routing-review", None)
    assert response.status == 400
    assert "MNEMOS_USER_PROFILE_ID" in json.loads(response.body)["error"]
