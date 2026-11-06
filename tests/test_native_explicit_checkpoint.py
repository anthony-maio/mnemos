"""Acceptance rejects unbound host output and preserves original replay identities."""

from copy import deepcopy
import base64
import json

import pytest

from tests.test_installed_artifact import load_script


@pytest.fixture
def runner():
    return load_script("scripts/check-native-explicit.py")


def test_codex_auth_preflight_rejects_expiring_credentials_without_exposing_them(runner, tmp_path):
    path = tmp_path / "auth.json"

    def credential(expiry):
        payload = (
            base64.urlsafe_b64encode(json.dumps({"exp": expiry}).encode()).decode().rstrip("=")
        )
        return {"auth_mode": "chatgpt", "tokens": {"access_token": "header." + payload + ".secret"}}

    raw = json.dumps(credential(5000))
    path.write_text(raw)
    result = runner.codex_auth_preflight(path, now=1000)
    assert result == {"auth_mode": "chatgpt", "expires_at": 5000, "minimum_remaining_seconds": 3600}
    assert path.read_text() == raw
    for value in (
        credential(4599),
        credential(True),
        {"auth_mode": "apikey", "OPENAI_API_KEY": "private"},
        {},
    ):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="fresh Codex") as failure:
            runner.codex_auth_preflight(path, now=1000)
        assert "secret" not in str(failure.value) and "private" not in str(failure.value)


def completed():
    process = {"status": "completed", "auth_copy_removed": True, "session_id": "native-1"}
    records = [
        {"type": "assistant", "session_id": "native-1", "message": {"model": "claude-sonnet-4-6"}},
        {
            "type": "result",
            "session_id": "native-1",
            "subtype": "success",
            "is_error": False,
            "result": "Completed the requested tools.",
            "total_cost_usd": 0.2,
        },
    ]
    return process, records


def test_native_terminal_requires_actual_model_session_and_cleanup(runner):
    process, records = completed()
    assert runner.verify_native_result(process, records)["model"] == "claude-sonnet-4-6"
    mutations = [
        lambda p, r: p.update(auth_copy_removed=False),
        lambda p, r: p.update(status="failed"),
        lambda p, r: r[0].update(session_id="another"),
        lambda p, r: r[0]["message"].update(model="another"),
        lambda p, r: r[1].update(is_error=True),
        lambda p, r: r[1].update(subtype="error_max_budget_usd"),
        lambda p, r: r[1].update(result=""),
        lambda p, r: r.pop(0),
    ]
    for mutate in mutations:
        changed_process, changed_records = deepcopy(process), deepcopy(records)
        mutate(changed_process, changed_records)
        with pytest.raises(ValueError):
            runner.verify_native_result(changed_process, changed_records)


def test_replay_uses_original_payload_and_only_committed_target_sources(runner):
    records = [
        {
            "operation": "remember",
            "payload": {"content": "marker-731", "event_id": "e1"},
            "result": {"status": "committed", "event_id": "stored-e1"},
        },
        {
            "operation": "capture",
            "payload": {"content": "marker-731", "event_id": "e2"},
            "result": {"status": "degraded"},
        },
        {
            "operation": "capture",
            "payload": {"content": "unrelated", "event_id": "e3"},
            "result": {"status": "committed"},
        },
        {
            "operation": "correct",
            "payload": {"content": "marker-731"},
            "result": {"status": "committed"},
        },
    ]
    selected = runner.original_replay_records(records, ("marker-731",))
    assert selected == [records[0]]
    assert selected[0]["payload"]["event_id"] == "e1"
    assert selected[0]["result"]["event_id"] == "stored-e1"
    with pytest.raises(ValueError):
        runner.original_replay_records(records, ("not-present",))


def test_auth_preflight_rejects_expired_or_short_lived_login_without_copying_it(runner, tmp_path):
    path = tmp_path / ".credentials.json"
    for expiry in (0, 4_000_000, True, "later", None):
        raw = json.dumps({"claudeAiOauth": {"accessToken": "test-token", "expiresAt": expiry}})
        path.write_text(raw)
        with pytest.raises(ValueError, match="fresh isolated Claude login"):
            runner.auth_preflight(path, now=1000)
        assert path.read_text() == raw
    path.write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "test-token", "expiresAt": 5_000_000}})
    )
    result = runner.auth_preflight(path, now=1000)
    assert result == {"expires_at_ms": 5_000_000, "minimum_remaining_seconds": 3600}
    assert "test-token" not in json.dumps(result)
