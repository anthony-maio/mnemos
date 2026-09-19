"""Hermes sends cumulative transcripts at turn and checkpoint boundaries."""

from pathlib import Path
from typing import Any

import pytest

from mnemos.adapters.hermes import MnemosHermesProvider
from mnemos.client import Outbox
from mnemos.scopes import ScopeRegistry


class QueuedClient:
    def __init__(self, path: Path) -> None:
        self.outbox = Outbox(path)

    async def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.outbox.append(payload)
        return {"status": "queued", "committed": False}


def provider(tmp_path: Path) -> tuple[MnemosHermesProvider, QueuedClient]:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="alice", enabled=True)
    client = QueuedClient(tmp_path / "outbox.json")
    instance = MnemosHermesProvider(client=client, registry=registry, user_profile_id="alice")
    instance.initialize("native-session", cwd=str(tmp_path), hermes_home=str(tmp_path / "hermes"))
    return instance, client


def test_appending_a_turn_does_not_recapture_earlier_transcript_entries(tmp_path: Path) -> None:
    instance, client = provider(tmp_path)
    first = [
        {"role": "user", "content": "Use cedar for the release label."},
        {"role": "assistant", "content": "I will use cedar."},
    ]
    instance.sync_turn("", "", messages=first)
    assert instance._flush()
    initial = {item["event_id"] for item in client.outbox.pending()}

    entire = first + [
        {"role": "user", "content": "Correct the release label to maple."},
        {"role": "assistant", "content": "I will use maple."},
    ]
    instance.sync_turn("", "", messages=entire)
    instance.on_session_end(entire)
    assert instance._flush()
    pending = client.outbox.pending()
    assert len(pending) == 4
    assert initial.issubset({item["event_id"] for item in pending})


def test_identical_messages_at_different_transcript_positions_are_distinct(tmp_path: Path) -> None:
    instance, client = provider(tmp_path)
    turn = [
        {"role": "user", "content": "Please continue."},
        {"role": "assistant", "content": "Continuing."},
    ]
    instance.sync_turn("", "", messages=turn + turn)
    assert instance._flush()
    assert len(client.outbox.pending()) == 4


def test_compression_checkpoint_reuses_the_completed_turn_occurrences(tmp_path: Path) -> None:
    instance, client = provider(tmp_path)
    messages = [
        {"role": "user", "content": "Use the project convention."},
        {"role": "assistant", "content": "Understood."},
    ]
    instance.sync_turn("", "", messages=messages)
    assert instance._flush()
    before = client.outbox.pending()
    assert "captured 2" in instance.on_pre_compress(messages, require_checkpoint=True)
    assert client.outbox.pending() == before


def test_recreated_provider_replays_the_identical_durable_payload(tmp_path: Path) -> None:
    instance, client = provider(tmp_path)
    messages = [{"role": "user", "content": "Keep the release label stable."}]
    instance.sync_turn("", "", messages=messages)
    assert instance._flush()
    before = client.outbox.pending()
    instance.shutdown()

    restarted = MnemosHermesProvider(
        client=client, registry=instance.registry, user_profile_id="alice"
    )
    restarted.initialize("native-session", cwd=str(tmp_path), hermes_home=str(tmp_path / "hermes"))
    restarted.sync_turn("", "", messages=messages)
    assert restarted._flush()
    assert client.outbox.pending() == before


def test_concurrent_providers_share_one_first_observation_time(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from mnemos.adapters.hermes_occurrences import occurrence_time

    home = str(tmp_path / "hermes")
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(
            pool.map(
                lambda stamp: occurrence_time(home, "same-scoped-event", stamp),
                ["2026-09-19T12:00:00+00:00", "2026-09-19T12:00:01+00:00"],
            )
        )
    assert values[0] == values[1]
    assert occurrence_time(home, "another-event", "2026-09-19T12:00:02+00:00") != values[0]


def test_invalid_occurrence_ledger_does_not_acknowledge_capture(tmp_path: Path) -> None:
    import sqlite3

    instance, client = provider(tmp_path)
    directory = tmp_path / "hermes" / "mnemos"
    directory.mkdir(parents=True)
    (directory / "occurrences.sqlite").write_bytes(b"broken database")
    with pytest.raises(sqlite3.DatabaseError):
        instance.sync_turn("Do not lose this fact", "Acknowledged")
    assert client.outbox.pending() == []


def test_relative_hermes_home_cannot_select_identity_state_from_process_cwd(
    tmp_path: Path,
) -> None:
    instance, client = provider(tmp_path)
    instance.initialize("native-session", cwd=str(tmp_path), hermes_home="relative-home")
    with pytest.raises(ValueError, match="must be absolute"):
        instance.sync_turn("Remember this", "Acknowledged")
    assert client.outbox.pending() == []


def test_missing_transcript_is_explicitly_provisional_and_separate_from_normal_capture(
    tmp_path: Path,
) -> None:
    instance, client = provider(tmp_path)
    instance.sync_turn("Use project conventions", "Understood")
    assert instance._flush()
    provisional = client.outbox.pending()
    assert len(provisional) == 2
    assert all(
        item["payload"].get("identity_status") == "provisional_unassociated" for item in provisional
    )
    assert any(
        row["status"] == "transcript_identity_unavailable"
        for row in instance.get_lifecycle_diagnostics()
    )
    instance.sync_turn(
        "Use project conventions",
        "Understood",
        messages=[
            {"role": "user", "content": "Use project conventions"},
            {"role": "assistant", "content": "Understood"},
        ],
    )
    assert instance._flush()
    pending = client.outbox.pending()
    assert len(pending) == 4
    assert sum(row["payload"].get("identity_status", "normal") == "normal" for row in pending) == 2
