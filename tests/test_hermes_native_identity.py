"""Durable Hermes message rows survive replay but not regeneration after undo."""

from pathlib import Path

import pytest

from mnemos.adapters.hermes import MnemosHermesProvider
from mnemos.client import Outbox
from mnemos.scopes import ScopeRegistry


def setup_provider(tmp_path):
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="alice", enabled=True)

    class Client:
        def __init__(self):
            self.outbox = Outbox(tmp_path / "outbox.json")

        async def capture(self, payload):
            self.outbox.append(payload)
            return {"status": "queued"}

    client = Client()
    provider = MnemosHermesProvider(client=client, registry=registry, user_profile_id="alice")
    provider.initialize("native-session", cwd=str(tmp_path), hermes_home=str(tmp_path / "home"))
    return provider, client


def transcript():
    return [
        {"role": "user", "content": "Use cedar.", "_row_id": 1},
        {"role": "assistant", "content": "Cedar recorded.", "_row_id": 2},
        {"role": "user", "content": "Use maple.", "_row_id": 3},
        {"role": "assistant", "content": "Maple recorded.", "_row_id": 4},
    ]


def test_native_rows_distinguish_regenerated_turn_from_replay(tmp_path: Path):
    provider, client = setup_provider(tmp_path)
    messages = transcript()
    provider.sync_turn("Use maple.", "Maple recorded.", messages=messages)
    assert provider._flush()
    initial = client.outbox.pending()
    assert len(initial) == 4
    provider.on_session_end(messages)
    assert client.outbox.pending() == initial

    provider.on_session_switch("native-session", rewound=True)
    replacement = messages[:2] + [
        {**messages[2], "_row_id": 5},
        {**messages[3], "_row_id": 6},
    ]
    provider.sync_turn("Use maple.", "Maple recorded.", messages=replacement)
    assert provider._flush()
    after = client.outbox.pending()
    assert len(after) == 6
    assert {row["event_id"] for row in initial}.issubset(row["event_id"] for row in after)
    assert len({row["payload"]["source_ref"] for row in after}) == 6
    provider.on_session_end(replacement)
    assert client.outbox.pending() == after


def test_native_rows_survive_truncated_views_and_provider_restart(tmp_path: Path):
    provider, client = setup_provider(tmp_path)
    messages = transcript()
    provider.sync_turn("Use maple.", "Maple recorded.", messages=messages)
    assert provider._flush()
    initial = client.outbox.pending()
    provider.shutdown()
    restarted = MnemosHermesProvider(
        client=client, registry=provider.registry, user_profile_id="alice"
    )
    restarted.initialize("native-session", cwd=str(tmp_path), hermes_home=str(tmp_path / "home"))
    assert "captured 2" in restarted.on_pre_compress(messages[2:], require_checkpoint=True)
    restarted.on_session_end(messages[2:])
    assert client.outbox.pending() == initial


@pytest.mark.parametrize("change", [{"content": "A conflicting payload."}, {"role": "assistant"}])
@pytest.mark.parametrize("restart", [False, True])
def test_native_row_payload_collision_is_not_a_new_occurrence(tmp_path: Path, change, restart):
    provider, client = setup_provider(tmp_path)
    original = transcript()[:1]
    provider.sync_turn("Use cedar.", "", messages=original)
    assert provider._flush()
    before = client.outbox.pending()
    if restart:
        provider.shutdown()
        provider = MnemosHermesProvider(
            client=client, registry=provider.registry, user_profile_id="alice"
        )
        provider.initialize("native-session", cwd=str(tmp_path), hermes_home=str(tmp_path / "home"))
    provider.sync_turn("", "", messages=[{**original[0], **change}])
    assert provider._flush() is False
    assert client.outbox.pending() == before
    assert any(row["status"] == "capture_failed" for row in provider.get_lifecycle_diagnostics())


@pytest.mark.parametrize("row_id", [True, 0, -1, "1", 1.5, None])
def test_malformed_native_identity_is_not_silently_replaced(tmp_path: Path, row_id):
    provider, client = setup_provider(tmp_path)
    messages = [{"role": "user", "content": "Use cedar.", "_row_id": row_id}]
    with pytest.raises(ValueError, match="row identity"):
        provider.sync_turn("Use cedar.", "", messages=messages)
    assert client.outbox.pending() == []


def test_repeated_native_row_identity_in_one_snapshot_is_rejected(tmp_path: Path):
    provider, client = setup_provider(tmp_path)
    messages = transcript()
    messages[1]["_row_id"] = messages[0]["_row_id"]
    with pytest.raises(ValueError, match="row identity"):
        provider.on_pre_compress(messages, require_checkpoint=True)
    assert client.outbox.pending() == []


def test_partial_native_identity_is_retained_without_fact_eligibility(tmp_path: Path):
    provider, client = setup_provider(tmp_path)
    messages = transcript()
    del messages[-1]["_row_id"]
    provider.sync_turn("Use maple.", "Maple recorded.", messages=messages)
    assert provider._flush()
    rows = client.outbox.pending()
    assert len(rows) == 4
    assert all(row["payload"].get("identity_status") == "provisional_unassociated" for row in rows)
    assert any(
        row["status"] == "partial_native_identity" for row in provider.get_lifecycle_diagnostics()
    )


def test_native_row_numbers_do_not_alias_other_sessions_or_projects(tmp_path: Path):
    provider, client = setup_provider(tmp_path)
    messages = transcript()
    provider.sync_turn("Use maple.", "Maple recorded.", messages=messages)
    assert provider._flush()
    initial = {row["event_id"] for row in client.outbox.pending()}
    provider.on_session_switch("other-session")
    provider.sync_turn("Use maple.", "Maple recorded.", messages=messages)
    assert provider._flush()
    assert len(client.outbox.pending()) == 8
    assert initial.issubset(row["event_id"] for row in client.outbox.pending())

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    provider.registry.register_project(foreign, user_profile_id="alice", enabled=True)
    provider.initialize("native-session", cwd=str(foreign), hermes_home=str(tmp_path / "home"))
    provider.sync_turn("Use maple.", "Maple recorded.", messages=messages)
    assert provider._flush()
    assert len(client.outbox.pending()) == 12


@pytest.mark.asyncio
async def test_partial_native_snapshot_stays_nonextractable_in_real_service(tmp_path: Path):
    from mnemos.config import MnemosConfig, ModuleConfig
    from mnemos.contracts import CaptureEvent
    from mnemos.engine import MnemosEngine
    from mnemos.memory_service import LocalMemoryService
    from mnemos.utils.embeddings import SimpleEmbeddingProvider
    from mnemos.utils.llm import UnavailableLLMProvider
    from mnemos.utils.storage import SQLiteStore

    provider, client = setup_provider(tmp_path)
    messages = transcript()
    del messages[-1]["_row_id"]
    provider.sync_turn("Use maple.", "Maple recorded.", messages=messages)
    assert provider._flush()
    store = SQLiteStore(str(tmp_path / "service.sqlite"))
    service = LocalMemoryService(
        MnemosEngine(
            config=MnemosConfig(
                modules=ModuleConfig(**{key: False for key in ModuleConfig.model_fields})
            ),
            store=store,
            embedder=SimpleEmbeddingProvider(),
            llm=UnavailableLLMProvider(),
        ),
        provider.registry,
    )
    try:
        for row in client.outbox.pending():
            result = service.capture(CaptureEvent.model_validate(row["payload"]))
            assert result["status"] == "committed" and result["fact_eligible"] is False
        assert await service.maintenance.run_once(force=True) == 4
        assert len(store.list_episodic_events()) == 4
        assert store.get_all() == []
    finally:
        store.close()


@pytest.mark.asyncio
async def test_committed_native_row_rejects_changed_payload_after_service_restart(tmp_path: Path):
    from mnemos.config import MnemosConfig, ModuleConfig
    from mnemos.contracts import CaptureEvent
    from mnemos.engine import MnemosEngine
    from mnemos.memory_service import LocalMemoryService
    from mnemos.utils.embeddings import SimpleEmbeddingProvider
    from mnemos.utils.llm import UnavailableLLMProvider
    from mnemos.utils.storage import SQLiteStore

    provider, client = setup_provider(tmp_path)
    provider.sync_turn("Use cedar.", "", messages=transcript()[:1])
    assert provider._flush()
    original = client.outbox.pending()[0]["payload"]
    expected_job = None
    for restart in (False, True):
        store = SQLiteStore(str(tmp_path / "service.sqlite"))
        service = LocalMemoryService(
            MnemosEngine(
                config=MnemosConfig(
                    modules=ModuleConfig(**{key: False for key in ModuleConfig.model_fields})
                ),
                store=store,
                embedder=SimpleEmbeddingProvider(),
                llm=UnavailableLLMProvider(),
            ),
            provider.registry,
        )
        try:
            accepted = service.capture(CaptureEvent.model_validate(original))
            assert accepted["status"] == "committed"
            if restart:
                assert accepted["job_id"] == expected_job
            expected_job = accepted["job_id"]
            if not restart:
                assert await service.maintenance.run_once(force=True) == 1
            before = [row.model_dump(mode="json") for row in store.list_episodic_events()]
            for change in ({"content": "Conflicting native content."}, {"role": "assistant"}):
                changed = provider._capture_payload(
                    provider._current_context(),
                    role=change.get("role", original["role"]),
                    content=change.get("content", original["content"]),
                    source=original["source_ref"],
                )
                assert changed["event_id"] == original["event_id"]
                with pytest.raises(ValueError, match="different payload"):
                    service.capture(CaptureEvent.model_validate(changed))
            assert [row.model_dump(mode="json") for row in store.list_episodic_events()] == before
            assert len(store.list_jobs()) == 1
        finally:
            store.close()
