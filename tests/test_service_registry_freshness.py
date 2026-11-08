from datetime import datetime, timezone

import pytest

from mnemos.contracts import CaptureEvent
from mnemos.service import MnemosService
from tests.test_memory_service import service


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["disabled", "missing", "malformed"])
async def test_default_owner_rejects_capture_after_external_registry_change(
    tmp_path, monkeypatch, change
):
    initial, context = service(tmp_path)
    registry_path = tmp_path / "registry.json"
    initial.registry.save(registry_path)
    monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(registry_path))
    monkeypatch.setattr("mnemos.runtime.build_configured_engine", lambda **kwargs: initial.engine)
    owner = MnemosService(descriptor_path=tmp_path / "runtime.json")
    memory = owner._make_default_dispatcher()
    event = CaptureEvent(
        event_id="queued-before-revocation",
        context=context,
        occurred_at=datetime.now(timezone.utc),
        role="assistant",
        content="This queued event must not bypass revocation.",
        source_kind="codex_hook",
    )
    try:
        if change == "disabled":
            initial.registry.projects[context.project_id].enabled = False
            initial.registry.save(registry_path)
        elif change == "missing":
            registry_path.unlink()
        else:
            registry_path.write_text("{")
        with pytest.raises((ValueError, OSError)):
            await memory.dispatch("capture", event.model_dump(mode="json"))
        assert initial.engine.store.list_episodic_events() == []
        assert initial.engine.store.list_jobs() == []
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_default_owner_refreshes_job_authorization_and_recovers(tmp_path, monkeypatch):
    initial, context = service(tmp_path)
    registry_path = tmp_path / "registry.json"
    initial.registry.save(registry_path)
    monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(registry_path))
    monkeypatch.setattr("mnemos.runtime.build_configured_engine", lambda **kwargs: initial.engine)
    memory = MnemosService(descriptor_path=tmp_path / "runtime.json")._make_default_dispatcher()
    event = CaptureEvent(
        event_id="before-pause",
        context=context,
        occurred_at=datetime.now(timezone.utc),
        role="assistant",
        content="A queued job still needs current authorization.",
        source_kind="codex_hook",
    )
    try:
        assert memory.capture(event)["status"] == "committed"
        job = initial.engine.store.list_jobs()[0]
        initial.registry.projects[context.project_id].enabled = False
        initial.registry.save(registry_path)
        with pytest.raises(ValueError, match="not enabled"):
            await memory.process_job(job)
        initial.registry.projects[context.project_id].enabled = True
        initial.registry.save(registry_path)
        second = event.model_copy(update={"event_id": "after-resume"})
        assert memory.capture(second)["status"] == "committed"
        assert memory.engine._scope_registry is memory.registry
        assert len(initial.engine.store.list_episodic_events()) == 2
    finally:
        await memory.close()
