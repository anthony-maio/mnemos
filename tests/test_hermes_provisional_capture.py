"""Exercise Hermes fallback evidence against the production capture contract."""

from pathlib import Path
from typing import Any

import pytest

from mnemos.adapters.hermes import MnemosHermesProvider
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.contracts import CaptureEvent
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


@pytest.mark.asyncio
async def test_fallback_callbacks_persist_separately_without_creating_facts(tmp_path: Path) -> None:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="alice", enabled=True)
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    memory = LocalMemoryService(
        MnemosEngine(
            config=MnemosConfig(
                modules=ModuleConfig(**{k: False for k in ModuleConfig.model_fields})
            ),
            store=store,
            embedder=SimpleEmbeddingProvider(),
            llm=UnavailableLLMProvider(),
        ),
        registry,
    )
    responses: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []

    class Client:
        async def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
            payloads.append(payload)
            result = memory.capture(CaptureEvent.model_validate(payload))
            responses.append(result)
            return result

    provider = MnemosHermesProvider(client=Client(), registry=registry, user_profile_id="alice")
    provider.initialize("s1", cwd=str(tmp_path), hermes_home=str(tmp_path / "hermes"))
    try:
        for _ in range(2):
            provider.sync_turn("The label is uncertain.", "I understand.")
            assert provider._flush()
        assert len(responses) == 4
        assert all(r["status"] == "committed" and r["fact_eligible"] is False for r in responses)
        assert len(store.list_episodic_events()) == 4
        assert all(job.kind == "capture_only" for job in store.list_jobs())
        assert await memory.maintenance.run_once(force=True) == 4
        assert store.get_all() == []
        assert all(job.result["fact_eligible"] is False for job in store.list_jobs())

        # Exact durable-payload replay remains idempotent even though separate
        # ambiguous callback receipts cannot be associated with one host turn.
        replay = memory.capture(CaptureEvent.model_validate(payloads[0]))
        assert replay["event_id"] == responses[0]["event_id"]
        assert len(store.list_episodic_events()) == 4
        for changes in (
            {"identity_status": "normal"},
            {"source_ref": "unverified-reference"},
            {"event_id": "forged-event"},
            {"identity_version": 2},
            {"role": "tool"},
            {"scope": "global"},
            {"source_kind": "explicit_remember"},
            {"context": {**payloads[0]["context"], "host": "codex"}},
        ):
            changed = CaptureEvent.model_validate({**payloads[0], **changes})
            try:
                rejected = memory.capture(changed)
            except ValueError:
                continue
            assert rejected["status"] == "rejected"
        explicit = memory.capture(CaptureEvent.model_validate(payloads[0]), explicit=True)
        assert explicit["status"] == "rejected"
    finally:
        provider.shutdown()
        store.close()

    reopened = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        assert len(reopened.list_episodic_events(status="deferred")) == 4
        assert reopened.get_all() == []
    finally:
        reopened.close()
