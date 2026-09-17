from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from benchmarks.local_memory.contracts import FactRecord, SessionScenario, SourceEvent
from benchmarks.local_memory.harness import DEFAULT_PROFILES
from benchmarks.local_memory.ollama_adapter import (
    OllamaEvaluatorAdapter,
    _TargetMappingError,
)
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeContext, ScopeRegistry
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


def _production_service(
    tmp_path: Path, profile: str = "alice"
) -> tuple[LocalMemoryService, ScopeContext]:
    registry = ScopeRegistry(profiles=[profile])
    registry.register_project(tmp_path, user_profile_id=profile, enabled=True)
    context = registry.resolve(
        tmp_path, user_profile_id=profile, session_id="session-v3", host="codex"
    )
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    engine = MnemosEngine(
        llm=UnavailableLLMProvider(), embedder=SimpleEmbeddingProvider(), store=store
    )
    return LocalMemoryService(engine, registry), context


def _scenario(*events: SourceEvent, status: str = "superseded") -> SessionScenario:
    return SessionScenario(
        session_id="session-v3",
        split="development",
        host="codex",
        project_id="project-v3",
        facts=(
            FactRecord(
                fact_id="fact-v3",
                content="Wrapped durable statement",
                critical=False,
                scope="project",
                source_label="synthetic_seed",
                gold_status=status,
                evidence_event_ids=("remember-v3",),
            ),
        ),
        queries=(),
        state_pairs=(),
        source_label="synthetic_seed",
        human_acceptance="not_run",
        events=events,
        scenario_types=("correction",),
    )


def _remember(operation: str = "remember") -> SourceEvent:
    return SourceEvent(
        event_id="remember-v3",
        occurred_at="2026-01-01T00:00:00Z",
        role="user",
        content="Remember explicitly: Wrapped durable statement",
        source_kind="explicit_remember" if operation == "remember" else "user_prompt",
        write_mode="explicit" if operation == "remember" else "automatic",
        lifecycle_operation=operation,
        evidence_fact_ids=("fact-v3",),
    )


def _action(operation: str, *, source_id: str = "remember-v3") -> SourceEvent:
    return SourceEvent(
        event_id=f"{operation}-v3",
        occurred_at="2026-01-01T00:01:00Z",
        role="user",
        content=(
            "Please replace the statement with the current plain text."
            if operation == "correct"
            else "Forget the temporary statement."
        ),
        source_kind=f"explicit_{operation}",
        write_mode="explicit",
        lifecycle_operation=operation,
        evidence_fact_ids=(),
        target_source_event_id=source_id,
        target_fact_id="fact-v3",
        replacement_content=("The current plain statement." if operation == "correct" else None),
    )


def _adapter(profile: str = "alice") -> OllamaEvaluatorAdapter:
    return OllamaEvaluatorAdapter(lambda _: None, user_profile_id=profile)


def test_production_service_resolves_wrapped_remember_and_uses_plain_correction(
    tmp_path: Path,
) -> None:
    service, context = _production_service(tmp_path)
    adapter = _adapter()
    profile = DEFAULT_PROFILES[0]
    session = _scenario(_remember(), _action("correct"))
    try:
        adapter._event_operation(
            service,
            session.events[0],
            session,
            {**context.model_dump(), "session_id": "session-v3-baseline-r1"},
            profile,
            1,
        )
        asyncio.run(service.maintenance.run_once(force=True))
        before = service.engine.store.get_all()[0]
        result = adapter._event_operation(
            service,
            session.events[1],
            session,
            {**context.model_dump(), "session_id": "session-v3-baseline-r1"},
            profile,
            1,
        )
        assert result["status"] == "committed"
        asyncio.run(service.maintenance.run_once(force=True))
        after = service.engine.store.get(before.id)
        assert after is not None
        assert after.id == before.id
        assert after.version == 2
        assert after.content == "The current plain statement."
    finally:
        cast(SQLiteStore, service.engine.store).close()


def test_strict_mapping_accepts_an_earlier_captured_user_event(tmp_path: Path) -> None:
    service, context = _production_service(tmp_path)
    service.engine.config.modules.surprisal = False
    service.engine.config.modules.sleep = False
    adapter = _adapter()
    profile = DEFAULT_PROFILES[0]
    source = _remember("capture")
    action = _action("correct")
    session = _scenario(source, action)
    runtime_context = {**context.model_dump(), "session_id": "session-v3-baseline-r1"}
    try:
        adapter._event_operation(service, source, session, runtime_context, profile, 1)
        asyncio.run(service.maintenance.run_once(force=True))
        target = adapter._resolve_runtime_target(
            service,
            action,
            session,
            context=runtime_context,
            profile=profile,
            repetition=1,
        )
        assert target is not None
        assert service.engine.store.get(target["id"]) is not None
    finally:
        cast(SQLiteStore, service.engine.store).close()


def test_strict_forget_uses_source_evidence_and_does_not_match_text(tmp_path: Path) -> None:
    service, context = _production_service(tmp_path)
    adapter = _adapter()
    profile = DEFAULT_PROFILES[0]
    remember = _remember()
    forget = _action("forget")
    session = _scenario(remember, forget, status="forgotten")
    runtime_context = {**context.model_dump(), "session_id": "session-v3-baseline-r1"}
    try:
        adapter._event_operation(service, remember, session, runtime_context, profile, 1)
        asyncio.run(service.maintenance.run_once(force=True))
        chunk = service.engine.store.get_all()[0]
        result = adapter._event_operation(service, forget, session, runtime_context, profile, 1)
        assert result["fact_id"] == chunk.id
        assert result["forgotten"] is True
        assert service.engine.store.get(chunk.id) is None
    finally:
        cast(SQLiteStore, service.engine.store).close()


def test_strict_mapping_rejects_foreign_profile_and_ambiguous_evidence(tmp_path: Path) -> None:
    service, context = _production_service(tmp_path)
    adapter = _adapter()
    profile = DEFAULT_PROFILES[0]
    session = _scenario(_remember(), _action("correct"))
    runtime_context = {**context.model_dump(), "session_id": "session-v3-baseline-r1"}
    try:
        adapter._event_operation(service, session.events[0], session, runtime_context, profile, 1)
        asyncio.run(service.maintenance.run_once(force=True))
        original = service.engine.store.get_all()[0]
        service.engine.store.store(
            MemoryChunk(
                content=original.content,
                embedding=original.embedding,
                metadata={**original.metadata, "user_profile_id": "bob"},
            )
        )
        resolved = adapter._resolve_runtime_target(
            service,
            session.events[1],
            session,
            context=runtime_context,
            profile=profile,
            repetition=1,
        )
        assert resolved is not None
        assert resolved["id"] == original.id

        service.engine.store.store(
            MemoryChunk(
                content="independent duplicate",
                embedding=original.embedding,
                metadata=dict(original.metadata),
            )
        )
        with pytest.raises(_TargetMappingError, match="multiple facts") as exc_info:
            adapter._resolve_runtime_target(
                service,
                session.events[1],
                session,
                context=runtime_context,
                profile=profile,
                repetition=1,
            )
        assert exc_info.value.status == "target_mapping_invalid"
    finally:
        cast(SQLiteStore, service.engine.store).close()


def test_strict_mapping_reports_missing_source_without_mutation(tmp_path: Path) -> None:
    service, context = _production_service(tmp_path)
    adapter = _adapter()
    profile = DEFAULT_PROFILES[0]
    session = _scenario(_remember(), _action("correct"))
    runtime_context = {**context.model_dump(), "session_id": "session-v3-baseline-r1"}
    try:
        with pytest.raises(_TargetMappingError) as exc_info:
            adapter._resolve_runtime_target(
                service,
                session.events[1],
                session,
                context=runtime_context,
                profile=profile,
                repetition=1,
            )
        assert exc_info.value.status == "missing_accepted_target"
        assert service.engine.store.get_all() == []
    finally:
        cast(SQLiteStore, service.engine.store).close()


def test_strict_mapping_rejects_scope_and_temporal_or_context_mismatches(tmp_path: Path) -> None:
    service, context = _production_service(tmp_path)
    adapter = _adapter()
    profile = DEFAULT_PROFILES[0]
    source = _remember()
    action = _action("correct")
    session = _scenario(source, action)
    runtime_context = {**context.model_dump(), "session_id": "session-v3-baseline-r1"}
    try:
        adapter._event_operation(service, source, session, runtime_context, profile, 1)
        asyncio.run(service.maintenance.run_once(force=True))

        wrong_scope = replace(session.facts[0], scope="workspace")
        with pytest.raises(_TargetMappingError) as scope_error:
            adapter._resolve_runtime_target(
                service,
                action,
                replace(session, facts=(wrong_scope,)),
                context=runtime_context,
                profile=profile,
                repetition=1,
            )
        assert scope_error.value.status == "target_mapping_invalid"

        late_source = replace(source, occurred_at="2026-01-01T00:02:00Z")
        with pytest.raises(_TargetMappingError) as time_error:
            adapter._resolve_runtime_target(
                service,
                action,
                replace(session, events=(late_source, action)),
                context=runtime_context,
                profile=profile,
                repetition=1,
            )
        assert time_error.value.status == "target_mapping_invalid"

        foreign_source = replace(source, target_project_id="foreign-project")
        with pytest.raises(_TargetMappingError) as context_error:
            adapter._resolve_runtime_target(
                service,
                action,
                replace(session, events=(foreign_source, action)),
                context=runtime_context,
                profile=profile,
                repetition=1,
            )
        assert context_error.value.status == "target_mapping_invalid"
    finally:
        cast(SQLiteStore, service.engine.store).close()
