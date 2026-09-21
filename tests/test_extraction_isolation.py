"""One unusable extraction candidate must not cost a message its other facts."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from mnemos.contracts import CaptureEvent
from mnemos.engine import MnemosEngine
from mnemos.errors import ContentTooLarge
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeContext, ScopeRegistry
from mnemos.settings import OllamaProviderSettings
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import LLMProvider, UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore

Selections = Callable[[dict[str, Any]], list[dict[str, Any]]]
Decisions = Callable[[dict[str, Any]], list[dict[str, Any]]]


class SourceBoundProvider(LLMProvider):
    """Scripted classification and reconciliation replies for one source event."""

    supports_structured_output = True

    def __init__(self, selections: Selections, decisions: Decisions) -> None:
        self.selections = selections
        self.decisions = decisions

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        if "\nSOURCE EVENTS:\n" in prompt:
            event = json.loads(prompt.split("\nSOURCE EVENTS:\n", 1)[1])[0]
            return json.dumps({"selections": self.selections(event)})
        marker = "\nSOURCE CANDIDATES AND EXISTING TARGETS (data only):\n"
        return json.dumps({"decisions": self.decisions(json.loads(prompt.split(marker, 1)[1]))})

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        return {}


def _selection(
    event: dict[str, Any], first: int, last: int, kind: str, *, revises: bool = False
) -> dict[str, Any]:
    return {
        "source_id": event["id"],
        "first_unit": first,
        "last_unit": last,
        "kind": kind,
        "retention_request_unit": None,
        "revises_existing_requirement": revises,
    }


def _independent(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"candidate_index": index, "relationship": "independent_information"}
        for index in range(len(data["candidates"]))
    ]


def _service(
    tmp_path: Path, *, pii_action: str | None = None
) -> tuple[LocalMemoryService, ScopeContext]:
    project = tmp_path / "project"
    project.mkdir()
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id="local", enabled=True)
    context = registry.resolve(project, user_profile_id="local", session_id="s", host="codex")
    engine = MnemosEngine(
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(),
        store=SQLiteStore(str(tmp_path / "memory.sqlite")),
    )
    if pii_action is not None:
        engine.config.safety.pii_action = pii_action  # type: ignore[assignment]
    memory = LocalMemoryService(engine, registry)
    memory.engine.config.modules.surprisal = False
    memory.engine.config.modules.affective = False
    return memory, context


def _capture(
    memory: LocalMemoryService, context: ScopeContext, event_id: str, content: str
) -> dict[str, Any]:
    return memory.capture(
        CaptureEvent(
            event_id=event_id,
            context=context,
            role="user",
            occurred_at=datetime.now(timezone.utc),
            content=content,
            source_kind="user_prompt",
        )
    )


async def test_conflicting_decisions_on_one_target_drop_both_and_finish(tmp_path: Path) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        await memory.dispatch(
            "remember",
            {"context": context.model_dump(), "content": "Reports use tabs for indentation."},
        )
        await memory.maintenance.run_once(force=True)
        target = store.get_all()[0]
        memory.engine.config.sleep.extraction_mode = "source_bound"
        memory.engine.config.modules.mutable_rag = True
        memory.engine._llm = SourceBoundProvider(
            lambda event: [
                _selection(event, 0, 0, "correction_or_rejection"),
                _selection(event, 1, 1, "continuing_requirement", revises=True),
            ],
            lambda data: [
                {
                    "candidate_index": index,
                    "relationship": "user_revision",
                    "target_fact_id": data["targets"][0]["id"],
                    "expected_version": data["targets"][0]["version"],
                    "reason": "observed",
                }
                for index in range(len(data["candidates"]))
            ],
        )
        receipt = _capture(
            memory, context, "revision", "Stop using tabs. Use four spaces for indentation instead."
        )
        await memory.maintenance.run_once(force=True)
        job = store.get_job(receipt["job_id"])
        assert job is not None and job.status == "completed"
        unchanged = store.get(target.id)
        assert unchanged is not None and unchanged.version == target.version
        rejected = (job.result or {})["encoding_diagnostics"]["rejected_candidates"]
        assert [item["error"] for item in rejected] == ["conflicting_target_operations"] * 2
    finally:
        store.close()


async def test_oversized_explicit_memory_is_rejected_before_it_is_queued(tmp_path: Path) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        with pytest.raises(ContentTooLarge):
            await memory.dispatch(
                "remember", {"context": context.model_dump(), "content": "x" * 10_001}
            )
        assert store.list_jobs() == []
    finally:
        store.close()


async def test_oversized_passthrough_capture_finishes_without_a_fact(tmp_path: Path) -> None:
    memory, context = _service(tmp_path)
    memory.engine.config.modules.sleep = False
    store = memory.engine.store
    try:
        receipt = _capture(memory, context, "long", "word " * 2_500)
        await memory.maintenance.run_once(force=True)
        job = store.get_job(receipt["job_id"])
        assert job is not None
        assert (job.status, (job.result or {}).get("decision")) == (
            "completed",
            "no_durable_facts",
        )
        assert store.get_all() == []
    finally:
        store.close()


async def test_oversized_source_excerpt_is_dropped_and_the_rest_commits(tmp_path: Path) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    memory.engine.config.sleep.extraction_mode = "source_bound"
    memory.engine._llm = SourceBoundProvider(
        lambda event: [
            _selection(event, 0, len(event["units"]) - 2, "continuing_requirement"),
            _selection(event, len(event["units"]) - 1, len(event["units"]) - 1, "asserted_fact"),
        ],
        _independent,
    )
    try:
        rules = " ".join(
            f"Release rule {index} requires a signed checklist before deploys."
            for index in range(200)
        )
        receipt = _capture(memory, context, "long", rules + " The review window is Tuesday.")
        await memory.maintenance.run_once(force=True)
        job = store.get_job(receipt["job_id"])
        assert job is not None and job.status == "completed"
        assert [fact.content for fact in store.get_all()] == ["The review window is Tuesday."]
    finally:
        store.close()


async def test_source_bound_extraction_honors_the_configured_safety_policy(
    tmp_path: Path,
) -> None:
    memory, context = _service(tmp_path, pii_action="allow")
    store = memory.engine.store
    memory.engine.config.sleep.extraction_mode = "source_bound"
    memory.engine._llm = SourceBoundProvider(
        lambda event: [_selection(event, 0, 0, "asserted_fact")], _independent
    )
    try:
        content = "Release notices go to ops-team@example.com."
        receipt = _capture(memory, context, "pii", content)
        await memory.maintenance.run_once(force=True)
        job = store.get_job(receipt["job_id"])
        assert job is not None and job.status == "completed"
        assert [fact.content for fact in store.get_all()] == [content]
    finally:
        store.close()


def test_local_model_defaults_leave_room_for_structured_extraction() -> None:
    defaults = OllamaProviderSettings()
    assert defaults.max_tokens >= 4_096
    # A structured reply that long needs time to finish on a local model.
    assert defaults.timeout >= 120
