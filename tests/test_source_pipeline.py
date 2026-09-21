"""Write-free two-stage source pipeline contract tests."""

from datetime import datetime, timezone
import json

import pytest

from benchmarks.local_memory.source_pipeline import run_source_pipeline
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.llm import LLMProvider


def event(event_id: str, content: str, role: str = "user") -> EpisodicEvent:
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=f"fingerprint-{event_id}",
        role=role,
        content=content,
        normalized_content=content.casefold(),
        scope="project",
        scope_id="project-1",
        source="manual",
        metadata={"user_profile_id": "profile-1", "project_id": "project-1"},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def target(content: str = "The marker is cedar.") -> MemoryChunk:
    return MemoryChunk(
        id="fact-1",
        content=content,
        version=3,
        metadata={
            "user_profile_id": "profile-1",
            "scope": "project",
            "scope_id": "project-1",
        },
    )


def selection(source_id: str, kind: str = "asserted_fact") -> dict[str, object]:
    return {
        "source_id": source_id,
        "first_unit": 0,
        "last_unit": 0,
        "kind": kind,
    }


class Provider(LLMProvider):
    supports_structured_output = True

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls = 0

    async def predict(self, prompt: str, **kwargs: object) -> str:
        self.calls += 1
        return json.dumps(self.response)

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        return {}


@pytest.mark.asyncio
async def test_mixed_pipeline_preserves_exact_content_evidence_and_version() -> None:
    sources = [
        event("add-source", "The review window is Tuesday."),
        event("support-source", "The marker is cedar."),
        event("unresolved-source", "The marker changed, but no replacement is accepted."),
    ]
    selection_provider = Provider(
        {
            "selections": [
                selection("add-source"),
                selection("support-source"),
                selection("unresolved-source", "correction_or_rejection"),
            ]
        }
    )
    reconciliation_provider = Provider(
        {
            "decisions": [
                {"candidate_index": 0, "operation": "add"},
                {
                    "candidate_index": 1,
                    "operation": "support",
                    "target_fact_id": "fact-1",
                    "expected_version": 3,
                },
                {
                    "candidate_index": 2,
                    "operation": "unresolved",
                    "reason": "No accepted replacement target was observed.",
                },
            ]
        }
    )

    result = await run_source_pipeline(
        sources,
        [target()],
        selection_provider=selection_provider,
        reconciliation_provider=reconciliation_provider,
    )

    assert [item.operation for item in result.reconciliations] == [
        "add",
        "support",
        "unresolved",
    ]
    assert [(item.operation, item.content, item.evidence_ids) for item in result.proposals] == [
        ("add", sources[0].content, ["add-source"]),
        ("support", sources[1].content, ["support-source"]),
    ]
    assert result.proposals[1].target_fact_id == "fact-1"
    assert result.proposals[1].expected_version == 3
    assert result.reviews[0].content == sources[2].content
    assert result.reviews[0].reason == "No accepted replacement target was observed."


@pytest.mark.asyncio
async def test_all_discarded_candidates_skip_reconciliation() -> None:
    source = event("discarded", "Which marker should we use?")
    selection_provider = Provider({"selections": [selection("discarded", "question")]})

    class FailingProvider(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            raise AssertionError("reconciliation must not run without candidates")

    result = await run_source_pipeline(
        [source],
        [],
        selection_provider=selection_provider,
        reconciliation_provider=FailingProvider({}),
    )
    assert result.candidates == []
    assert result.proposals == []
    assert result.reviews == []


@pytest.mark.asyncio
async def test_assistant_conflict_is_review_only() -> None:
    source = event("assistant-source", "The report disputes the marker.", role="assistant")
    selection_provider = Provider(
        {"selections": [selection("assistant-source", "attributed_history")]}
    )
    reconciliation_provider = Provider(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "operation": "conflict",
                    "target_fact_id": "fact-1",
                    "expected_version": 3,
                    "reason": "Assistant history reports disagreement; it has no replacement authority.",
                }
            ]
        }
    )
    result = await run_source_pipeline(
        [source],
        [target()],
        selection_provider=selection_provider,
        reconciliation_provider=reconciliation_provider,
    )
    assert result.proposals == []
    assert result.reviews[0].operation == "conflict"
    assert result.reviews[0].source_role == "assistant"
    assert result.reviews[0].source_kind == "attributed_history"
    assert result.reviews[0].target_fact_id == "fact-1"
    assert result.reviews[0].expected_version == 3


@pytest.mark.asyncio
async def test_foreign_target_is_rejected_before_any_provider_call() -> None:
    source = event("source", "The marker is cedar.")
    foreign = target()
    foreign.metadata["scope_id"] = "other-project"
    selection_provider = Provider({"selections": []})
    reconciliation_provider = Provider({"decisions": []})
    with pytest.raises(ValueError, match="foreign scope"):
        await run_source_pipeline(
            [source],
            [foreign],
            selection_provider=selection_provider,
            reconciliation_provider=reconciliation_provider,
        )
    assert selection_provider.calls == 0
    assert reconciliation_provider.calls == 0


@pytest.mark.asyncio
async def test_original_event_mutation_during_classification_is_rejected() -> None:
    source = event("source", "The marker is cedar.")
    selection_provider = Provider({"selections": []})

    class MutatingProvider(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            source.content = "Changed while classifying."
            return await super().predict(prompt, **kwargs)

    with pytest.raises(ValueError, match="during classification"):
        await run_source_pipeline(
            [source],
            [],
            selection_provider=MutatingProvider(selection_provider.response),
            reconciliation_provider=Provider({"decisions": []}),
        )


@pytest.mark.asyncio
async def test_original_target_mutation_during_reconciliation_is_rejected() -> None:
    source = event("source", "The review window is Tuesday.")
    observed = target()
    selection_provider = Provider({"selections": [selection("source")]})

    class MutatingProvider(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            observed.content = "Changed while reconciling."
            return await super().predict(prompt, **kwargs)

    with pytest.raises(ValueError, match="during reconciliation"):
        await run_source_pipeline(
            [source],
            [observed],
            selection_provider=selection_provider,
            reconciliation_provider=MutatingProvider(
                {"decisions": [{"candidate_index": 0, "operation": "add"}]}
            ),
        )


@pytest.mark.asyncio
async def test_original_target_mutation_during_all_discarded_classification_is_rejected() -> None:
    source = event("source", "Which marker should we use?")
    observed = target()

    class MutatingProvider(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            observed.version += 1
            return await super().predict(prompt, **kwargs)

    with pytest.raises(ValueError, match="during classification"):
        await run_source_pipeline(
            [source],
            [observed],
            selection_provider=MutatingProvider({"selections": [selection("source", "question")]}),
            reconciliation_provider=Provider({"decisions": []}),
        )


@pytest.mark.asyncio
async def test_original_source_authority_mutation_during_reconciliation_is_rejected() -> None:
    source = event("source", "The review window is Tuesday.")

    class MutatingProvider(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            source.role = "assistant"
            return await super().predict(prompt, **kwargs)

    with pytest.raises(ValueError, match="during reconciliation"):
        await run_source_pipeline(
            [source],
            [],
            selection_provider=Provider({"selections": [selection("source")]}),
            reconciliation_provider=MutatingProvider(
                {"decisions": [{"candidate_index": 0, "operation": "add"}]}
            ),
        )
