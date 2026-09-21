"""Instructional examples change only the prompt, not classifier safeguards."""

from datetime import datetime, timezone
import json

import pytest

from benchmarks.local_memory.source_classification import ClassifiedInventory
from benchmarks.local_memory.source_instruction_examples import (
    INSTRUCTIONAL_EXAMPLES,
    classify_source_memories_with_instruction_examples,
)
from benchmarks.local_memory.source_role_binding import (
    classify_source_memories_with_role_binding,
)
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import LLMProvider


def source(content: str = "The marker is cedar.", role: str = "user") -> EpisodicEvent:
    return EpisodicEvent(
        id="real-source",
        event_fingerprint="fingerprint-real-source",
        role=role,
        content=content,
        normalized_content=content.lower(),
        scope="project",
        scope_id="project",
        source="manual",
        metadata={"user_profile_id": "profile", "project_id": "project"},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class Provider(LLMProvider):
    supports_structured_output = True

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def predict(self, prompt: str, **kwargs: object) -> str:
        self.calls.append((prompt, dict(kwargs)))
        return json.dumps(self.response)

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        _ = prompt, labels
        return {}


def test_examples_are_separate_and_use_fake_ids() -> None:
    assert "END INSTRUCTIONAL EXAMPLES" in INSTRUCTIONAL_EXAMPLES
    assert "demo-mixed-9f3" in INSTRUCTIONAL_EXAMPLES
    assert "demo-plan-a7q" in INSTRUCTIONAL_EXAMPLES
    assert "demo-rule-k2m" in INSTRUCTIONAL_EXAMPLES
    assert "not current source" in INSTRUCTIONAL_EXAMPLES


@pytest.mark.asyncio
async def test_prompt_examples_preserve_exact_real_source_payload_and_schema() -> None:
    event = source("The marker is cedar. Use lowercase for this reply only.")
    provider = Provider({"selections": []})
    inventory, memories = await classify_source_memories_with_instruction_examples(
        [event], provider
    )
    baseline_provider = Provider({"selections": []})
    await classify_source_memories_with_role_binding([event], baseline_provider)
    assert inventory == ClassifiedInventory(selections=[])
    assert memories == []
    prompt, kwargs = provider.calls[0]
    assert INSTRUCTIONAL_EXAMPLES in prompt
    assert prompt.index("END INSTRUCTIONAL EXAMPLES") < prompt.index("SOURCE EVENTS:")
    marker = "\nSOURCE EVENTS:\n"
    assert prompt.split(marker, 1)[1] == baseline_provider.calls[0][0].split(marker, 1)[1]
    assert kwargs["format"] == baseline_provider.calls[0][1]["format"]


@pytest.mark.asyncio
async def test_fake_example_id_is_rejected_by_runtime_guards() -> None:
    provider = Provider(
        {
            "selections": [
                {
                    "source_id": "demo-mixed-9f3",
                    "first_unit": 0,
                    "last_unit": 0,
                    "kind": "asserted_fact",
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="unavailable source"):
        await classify_source_memories_with_instruction_examples([source()], provider)


@pytest.mark.asyncio
async def test_original_snapshot_guard_remains_active() -> None:
    event = source()

    class MutatingProvider(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            self.calls.append((prompt, dict(kwargs)))
            event.content = "Changed while the classifier was running."
            return json.dumps({"selections": []})

    with pytest.raises(ValueError, match="changed during classification"):
        await classify_source_memories_with_instruction_examples([event], MutatingProvider({}))


@pytest.mark.asyncio
async def test_unstructured_provider_still_gets_final_runtime_guards() -> None:
    event = source(role="assistant")
    provider = Provider(
        {
            "selections": [
                {
                    "source_id": "real-source",
                    "first_unit": 0,
                    "last_unit": 0,
                    "kind": "asserted_fact",
                }
            ]
        }
    )
    provider.supports_structured_output = False
    with pytest.raises(ValueError, match="Non-user"):
        await classify_source_memories_with_instruction_examples([event], provider)
    assert provider.calls[0][1] == {}
