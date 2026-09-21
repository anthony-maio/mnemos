"""The uncertainty-example ablation changes only one prompt block."""

from datetime import datetime, timezone
import json

import pytest

from benchmarks.local_memory.source_classification import ClassifiedInventory
from benchmarks.local_memory.source_instruction_ablation import (
    classify_source_memories_without_uncertainty_example,
)
from benchmarks.local_memory.source_instruction_examples import (
    classify_source_memories_with_instruction_examples,
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


@pytest.mark.asyncio
async def test_ablation_changes_only_uncertainty_example_and_keeps_schema_payload() -> None:
    event = source("The marker is cedar. EXAMPLE 2 -- source text only.")
    response = {"selections": []}
    v14 = Provider(response)
    ablated = Provider(response)
    await classify_source_memories_with_instruction_examples([event], v14)
    await classify_source_memories_without_uncertainty_example([event], ablated)

    v14_prompt, v14_kwargs = v14.calls[0]
    ablated_prompt, ablated_kwargs = ablated.calls[0]
    marker = "\nSOURCE EVENTS:\n"
    v14_instructions, v14_payload = v14_prompt.split(marker, 1)
    ablated_instructions, ablated_payload = ablated_prompt.split(marker, 1)
    start = v14_instructions.index(
        "EXAMPLE 2 -- possible plan, explicit non-approval, and an actual remember request\n"
    )
    end = v14_instructions.index(
        "EXAMPLE 3 -- continuing requirement without a remember request\n", start
    )
    expected_instructions = (v14_instructions[:start] + v14_instructions[end:]).replace(
        "These three examples", "These two examples", 1
    )
    assert ablated_prompt == expected_instructions + marker + v14_payload
    assert ablated_payload == v14_payload
    assert ablated_kwargs["format"] == v14_kwargs["format"]
    assert "These three examples" in v14_instructions
    assert "These two examples" in ablated_instructions
    assert "demo-mixed-9f3" in ablated_instructions
    assert "demo-rule-k2m" in ablated_instructions
    assert "demo-plan-a7q" not in ablated_instructions
    assert "possible plan, explicit non-approval" not in ablated_instructions
    assert "EXAMPLE 2 -- source text only." in ablated_payload


@pytest.mark.asyncio
async def test_fake_example_id_still_hits_v14_runtime_guards() -> None:
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
        await classify_source_memories_without_uncertainty_example([source()], provider)


@pytest.mark.asyncio
async def test_snapshot_guard_and_unstructured_final_guard_remain_active() -> None:
    event = source(role="assistant")

    class MutatingProvider(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            self.calls.append((prompt, dict(kwargs)))
            event.content = "Changed while the classifier was running."
            return json.dumps({"selections": []})

    with pytest.raises(ValueError, match="changed during classification"):
        await classify_source_memories_without_uncertainty_example([event], MutatingProvider({}))

    unstructured = Provider(
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
    unstructured.supports_structured_output = False
    with pytest.raises(ValueError, match="Non-user"):
        await classify_source_memories_without_uncertainty_example(
            [source(role="assistant")], unstructured
        )
    assert unstructured.calls[0][1] == {}
    assert ClassifiedInventory(selections=[]).selections == []
