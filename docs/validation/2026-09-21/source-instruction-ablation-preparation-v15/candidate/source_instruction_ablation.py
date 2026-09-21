"""Controlled ablation of the v14 uncertainty instructional example.

This removes exactly the possible-plan/uncertainty demonstration from the
committed instructional-example intervention. It is an experiment, not an
accepted classifier change; role binding and all final guards remain owned by
the v14 wrapper.
"""

from __future__ import annotations

from typing import Any

from .source_classification import ClassifiedInventory
from .source_instruction_examples import (
    classify_source_memories_with_instruction_examples,
)
from .source_selection import ExtractiveMemory
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import LLMProvider

_SOURCE_MARKER = "\nSOURCE EVENTS:\n"
_EXAMPLE_TWO_START = (
    "EXAMPLE 2 -- possible plan, explicit non-approval, and an actual remember request\n"
)
_EXAMPLE_THREE_START = "EXAMPLE 3 -- continuing requirement without a remember request\n"


def _remove_uncertainty_example(prompt: str) -> str:
    if _SOURCE_MARKER not in prompt:
        raise ValueError("Classifier prompt has no source-event boundary")
    instructions, source_payload = prompt.split(_SOURCE_MARKER, 1)
    start = instructions.find(_EXAMPLE_TWO_START)
    end = instructions.find(_EXAMPLE_THREE_START, start + len(_EXAMPLE_TWO_START))
    if start < 0 or end < 0:
        raise ValueError("Classifier prompt has no v14 uncertainty example")
    instructions = instructions[:start] + instructions[end:]
    instructions = instructions.replace("These three examples", "These two examples", 1)
    return instructions + _SOURCE_MARKER + source_payload


class _AblationProvider(LLMProvider):
    """Forward provider behavior while removing one prompt example only."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self.supports_structured_output = bool(
            getattr(provider, "supports_structured_output", False)
        )
        self.supports_model_override = bool(getattr(provider, "supports_model_override", False))

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        return await self._provider.predict(_remove_uncertainty_example(prompt), **kwargs)

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        return await self._provider.classify(prompt, labels)


async def classify_source_memories_without_uncertainty_example(
    events: list[EpisodicEvent], provider: LLMProvider
) -> tuple[ClassifiedInventory, list[ExtractiveMemory]]:
    """Run v14 classification with only its uncertainty example ablated."""

    return await classify_source_memories_with_instruction_examples(
        events, _AblationProvider(provider)
    )


__all__ = ["classify_source_memories_without_uncertainty_example"]
