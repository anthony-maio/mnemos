"""Add prompt-only instructional examples to the role-bound classifier.

The examples are demonstrations of classification decisions, not source events
or evidence.  Role binding, source snapshots, structured decoding, rendering,
and the write firewall remain owned by the existing classifier.
"""

from __future__ import annotations

from typing import Any

from .source_classification import ClassifiedInventory
from .source_role_binding import classify_source_memories_with_role_binding
from .source_selection import ExtractiveMemory
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import LLMProvider

INSTRUCTIONAL_EXAMPLES = """EXPERIMENTAL INSTRUCTIONAL EXAMPLES
These three examples are demonstrations only. They are not current source
evidence, must not be retained, and their fake IDs are not available sources.
Use the same exact contiguous-range and qualification rules for the real
SOURCE EVENTS below.

EXAMPLE 1 -- durable fact mixed with one-turn instructions
Fake source ID: demo-mixed-9f3
Synthetic numbered units:
  0: "The project marker is amber-lark."
  1: "Use a terse reply for this turn and do not call tools."
Expected classification:
  {"source_id":"demo-mixed-9f3","first_unit":0,"last_unit":0,
   "kind":"asserted_fact","retention_request_unit":null}
Select the durable project fact without the unrelated one-turn instruction.

EXAMPLE 2 -- possible plan, explicit non-approval, and an actual remember request
Fake source ID: demo-plan-a7q
Synthetic numbered units:
  0: "We might explore a harbor lantern review next season."
  1: "I have not approved that possible plan."
  2: "Please remember that it remains unapproved and uncertain."
Expected classification:
  {"source_id":"demo-plan-a7q","first_unit":0,"last_unit":2,
   "kind":"explicitly_retained_uncertainty","retention_request_unit":2}
Keep the possible and unapproved status because the user explicitly asks to
remember it. Without unit 2, mere uncertainty would not be a retention request.

EXAMPLE 3 -- continuing requirement without a remember request
Fake source ID: demo-rule-k2m
Synthetic numbered units:
  0: "For future reports, include the observed checksum."
Expected classification:
  {"source_id":"demo-rule-k2m","first_unit":0,"last_unit":0,
   "kind":"continuing_requirement","retention_request_unit":null}
An ordinary continuing requirement has a null retention_request_unit unless an
actual request to remember it is present.

END INSTRUCTIONAL EXAMPLES
"""


def _prompt_with_examples(prompt: str) -> str:
    marker = "\nSOURCE EVENTS:\n"
    if marker not in prompt:
        raise ValueError("Classifier prompt has no source-event boundary")
    instructions, source_payload = prompt.split(marker, 1)
    return instructions + "\n\n" + INSTRUCTIONAL_EXAMPLES + marker + source_payload


class _InstructionExampleProvider(LLMProvider):
    """Forward provider capabilities while changing only the classifier prompt."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self.supports_structured_output = bool(
            getattr(provider, "supports_structured_output", False)
        )
        self.supports_model_override = bool(getattr(provider, "supports_model_override", False))

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        return await self._provider.predict(_prompt_with_examples(prompt), **kwargs)

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        return await self._provider.classify(prompt, labels)


async def classify_source_memories_with_instruction_examples(
    events: list[EpisodicEvent], provider: LLMProvider
) -> tuple[ClassifiedInventory, list[ExtractiveMemory]]:
    """Run role-bound classification with prompt-only demonstrations."""

    return await classify_source_memories_with_role_binding(
        events, _InstructionExampleProvider(provider)
    )


__all__ = [
    "INSTRUCTIONAL_EXAMPLES",
    "classify_source_memories_with_instruction_examples",
]
