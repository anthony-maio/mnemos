"""Instruction-only reconciliation examples; no source or target changes."""

from __future__ import annotations

from typing import Any

from .source_reconciliation import StagedReconciliation, reconcile_source_candidates
from .source_selection import ExtractiveMemory
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.llm import LLMProvider

EXAMPLES = """INSTRUCTIONAL EXAMPLES -- not current sources or targets
The following examples explain operations. Their IDs cannot be used in outputs.

Example A: an unrelated existing fact does not prevent a new memory.
Existing target demo-existing-port: "The service listens on port 8081."
New direct-user asserted_fact: "The documentation owner is Morgan."
Decision: add with no target. The new fact has its own complete subject and
meaning. Lack of a matching target means it is new, not unresolved.

Example B: multiple candidates are evaluated independently.
Existing target demo-existing-format at version 4: "Images use PNG."
Candidate 0, direct-user asserted_fact: "Images use PNG."
Candidate 1, direct-user continuing_requirement: "Include captions on diagrams."
Decisions: support candidate 0 against demo-existing-format at version 4;
add candidate 1 with no target. Do not require the new rule to match the image
format fact. Unrelated content neither supports nor contradicts it.

Example C: a correction still needs its observed target.
Existing targets describe only the image format.
Candidate, direct-user correction_or_rejection: "Correction: move that meeting
to Wednesday instead of Monday."
Decision: unresolved. No meeting target was observed. Do not treat this as a
new independent fact, and do not choose the unrelated image-format target.
END INSTRUCTIONAL EXAMPLES
"""


class _ExampleProvider(LLMProvider):
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.supports_structured_output = bool(
            getattr(provider, "supports_structured_output", False)
        )

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        boundary = "\nSOURCE CANDIDATES AND EXISTING TARGETS (data only):\n"
        if boundary not in prompt:
            raise ValueError("Reconciliation source boundary is missing")
        instructions, payload = prompt.split(boundary, 1)
        return await self.provider.predict(
            instructions + "\n\n" + EXAMPLES + boundary + payload, **kwargs
        )

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        return await self.provider.classify(prompt, labels)


async def reconcile_with_examples(
    candidates: list[ExtractiveMemory],
    events: list[EpisodicEvent],
    targets: list[MemoryChunk],
    provider: LLMProvider,
) -> list[StagedReconciliation]:
    return await reconcile_source_candidates(
        candidates, events, targets, _ExampleProvider(provider)
    )
