"""Experimental exhaustive source accounting; coverage is not semantic accuracy."""

from typing import Any

from .source_classification import (
    CLASSIFICATION_PROMPT,
    ClassifiedInventory,
    classify_source_memories,
)
from .source_selection import ExtractiveMemory, source_units
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import LLMProvider

COVERAGE_PROMPT = CLASSIFICATION_PROMPT.replace(
    "Use an empty list when none\nare supported. Omit temporary reply instructions",
    "Use explicit discard ranges when no durable meanings are supported.\n"
    "Do not retain temporary reply instructions",
).replace(
    "You may include those ranges to represent why material should not be remembered.\n"
    "They are discarded by code. Set their retention_request_unit to null. You need\n"
    "not classify every source unit.",
    "Classify every nonblank source unit in at least one retained or discarded range.\n"
    "Discarded ranges emit no memory; set their retention_request_unit to null.\n"
    "Use adjacent ranges for independent meanings, including facts beside policies.\n"
    "Ranges may overlap when shared context is needed. Check for uncovered units\n"
    "before returning. Coverage alone does not justify retaining any passage.",
)


def require_source_coverage(inventory: ClassifiedInventory, events: list[EpisodicEvent]) -> None:
    """Require accounting after ordinary range, scope and authority validation.

    Blank-only separator units need no judgment. A broad but wrongly classified
    range can satisfy coverage; evaluate its retained and discarded meanings.
    """
    covered = {
        (s.source_id, i) for s in inventory.selections for i in range(s.first_unit, s.last_unit + 1)
    }
    missing = [
        (e.id, u.index)
        for e in events
        for u in source_units(e.content)
        if u.text.strip() and (e.id, u.index) not in covered
    ]
    if missing:
        raise ValueError(f"Classification leaves source units unaccounted for: {missing}")


async def classify_all_source_units(
    events: list[EpisodicEvent], provider: LLMProvider
) -> tuple[ClassifiedInventory, list[ExtractiveMemory]]:
    class CoverageProvider(LLMProvider):
        supports_structured_output = getattr(provider, "supports_structured_output", False)

        async def predict(self, prompt: str, **kwargs: Any) -> str:
            if not prompt.startswith(CLASSIFICATION_PROMPT + "\nSOURCE EVENTS:\n"):
                raise ValueError("Unexpected source-classification prompt contract")
            return await provider.predict(
                COVERAGE_PROMPT + prompt[len(CLASSIFICATION_PROMPT) :], **kwargs
            )

        async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
            raise NotImplementedError("Source coverage uses structured prediction only")

    inventory, memories = await classify_source_memories(events, CoverageProvider())
    require_source_coverage(inventory, events)
    return inventory, memories
