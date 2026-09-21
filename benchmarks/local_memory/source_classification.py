"""Experimental source selection with explicit non-retained classifications.

Labels are fallible model judgments. Discarded selections remain in the returned
inventory for audit, but never become memory excerpts or durable writes.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .assertion_inventory import Inventory, validate_inventory
from .source_selection import (
    ExtractiveMemory,
    Kind,
    SELECTION_PROMPT,
    Selection,
    SelectionInventory,
    render_selections,
    source_units,
)
from mnemos.memory_safety import MemoryWriteFirewall
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import LLMProvider

DiscardKind = Literal[
    "question",
    "temporary_instruction",
    "quoted_material",
    "unrequested_speculation",
    "unaccepted_suggestion",
]
DISCARD_KINDS = frozenset(
    {
        "question",
        "temporary_instruction",
        "quoted_material",
        "unrequested_speculation",
        "unaccepted_suggestion",
    }
)


class ClassifiedSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1)
    first_unit: int = Field(ge=0, strict=True)
    last_unit: int = Field(ge=0, strict=True)
    kind: Kind | DiscardKind
    retention_request_unit: int | None = Field(default=None, ge=0, strict=True)
    revises_existing_requirement: bool = Field(default=False, strict=True)


class ClassifiedInventory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selections: list[ClassifiedSelection]


def render_classifications(
    inventory: ClassifiedInventory, events: list[EpisodicEvent]
) -> list[ExtractiveMemory]:
    """Validate all ranges, then render only retained kinds with original guards.

    Overlap does not imply cancellation: a surrounding quote classification must
    not silently erase an independently selected manuscript requirement. Exact
    duplicate ranges, including contradictory keep/discard labels, are rejected.
    """
    validate_inventory(Inventory(assertions=[]), events)
    sources = {e.id: e for e in events}
    seen = set()
    retained = []
    firewall = MemoryWriteFirewall()
    for item in inventory.selections:
        event = sources.get(item.source_id)
        if event is None:
            raise ValueError("Classification cites unavailable source")
        units = source_units(event.content)
        if item.first_unit > item.last_unit or item.last_unit >= len(units):
            raise ValueError("Classification range is outside its source or reversed")
        key = (event.id, item.first_unit, item.last_unit)
        if key in seen:
            raise ValueError("Classification repeats a source range")
        seen.add(key)
        content = event.content[units[item.first_unit].start : units[item.last_unit].end]
        if not content.strip():
            raise ValueError("Classification must contain nonblank source text")
        safe = firewall.apply(content)
        if not safe.allowed or safe.content != content:
            raise ValueError("Classified source range violates write policy")
        if item.kind in DISCARD_KINDS:
            if item.retention_request_unit is not None:
                raise ValueError("Discarded classification cannot cite a retention request")
            if item.revises_existing_requirement:
                raise ValueError("Discarded classification cannot revise a requirement")
        else:
            retained.append(Selection.model_validate(item.model_dump()))
    return render_selections(SelectionInventory(selections=retained), events)


CLASSIFICATION_PROMPT = SELECTION_PROMPT + """

The output schema additionally permits explicit non-retained classifications:
- question: an inquiry or options comparison, not an asserted preference.
- temporary_instruction: a one-turn tool, reply or formatting instruction.
- quoted_material: inspection-only quotations, fictional dialogue or fixtures
  that are not independently asserted facts or requested manuscript requirements.
- unrequested_speculation: hypothetical or uncertain material without an actual
  request to remember it.
- unaccepted_suggestion: a proposal that has not been accepted as a project rule.

You may include those ranges to represent why material should not be remembered.
They are discarded by code. Set their retention_request_unit to null. You need
not classify every source unit. Still select every independently durable meaning
in a mixed message, retaining its qualifications. Do not label a real historical
report as quoted_material merely because it attributes a past result. Do not
discard an explicit rejection, a genuine manuscript requirement, or an explicitly
requested unresolved consideration just because it discusses a possible plan.

Use continuing_requirement for a complete standalone prospective negative rule,
including a user's direct statement that a feature, practice or option will not
be kept in the project. Preserve the exact negation even when no earlier fact is
available. Use correction_or_rejection only for a correction, supersession or
rejection that refers to an existing assertion or prior project state and may
need target reconciliation. Do not relabel a standalone negative requirement as
correction_or_rejection merely because it contains words such as "not" or
"won't", and do not invent a missing target or replacement.
When a complete prospective replacement requirement is explicitly prefaced by
rejecting or superseding an earlier rule, keep it as continuing_requirement and
set revises_existing_requirement to true. A referential rejection without a
complete new requirement stays correction_or_rejection. Set the flag to false
for all other kinds and for discarded classifications.
"""


async def classify_source_memories(
    events: list[EpisodicEvent], provider: LLMProvider
) -> tuple[ClassifiedInventory, list[ExtractiveMemory]]:
    snapshot = [e.model_copy(deep=True) for e in events]
    validate_inventory(Inventory(assertions=[]), snapshot)
    sources = [
        {
            "id": e.id,
            "role": e.role,
            "occurred_at": e.occurred_at.isoformat(),
            "units": [{"index": u.index, "text": u.text} for u in source_units(e.content)],
        }
        for e in snapshot
    ]
    schema = ClassifiedInventory.model_json_schema()
    schema["$defs"]["ClassifiedSelection"]["properties"]["source_id"] = {
        "type": "string",
        "enum": sorted(e.id for e in snapshot),
    }
    kwargs: dict[str, Any] = {}
    if getattr(provider, "supports_structured_output", False):
        kwargs["format"] = schema
    raw = await provider.predict(
        CLASSIFICATION_PROMPT + "\nSOURCE EVENTS:\n" + json.dumps(sources), **kwargs
    )
    if [e.model_dump(mode="json") for e in events] != [e.model_dump(mode="json") for e in snapshot]:
        raise ValueError("Source events changed during classification")
    inventory = ClassifiedInventory.model_validate_json(raw)
    return inventory, render_classifications(inventory, snapshot)
