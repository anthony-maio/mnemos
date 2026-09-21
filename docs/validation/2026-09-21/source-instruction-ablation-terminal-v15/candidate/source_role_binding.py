"""Bind source roles into the source-classification decoder contract.

This module is a mechanical authority constraint for structured decoding. It
does not determine whether a passage is durable, true, or semantically well
classified. The existing classifier, renderer, firewall, and snapshot guard
remain authoritative after decoding.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .assertion_inventory import Inventory, RETAINED_KINDS, validate_inventory
from .source_classification import (
    DISCARD_KINDS,
    ClassifiedInventory,
    classify_source_memories,
)
from .source_selection import ExtractiveMemory
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import LLMProvider

USER_SOURCE_KINDS = frozenset((*RETAINED_KINDS, *DISCARD_KINDS))
NON_USER_SOURCE_KINDS = frozenset(("attributed_history", *DISCARD_KINDS))


def build_source_role_binding_schema(events: list[EpisodicEvent]) -> dict[str, Any]:
    """Return a source-specific schema whose kind enum follows each event role.

    Each selection alternative binds one source ID and its permitted kinds. A
    user source can use every existing retained or discard kind. Every other
    source can retain attributed history or an explicit discard classification,
    but cannot emit a direct fact, preference, requirement, uncertainty, or
    correction. Runtime rendering still rechecks these rules after decoding.
    """

    snapshot = [event.model_copy(deep=True) for event in events]
    validate_inventory(Inventory(assertions=[]), snapshot)
    schema = deepcopy(ClassifiedInventory.model_json_schema())
    definitions = schema.get("$defs")
    selection_schema = (
        definitions.get("ClassifiedSelection") if isinstance(definitions, dict) else None
    )
    if not isinstance(selection_schema, dict):
        raise ValueError("Classification schema has no selection properties")
    branches: list[dict[str, Any]] = []
    for event in snapshot:
        branch = deepcopy(selection_schema)
        branch_properties = branch.get("properties")
        if not isinstance(branch_properties, dict):
            raise ValueError("Classification schema has an invalid selection branch")
        allowed_kinds = USER_SOURCE_KINDS if event.role == "user" else NON_USER_SOURCE_KINDS
        branch_properties["source_id"] = {"type": "string", "enum": [event.id]}
        branch_properties["kind"] = {
            "type": "string",
            "enum": sorted(allowed_kinds),
        }
        branches.append(branch)
    if not branches:
        raise ValueError("Classification schema cannot be bound without source events")
    selections = schema.get("properties", {}).get("selections")
    if not isinstance(selections, dict):
        raise ValueError("Classification schema has no selections property")
    selections["items"] = {"oneOf": branches}
    return schema


class _RoleBoundProvider(LLMProvider):
    """Forward the original prompt while replacing only structured schema."""

    def __init__(self, provider: LLMProvider, schema: dict[str, Any]) -> None:
        self._provider = provider
        self._schema = deepcopy(schema)
        self.supports_structured_output = bool(
            getattr(provider, "supports_structured_output", False)
        )
        self.supports_model_override = bool(getattr(provider, "supports_model_override", False))

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        forwarded = dict(kwargs)
        if self.supports_structured_output:
            forwarded["format"] = deepcopy(self._schema)
        return await self._provider.predict(prompt, **forwarded)

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        return await self._provider.classify(prompt, labels)


async def classify_source_memories_with_role_binding(
    events: list[EpisodicEvent], provider: LLMProvider
) -> tuple[ClassifiedInventory, list[ExtractiveMemory]]:
    """Classify with role-bound structured decoding and existing final guards.

    The classifier prompt is unchanged. This wrapper only supplies a deep-copied
    role-bound schema to providers that advertise structured output. Providers
    without that capability receive the original prompt and no format option;
    their response still passes through the existing Pydantic and renderer
    validation in :func:`classify_source_memories`.
    """

    schema = build_source_role_binding_schema(events)
    bound_provider = _RoleBoundProvider(provider, schema)
    return await classify_source_memories(events, bound_provider)


# Short aliases keep the experimental wrapper easy to discover without
# changing the established classifier name or its prompt contract.
classify_source_memories_role_bound = classify_source_memories_with_role_binding


__all__ = [
    "NON_USER_SOURCE_KINDS",
    "USER_SOURCE_KINDS",
    "build_source_role_binding_schema",
    "classify_source_memories_role_bound",
    "classify_source_memories_with_role_binding",
]
