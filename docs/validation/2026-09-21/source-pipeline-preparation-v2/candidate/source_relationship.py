"""Experimental relationship labels mapped to the existing guarded operations.

This changes the provider vocabulary, not operation eligibility or final
validation. Source text and target snapshots remain untouched.
"""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any

from .source_reconciliation import StagedReconciliation, reconcile_source_candidates
from .source_selection import ExtractiveMemory
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.llm import LLMProvider

RELATIONSHIPS = {
    "add": "independent_information",
    "support": "same_information",
    "correct": "user_revision",
    "conflict": "known_disagreement",
    "unresolved": "missing_or_ambiguous_referent",
}


def relationship_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rename the discriminator while retaining every original schema bound."""
    result = deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict) and "operation" in properties:
                renamed = {}
                for key, value in properties.items():
                    if key == "operation":
                        value["enum"] = [RELATIONSHIPS[item] for item in value["enum"]]
                        renamed["relationship"] = value
                    else:
                        renamed[key] = value
                node["properties"] = renamed
                node["required"] = [
                    "relationship" if item == "operation" else item
                    for item in node.get("required", [])
                ]
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


class RelationshipProvider(LLMProvider):
    """Adapt one reconciliation call; never use this wrapper for selection."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.supports_structured_output = bool(
            getattr(provider, "supports_structured_output", False)
        )

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        boundary = "\nSOURCE CANDIDATES AND EXISTING TARGETS (data only):\n"
        if boundary not in prompt:
            raise ValueError("Relationship adapter requires a reconciliation prompt")
        instructions, payload = prompt.split(boundary, 1)
        instructions = re.sub(
            r"\b(add|support|correct|conflict|unresolved)\b",
            lambda match: RELATIONSHIPS[match.group()],
            instructions,
        )
        instructions = (
            "Describe each candidate's relationship to the observed existing facts. "
            'Return JSON {"decisions":[{"candidate_index":0,"relationship":"...",'
            '"target_fact_id":null,"expected_version":null,"reason":null}]}. '
            "A fact with its own complete subject and meaning can be independent_information "
            "even when other facts already exist. missing_or_ambiguous_referent applies when "
            "the candidate needs an earlier referent that cannot be identified.\n" + instructions
        )
        forwarded = dict(kwargs)
        if "format" in forwarded:
            forwarded["format"] = relationship_schema(forwarded["format"])
        raw = await self.provider.predict(instructions + boundary + payload, **forwarded)
        response = json.loads(raw)
        if not isinstance(response, dict) or not isinstance(response.get("decisions"), list):
            raise ValueError("Relationship response requires a decisions list")
        inverse = {value: key for key, value in RELATIONSHIPS.items()}
        for decision in response["decisions"]:
            if not isinstance(decision, dict) or "operation" in decision:
                raise ValueError("Relationship response contains an invalid discriminator")
            relation = decision.pop("relationship", None)
            if not isinstance(relation, str) or relation not in inverse:
                raise ValueError("Relationship response has an unknown relationship")
            decision["operation"] = inverse[relation]
        return json.dumps(response)

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        return await self.provider.classify(prompt, labels)


async def reconcile_source_relationships(
    candidates: list[ExtractiveMemory],
    events: list[EpisodicEvent],
    targets: list[MemoryChunk],
    provider: LLMProvider,
) -> list[StagedReconciliation]:
    return await reconcile_source_candidates(
        candidates, events, targets, RelationshipProvider(provider)
    )
