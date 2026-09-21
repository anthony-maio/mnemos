"""The v3 qualified-inventory contract relaxes only retention evidence syntax."""

from datetime import datetime, timezone
import json

import pytest

from benchmarks.local_memory.assertion_inventory import (
    Qualification,
    QualifiedAssertion,
    QualifiedInventory,
    inventory_qualified_assertions_v3,
    retained_qualified_assertions_v3,
    validate_qualified_inventory,
    validate_qualified_inventory_v3,
)
from mnemos.types import EpisodicEvent

SOURCE_TEXT = (
    "The migration may require a schema bump. Please remember this uncertainty. "
    "Until review, keep it unconfirmed."
)


def event(event_id="source", content=SOURCE_TEXT):
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=event_id,
        role="user",
        content=content,
        normalized_content=content.lower(),
        scope="project",
        scope_id="project",
        source="manual",
        metadata={"user_profile_id": "profile", "project_id": "project"},
        occurred_at=datetime.now(timezone.utc),
    )


def assertion_v3(**changes):
    fields = dict(
        source_id="source",
        quote="The migration may require a schema bump.",
        content="The migration may require a schema bump.",
        kind="asserted_fact",
        qualifications=[],
        retention_request_quote=None,
    )
    fields.update(changes)
    return QualifiedAssertion(**fields)


def test_v3_uncertainty_still_requires_source_bound_request_quote():
    with pytest.raises(ValueError, match="Retained uncertainty"):
        validate_qualified_inventory_v3(
            QualifiedInventory(
                assertions=[
                    assertion_v3(
                        kind="explicitly_retained_uncertainty",
                        retention_request_quote=None,
                    )
                ]
            ),
            [event()],
        )


@pytest.mark.parametrize("kind", ["asserted_fact", "temporary_instruction"])
def test_v3_allows_optional_source_bound_request_quote_without_changing_eligibility(kind):
    inventory = QualifiedInventory(
        assertions=[
            assertion_v3(
                kind=kind,
                retention_request_quote="Please remember this uncertainty.",
            )
        ]
    )
    validated = validate_qualified_inventory_v3(inventory, [event()])
    retained = retained_qualified_assertions_v3(validated, [event()])
    assert validated.assertions[0].retention_request_quote == "Please remember this uncertainty."
    assert (len(retained) == 1) is (kind == "asserted_fact")


@pytest.mark.parametrize("quote", ["", " ", "This quote is not in the source."])
def test_v3_rejects_blank_or_non_source_optional_request_quote(quote):
    with pytest.raises(ValueError, match="Retention request quote"):
        validate_qualified_inventory_v3(
            QualifiedInventory(assertions=[assertion_v3(retention_request_quote=quote)]),
            [event()],
        )


def test_v3_rejects_request_quote_from_another_source():
    other = event("other", "Please remember this other source request.")
    with pytest.raises(ValueError, match="Retention request quote"):
        validate_qualified_inventory_v3(
            QualifiedInventory(
                assertions=[
                    assertion_v3(
                        retention_request_quote="Please remember this other source request."
                    )
                ]
            ),
            [event(), other],
        )


def test_v2_keeps_rejecting_optional_request_quote_on_general_assertions():
    item = QualifiedAssertion(
        source_id="source",
        quote="The migration may require a schema bump.",
        content="The migration may require a schema bump.",
        kind="asserted_fact",
        qualifications=[],
        retention_request_quote="Please remember this uncertainty.",
    )
    with pytest.raises(ValueError, match="only to explicitly"):
        validate_qualified_inventory(QualifiedInventory(assertions=[item]), [event()])


def test_v3_preserves_qualification_firewall():
    item = assertion_v3(
        retention_request_quote="Please remember this uncertainty.",
        qualifications=[
            Qualification(
                quote="Until review, keep it unconfirmed.",
                content="api_key=supersecret",
            )
        ],
    )
    with pytest.raises(ValueError, match="Qualification violates"):
        retained_qualified_assertions_v3(QualifiedInventory(assertions=[item]), [event()])


async def test_v3_generation_reuses_source_binding_and_optional_request_rule():
    class Provider:
        supports_structured_output = True

        async def predict(self, prompt, **kwargs):
            props = kwargs["format"]["$defs"]["QualifiedAssertion"]["properties"]
            assert props["source_id"]["enum"] == ["source"]
            assert "retention_request_quote may be a real, nonblank" in prompt
            assert "For every other kind set retention_request_quote to null." not in prompt
            return json.dumps({"assertions": []})

    result = await inventory_qualified_assertions_v3([event()], Provider())
    assert result.assertions == []
