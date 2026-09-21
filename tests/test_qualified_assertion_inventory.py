"""Qualification attachment and citation limits in the v2 diagnostic."""

from datetime import datetime, timezone
import json

import pytest

from benchmarks.local_memory.assertion_inventory import (
    Qualification,
    QualifiedAssertion,
    QualifiedInventory,
    inventory_qualified_assertions,
    retained_qualified_assertions,
    validate_qualified_inventory,
)
from mnemos.types import EpisodicEvent


def event():
    text = "The migration may require a schema bump. Please remember this uncertainty. Until review, keep it unconfirmed."
    return EpisodicEvent(
        id="source",
        event_fingerprint="source",
        role="user",
        content=text,
        normalized_content=text.lower(),
        scope="project",
        scope_id="project",
        source="manual",
        metadata={"user_profile_id": "profile", "project_id": "project"},
        occurred_at=datetime.now(timezone.utc),
    )


def assertion(**changes):
    fields = dict(
        source_id="source",
        quote="The migration may require a schema bump.",
        content="The migration may require a schema bump.",
        kind="explicitly_retained_uncertainty",
        retention_request_quote="Please remember this uncertainty.",
        qualifications=[
            Qualification(
                quote="Until review, keep it unconfirmed.",
                content="Until review, the schema bump remains unconfirmed.",
            )
        ],
    )
    fields.update(changes)
    return QualifiedAssertion(**fields)


def test_qualification_stays_attached_without_mutating_original():
    original = assertion()
    inventory = QualifiedInventory(assertions=[original])
    kept = retained_qualified_assertions(inventory, [event()])
    assert len(kept) == 1
    assert (
        kept[0].content
        == "The migration may require a schema bump. Until review, the schema bump remains unconfirmed."
    )
    assert original.content == "The migration may require a schema bump."
    assert kept[0].qualifications[0].quote in event().content


@pytest.mark.parametrize("quote", [None, "", " ", "Please store this forever."])
def test_retained_uncertainty_needs_source_request_span(quote):
    with pytest.raises(ValueError, match="request quote"):
        validate_qualified_inventory(
            QualifiedInventory(assertions=[assertion(retention_request_quote=quote)]), [event()]
        )


def test_other_kinds_cannot_fabricate_retention_request():
    with pytest.raises(ValueError, match="only to explicitly"):
        validate_qualified_inventory(
            QualifiedInventory(assertions=[assertion(kind="asserted_fact")]), [event()]
        )


def test_qualification_must_quote_same_source():
    item = assertion(
        qualifications=[
            Qualification(quote="External unrelated report", content="Unrelated condition")
        ]
    )
    with pytest.raises(ValueError, match="Qualification quote"):
        validate_qualified_inventory(QualifiedInventory(assertions=[item]), [event()])


def test_discarded_assertion_does_not_leak_its_qualifications():
    item = assertion(kind="unrequested_speculation", retention_request_quote=None)
    assert retained_qualified_assertions(QualifiedInventory(assertions=[item]), [event()]) == []


def test_request_quote_validity_does_not_establish_request_semantics():
    # A real span can still be semantically wrong. Keep this limit explicit.
    item = assertion(retention_request_quote="The migration may require a schema bump.")
    assert validate_qualified_inventory(QualifiedInventory(assertions=[item]), [event()])


def test_rendered_content_limit_includes_attached_qualifications():
    item = assertion(content="x" * 9990)
    with pytest.raises(ValueError):
        retained_qualified_assertions(QualifiedInventory(assertions=[item]), [event()])


def test_repeated_qualifications_rejected():
    item = assertion()
    item.qualifications *= 2
    with pytest.raises(ValueError, match="repeats a qualification"):
        validate_qualified_inventory(QualifiedInventory(assertions=[item]), [event()])


async def test_v2_schema_is_source_bound_and_contains_no_targets():
    class Provider:
        supports_structured_output = True

        async def predict(self, prompt, **kwargs):
            props = kwargs["format"]["$defs"]["QualifiedAssertion"]["properties"]
            assert props["source_id"]["enum"] == ["source"]
            assert "qualifications" in props and "retention_request_quote" in props
            assert "target_fact_id" not in props
            assert "EXISTING FACTS" not in prompt
            return json.dumps({"assertions": []})

    assert (await inventory_qualified_assertions([event()], Provider())).assertions == []
