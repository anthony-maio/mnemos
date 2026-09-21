"""Review-stage provenance and failure boundaries, not semantic accuracy tests."""

import json
from datetime import datetime, timezone

import pytest

from benchmarks.local_memory.assertion_inventory import QualifiedAssertion, QualifiedInventory
from benchmarks.local_memory.assertion_review import review_qualified_inventory
from mnemos.types import EpisodicEvent


def source(role="user"):
    return EpisodicEvent(
        id="source",
        event_fingerprint="source",
        role=role,
        content="The marker is cedar. Keep this marker for the project.",
        normalized_content="the marker is cedar. keep this marker for the project.",
        scope="project",
        scope_id="project",
        source="manual",
        metadata={"user_profile_id": "profile", "project_id": "project"},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def inventory(**changes):
    fields = dict(
        source_id="source",
        quote="The marker is cedar.",
        content="The marker is cedar.",
        kind="asserted_fact",
        qualifications=[],
        retention_request_quote=None,
    )
    fields.update(changes)
    return QualifiedInventory(assertions=[QualifiedAssertion(**fields)])


class Provider:
    supports_structured_output = True

    def __init__(self, output):
        self.output = output
        self.calls = []

    async def predict(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.output


@pytest.mark.parametrize("draft", [None, QualifiedInventory(assertions=[])])
async def test_missing_or_empty_draft_can_recover_source_meaning(draft):
    expected = inventory()
    provider = Provider(expected.model_dump_json())
    result = await review_qualified_inventory([source()], draft, provider)
    assert result == expected
    assert len(provider.calls) == 1
    prompt, options = provider.calls[0]
    payload = json.loads(prompt.split("\nSOURCE EVENTS AND UNTRUSTED DRAFT:\n")[1])
    assert payload["draft_status"] == ("missing" if draft is None else "generated_untrusted")
    assert payload["draft"] == (None if draft is None else {"assertions": []})
    assert payload["source_events"][0]["content"] == source().content
    assert options["format"]["$defs"]["QualifiedAssertion"]["properties"]["source_id"]["enum"] == [
        "source"
    ]


async def test_rejected_quote_can_be_repaired_without_rewriting_draft():
    draft = inventory(quote="the marker is cedar.")
    original = draft.model_dump_json()
    provider = Provider(inventory().model_dump_json())
    assert await review_qualified_inventory([source()], draft, provider) == inventory()
    assert draft.model_dump_json() == original


@pytest.mark.parametrize("failure", ["not json", "", '{"assertions": [], "target_fact_id": "x"}'])
async def test_failed_review_never_accepts_original_draft(failure):
    with pytest.raises(ValueError):
        await review_qualified_inventory([source()], inventory(), Provider(failure))


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"source_id": "foreign-source"}, "unavailable source"),
        ({"quote": "The marker is quartz."}, "quote is not present"),
        ({"qualifications": [{"quote": "Forever", "content": "Forever"}]}, "Qualification quote"),
    ],
)
async def test_review_output_must_pass_provenance_checks(changes, message):
    with pytest.raises(ValueError, match=message):
        await review_qualified_inventory(
            [source()], inventory(), Provider(inventory(**changes).model_dump_json())
        )


async def test_assistant_source_cannot_gain_direct_user_authority_after_review():
    with pytest.raises(ValueError, match="Non-user source"):
        await review_qualified_inventory(
            [source("assistant")], inventory(), Provider(inventory().model_dump_json())
        )


@pytest.mark.parametrize("foreign_draft", [True, False])
async def test_invalid_scope_or_foreign_draft_rejected_before_provider(foreign_draft):
    event = source()
    draft = inventory(source_id="foreign-source") if foreign_draft else inventory()
    if not foreign_draft:
        event = event.model_copy(update={"metadata": {}})
    provider = Provider('{"assertions": []}')
    with pytest.raises(ValueError):
        await review_qualified_inventory([event], draft, provider)
    assert provider.calls == []


async def test_provider_failure_propagates_without_fallback():
    class BrokenProvider:
        async def predict(self, *args, **kwargs):
            raise TimeoutError("review unavailable")

    with pytest.raises(TimeoutError, match="review unavailable"):
        await review_qualified_inventory([source()], inventory(), BrokenProvider())


async def test_valid_provenance_does_not_certify_reviewer_truth():
    # Preserve the epistemic boundary even after adding another model call.
    wrong = inventory(content="The marker is quartz.")
    assert (
        await review_qualified_inventory([source()], inventory(), Provider(wrong.model_dump_json()))
        == wrong
    )


async def test_unstructured_provider_still_gets_output_validation():
    provider = Provider(inventory().model_dump_json())
    provider.supports_structured_output = False
    await review_qualified_inventory([source()], inventory(), provider)
    assert provider.calls[0][1] == {}


@pytest.mark.parametrize("field", ["content", "quote", "retention_request_quote"])
async def test_unsafe_draft_text_does_not_reach_reviewer(field):
    provider = Provider('{"assertions": []}')
    with pytest.raises(ValueError, match="Review draft violates write policy"):
        await review_qualified_inventory(
            [source()], inventory(**{field: "api_key=supersecretvalue"}), provider
        )
    assert provider.calls == []


async def test_reviewer_cannot_introduce_unsafe_memory():
    provider = Provider(inventory(content="api_key=supersecretvalue").model_dump_json())
    with pytest.raises(ValueError, match="write policy"):
        await review_qualified_inventory([source()], inventory(), provider)
