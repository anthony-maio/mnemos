"""The capacity path must retain authority and evidence through local inference."""

import asyncio
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmarks.local_memory.resident_support_path import capacity_retrieve, validate_outputs


def setup():
    source = dict(
        evidence_id="s1",
        fact_id="f1",
        source_event_id="e1",
        source_ref="ref",
        excerpt="Owned excerpt",
    )
    metadata = dict(
        user_profile_id="u",
        scope="project",
        scope_id="p",
        source="manual",
        assertion_status="asserted",
    )
    chunk = SimpleNamespace(id="f1", version=1, content="A retained claim", metadata=metadata)
    evidence = SimpleNamespace(**source, model_dump=lambda **kw: deepcopy(source))
    event_data = dict(
        id="e1", content="Owned excerpt", metadata=deepcopy(metadata), status="consolidated"
    )
    event = SimpleNamespace(
        metadata=event_data["metadata"], model_dump=lambda **kw: deepcopy(event_data)
    )
    store = SimpleNamespace(
        read_snapshot=nullcontext,
        get=lambda _: chunk,
        list_fact_evidence=lambda _: [evidence],
        get_episodic_event=lambda _: event,
    )
    fact = dict(
        id="f1",
        version=1,
        content=chunk.content,
        scope="project",
        scope_id="p",
        assertion_status="asserted",
        sources=[{**source, "original_source_status": "retained_locally"}],
    )

    async def retrieve(_):
        return dict(facts=[deepcopy(fact)], context="Already rendered")

    state = dict(authorized=True)

    def validate(_):
        if not state["authorized"]:
            raise ValueError("revoked")

    memory = SimpleNamespace(
        engine=SimpleNamespace(store=store),
        retrieve=retrieve,
        _refresh_registry=lambda: None,
        registry=SimpleNamespace(validate_context=validate),
        _original_source_status=lambda *a: "retained_locally",
    )
    request = SimpleNamespace(
        query="What is retained?",
        context=SimpleNamespace(user_profile_id="u", project_id="p", workspace_id=None),
        allowed_scopes=["project"],
    )
    return memory, request, fact, chunk, source, event_data, state


def test_negative_random_outputs_still_return_all_facts_and_sources():
    memory, request, fact, *_ = setup()

    async def infer(cohort):
        assert cohort["candidates"][0]["sources"][0]["excerpt"] == "Owned excerpt"
        return dict(target_ids=["f1"], values=[-999.0], tokens=123)

    result = asyncio.run(capacity_retrieve(memory, request, infer, ["f1"]))
    assert result["facts"] == [fact]
    assert result["context"] == "Already rendered"
    assert result["capacity"]["semantic_decisions"] is False


@pytest.mark.parametrize(
    "mutation", ["version", "content", "evidence", "event", "authority", "revoke"]
)
def test_changes_while_worker_runs_fail_whole_response(mutation):
    memory, request, _, chunk, source, event, state = setup()

    async def infer(_):
        if mutation == "version":
            chunk.version += 1
        elif mutation == "content":
            chunk.content = "Changed"
        elif mutation == "evidence":
            source["excerpt"] = "Changed"
        elif mutation == "event":
            event["status"] = "failed"
        elif mutation == "authority":
            chunk.metadata["user_profile_id"] = "other"
        else:
            state["authorized"] = False
        return dict(target_ids=["f1"], values=[-1.0], tokens=123)

    with pytest.raises(ValueError):
        asyncio.run(capacity_retrieve(memory, request, infer, ["f1"]))


@pytest.mark.parametrize(
    "response",
    [
        dict(target_ids=["other"], values=[1.0], tokens=1),
        dict(target_ids=["f1"], values=[float("nan")], tokens=1),
        dict(target_ids=["f1"], values=[], tokens=1),
    ],
)
def test_worker_identity_shape_and_finiteness_are_required(response):
    with pytest.raises(ValueError):
        validate_outputs(response, ["f1"])


def test_missing_owned_source_cannot_be_omitted():
    memory, request, *_ = setup()
    memory.engine.store.get_episodic_event = lambda _: None

    async def infer(_):
        pytest.fail("Missing source must reject before worker invocation")

    with pytest.raises(ValueError):
        asyncio.run(capacity_retrieve(memory, request, infer, ["f1"]))


def test_unavailable_worker_is_a_failure_without_returning_unchecked_facts():
    memory, request, *_ = setup()

    async def infer(_):
        raise ConnectionError("Resident worker unavailable")

    with pytest.raises(ConnectionError):
        asyncio.run(capacity_retrieve(memory, request, infer, ["f1"]))


def test_empty_cohort_still_revalidates_authorization_after_worker():
    memory, request, *_, state = setup()

    async def retrieve(_):
        return dict(facts=[], context="")

    memory.retrieve = retrieve

    async def infer(cohort):
        assert cohort["candidates"] == []
        state["authorized"] = False
        return dict(target_ids=[], values=[], tokens=43)

    with pytest.raises(ValueError, match="revoked"):
        asyncio.run(capacity_retrieve(memory, request, infer, []))
