"""The composed path must retain real retrieval and current-evidence boundaries."""

from datetime import date

import pytest

from benchmarks.local_memory.answer_support_comparison import _seed_case, grade_selection
from benchmarks.local_memory.retrieval_support import retrieve_with_support
from mnemos.contracts import RetrievalRequest
from mnemos.utils.reliability import MnemosTransientError
from tests.test_answer_support_comparison import FixedEmbedding
from tests.test_support_intersection import Selector


@pytest.fixture
def prepared(tmp_path):
    service, context = _seed_case(
        tmp_path / "data",
        {
            "id": "boundary",
            "facts": [
                {"id": "one", "content": "Release checks live in docs/release.md."},
                {"id": "two", "content": "The release marker is pine-529."},
                {
                    "id": "foreign",
                    "content": "The other project's checks live in SECRET.md.",
                    "scope": "foreign",
                },
            ],
        },
        FixedEmbedding(),
    )
    request = RetrievalRequest(
        query="release checks", context=context, profile="baseline", max_context_tokens=2048
    )
    yield service, request
    service.engine.store.close()


async def run(prepared, first=None, second=None):
    service, request = prepared
    return await retrieve_with_support(
        service,
        request,
        extractive=first or Selector(("one",)),
        contrast=second or Selector(("one",)),
        reference_date=date(2026, 11, 6),
    )


@pytest.mark.asyncio
async def test_selectors_receive_only_actual_candidates_and_output_is_source_backed(prepared):
    first, second = Selector(("one",)), Selector(("one",))
    result = await run(prepared, first, second)
    assert result.status == "assessed" and result.selected_ids == ("one",)
    assert result.candidate_ids == tuple(row["id"] for row in result.baseline["facts"])
    assert tuple(row.fact_id for row in first.calls[0].candidates) == result.candidate_ids
    assert first.calls[0] is second.calls[0] and "foreign" not in result.candidate_ids
    assert "SECRET" not in result.context and "pine-529" not in result.context
    assert result.context == "[one version 1] Release checks live in docs/release.md.\n"
    assert result.context_bytes == len(result.context.encode()) <= 2048
    assert result.elapsed_ms >= result.retrieval_ms + result.support_ms


@pytest.mark.asyncio
async def test_empty_selection_is_distinct_from_provider_failure(prepared):
    result = await run(prepared, Selector(), Selector())
    assert result.status == "assessed" and not result.context
    assert grade_selection([], list(result.selected_ids), result.status)["correct_abstention"]

    async def offline(snapshot, fence):
        raise MnemosTransientError("offline")

    result = await run(prepared, offline)
    assert result.status == "unavailable" and not result.context
    assert not grade_selection([], list(result.selected_ids), result.status)["correct_abstention"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["correct", "forget", "source", "scope"])
@pytest.mark.parametrize("stage", [0, 1])
async def test_change_to_selected_or_unselected_cohort_prevents_context(prepared, mutation, stage):
    store = prepared[0].engine.store

    async def change(snapshot):
        if mutation == "forget":
            store.delete("two")
        elif mutation == "source":
            store._conn.execute("DELETE FROM fact_evidence WHERE fact_id='two'")
            store._conn.commit()
        else:
            chunk = store.get("two")
            if mutation == "correct":
                chunk = chunk.reconsolidate("The marker was withdrawn.")
            else:
                chunk.metadata["scope_id"] = "foreign"
            store.store(chunk)

    selectors = [Selector(("one",)), Selector(("one",))]
    selectors[stage].action = change
    result = await run(prepared, *selectors)
    assert result.status == "unavailable" and result.selected_ids == () and not result.context


@pytest.mark.asyncio
async def test_missing_retained_source_is_unavailable_not_abstention(prepared):
    store = prepared[0].engine.store
    store._conn.execute("DELETE FROM fact_evidence")
    store._conn.commit()
    first, second = Selector(), Selector()
    result = await run(prepared, first, second)
    assert result.status == "unavailable" and not result.context
    assert not first.calls and not second.calls


@pytest.mark.asyncio
async def test_correction_between_retrieval_return_and_support_is_rejected(prepared, monkeypatch):
    service = prepared[0]
    retrieve = service.retrieve

    async def racing(request):
        baseline = await retrieve(request)
        service.engine.store.store(service.engine.store.get("one").reconsolidate("Changed."))
        return baseline

    monkeypatch.setattr(service, "retrieve", racing)
    first = Selector()
    result = await run(prepared, first)
    assert result.status == "unavailable" and not first.calls


@pytest.mark.asyncio
async def test_final_packing_is_followed_by_another_cohort_check(prepared, monkeypatch):
    from benchmarks.local_memory import retrieval_support

    pack = retrieval_support.pack_context

    def racing(*args, **kwargs):
        output = pack(*args, **kwargs)
        prepared[0].engine.store.delete("two")
        return output

    monkeypatch.setattr(retrieval_support, "pack_context", racing)
    result = await run(prepared)
    assert result.status == "unavailable" and not result.context


@pytest.mark.asyncio
async def test_large_records_cannot_escape_the_fixed_complete_context_budget(prepared):
    store = prepared[0].engine.store
    chunk = store.get("two")
    chunk.content = "release " * 1024
    store.store(chunk)
    first = Selector(("one",))
    result = await run(prepared, first)
    assert result.status == "assessed" and result.selected_ids == ("one",)
    assert "two" not in result.candidate_ids and result.context_bytes <= 2048


@pytest.mark.asyncio
async def test_foreign_candidate_in_a_retrieval_response_fails_before_selection(
    prepared, monkeypatch
):
    service = prepared[0]
    retrieve = service.retrieve

    async def foreign(request):
        baseline = await retrieve(request)
        baseline["facts"][0]["id"] = "foreign"
        return baseline

    monkeypatch.setattr(service, "retrieve", foreign)
    first = Selector()
    result = await run(prepared, first)
    assert result.status == "unavailable" and not first.calls and not result.context


@pytest.mark.asyncio
async def test_degraded_retrieval_cannot_be_scored_as_abstention(prepared, monkeypatch):
    async def unavailable(request):
        return {"degraded": True, "facts": [], "context": ""}

    monkeypatch.setattr(prepared[0], "retrieve", unavailable)
    result = await run(prepared)
    assert result.status == "unavailable"
    assert not grade_selection([], [], result.status)["correct_abstention"]


@pytest.mark.asyncio
async def test_empty_accepted_store_abstains_without_a_provider(prepared):
    for key in ["one", "two", "foreign"]:
        prepared[0].engine.store.delete(key)
    first, second = Selector(), Selector()
    result = await run(prepared, first, second)
    assert result.status == "assessed" and not result.candidate_ids and not result.context
    assert not first.calls and not second.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("profile", "biomimetic"),
        ("top_k", 6),
        ("max_context_tokens", 2049),
        ("view", "captured_evidence"),
        ("response_format", "compact"),
        ("allowed_scopes", ["project", "workspace", "global"]),
    ],
)
async def test_development_contract_cannot_drift(prepared, field, value):
    setattr(prepared[1], field, value)
    with pytest.raises(ValueError, match="frozen"):
        await run(prepared)
