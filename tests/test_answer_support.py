"""Support judgments require current, scoped facts and retained evidence."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from mnemos.answer_support import AnswerSupportConfig, assess_answer_support
from mnemos.judgment import ChoiceAnswer, MockJudge, NoulAnswer
from mnemos.scopes import ScopeRegistry, context_metadata
from mnemos.types import MemoryChunk
from mnemos.utils.reliability import MnemosConfigurationError, MnemosTransientError
from mnemos.utils.storage import SQLiteStore


@pytest.fixture
def memory(tmp_path):
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="manual")
    store = SQLiteStore(str(tmp_path / "support.sqlite"))
    yield store, registry, context
    store.close()


def fact(
    memory, identifier="release", content="Release checks live in docs/release.md.", **metadata
):
    store, _, context = memory
    chunk = MemoryChunk(
        id=identifier,
        content=content,
        metadata={
            **context_metadata(context, "project"),
            "assertion_status": "asserted",
            **metadata,
        },
    )
    store.store(chunk)
    store._conn.execute(
        "INSERT INTO fact_evidence "
        "(evidence_id, fact_id, source_event_id, excerpt, source_ref, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            identifier + "-evidence",
            identifier,
            None,
            content,
            "fixture:" + identifier,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    store._conn.commit()
    return chunk


async def assess(memory, judge, ids=("release",), **kwargs):
    store, registry, context = memory
    return await assess_answer_support(
        "Where are the release checks?",
        ids,
        store=store,
        registry=registry,
        context=context,
        judge=judge,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_supported_result_cites_exact_retained_evidence_without_rewriting(memory):
    chunk = fact(memory)
    judge = MockJudge({"support:0": "evidence:0"})
    before = chunk.model_dump()
    result = await assess(memory, judge)
    assert result.status == "assessed"
    assert [item.fact_id for item in result.supported] == [chunk.id]
    decision = result.decisions[0]
    assert decision.disposition == "supported"
    assert decision.evidence_ids == ("release-evidence",)
    assert result.supported[0].content == chunk.content
    assert memory[0].get(chunk.id).model_dump() == before
    state, questions, purpose = judge.calls[0]
    assert state["query"] == "Where are the release checks?"
    assert purpose == "answer_support_v1"
    assert "related" in questions["support:0"].keys


@pytest.mark.asyncio
@pytest.mark.parametrize("disposition", ["unsupported", "related", "unresolved"])
async def test_related_or_unresolved_text_is_not_promoted_to_supported(memory, disposition):
    fact(memory, content="The deployment check passed, but review is still pending.")
    result = await assess(memory, MockJudge({"support:0": disposition}))
    assert result.status == "assessed"
    assert result.supported == ()
    assert result.decisions[0].disposition == disposition


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        {"scope_id": "foreign"},
        {"user_profile_id": "foreign"},
        {"source": "captured_passthrough"},
        {"assertion_status": "unknown"},
    ],
)
async def test_ineligible_facts_never_reach_judge(memory, metadata):
    fact(memory, content="Private or unverified text", **metadata)
    judge = MockJudge({"support:*": "evidence:0"})
    result = await assess(memory, judge)
    assert not judge.calls
    assert not result.supported
    assert result.decisions[0].disposition == "ineligible"


@pytest.mark.asyncio
async def test_conflict_cannot_be_silently_resolved_by_judge(memory):
    fact(memory, assertion_status="conflict")
    judge = MockJudge({"support:*": "evidence:0"})
    result = await assess(memory, judge)
    assert not judge.calls
    assert result.decisions[0].disposition == "unresolved"
    assert not result.supported


@pytest.mark.asyncio
async def test_fact_without_retained_evidence_is_unassessed(memory):
    fact(memory)
    memory[0]._conn.execute("DELETE FROM fact_evidence")
    memory[0]._conn.commit()
    judge = MockJudge({"support:*": "evidence:0"})
    result = await assess(memory, judge)
    assert result.status == "partial"
    assert result.decisions[0].reason == "no_retained_evidence"
    assert not result.supported and not judge.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["correct", "forget", "evidence", "scope"])
async def test_awaited_judgment_cannot_validate_a_changed_snapshot(memory, change):
    fact(memory)
    store = memory[0]

    class ChangingJudge(MockJudge):
        async def ask(self, *args, **kwargs):
            await asyncio.sleep(0)
            if change == "forget":
                store.delete("release")
            elif change == "evidence":
                store._conn.execute("UPDATE fact_evidence SET excerpt='Changed evidence'")
                store._conn.commit()
            else:
                updated = store.get("release")
                if change == "correct":
                    updated = updated.reconsolidate("Release checks moved to RELEASE.md.")
                else:
                    updated.metadata["scope_id"] = "foreign"
                store.store(updated)
            return await super().ask(*args, **kwargs)

    result = await assess(memory, ChangingJudge({"support:*": "evidence:0"}))
    assert result.status == "partial"
    assert not result.supported
    assert result.decisions[0].disposition == "stale"


@pytest.mark.asyncio
async def test_paused_project_is_revalidated_after_judge(memory):
    fact(memory)

    class PausingJudge(MockJudge):
        async def ask(self, *args, **kwargs):
            memory[1].projects[memory[2].project_id].enabled = False
            return await super().ask(*args, **kwargs)

    with pytest.raises(ValueError, match="not enabled"):
        await assess(memory, PausingJudge({"support:*": "evidence:0"}))


@pytest.mark.asyncio
async def test_low_probability_support_is_unresolved(memory):
    fact(memory)

    def script(key, question, state):
        return ChoiceAnswer("evidence:0", {"evidence:0": 0.6, "related": 0.4}, 0.6)

    result = await assess(memory, MockJudge(script))
    assert not result.supported
    assert result.decisions[0].reason == "low_probability"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transient", "malformed", "timeout"])
async def test_provider_failure_is_not_successful_abstention(memory, failure):
    fact(memory)

    class FailingJudge(MockJudge):
        async def ask(self, *args, **kwargs):
            if failure == "transient":
                raise MnemosTransientError("offline")
            if failure == "timeout":
                await asyncio.sleep(1)
            return await super().ask(*args, **kwargs)

    result = await assess(
        memory,
        FailingJudge({"support:*": NoulAnswer(1.0)}),
        config=AnswerSupportConfig(timeout_seconds=0.01),
    )
    assert result.status == "unavailable"
    assert result.error_type
    assert not result.supported


@pytest.mark.asyncio
async def test_configuration_failure_is_not_hidden(memory):
    fact(memory)

    class MisconfiguredJudge(MockJudge):
        async def ask(self, *args, **kwargs):
            raise MnemosConfigurationError("invalid provider")

    with pytest.raises(MnemosConfigurationError):
        await assess(memory, MisconfiguredJudge())


@pytest.mark.asyncio
async def test_request_limits_fail_before_provider_call(memory):
    fact(memory)
    judge = MockJudge()
    with pytest.raises(ValueError, match="candidate limit"):
        await assess(
            memory, judge, ids=("release", "other"), config=AnswerSupportConfig(max_candidates=1)
        )
    result = await assess(memory, judge, config=AnswerSupportConfig(max_request_bytes=100))
    assert result.status == "unavailable"
    assert result.error_type == "JudgmentRequestTooLarge"
    assert not judge.calls
