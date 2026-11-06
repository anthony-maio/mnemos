"""The support treatment filters the exact baseline candidates without extraction."""

from datetime import datetime, timezone
import subprocess

import pytest

from benchmarks.local_memory.answer_support_comparison import (
    _seed_case,
    compare_query,
    grade_selection,
    SupportTreatment,
    ObservedJudge,
)
from mnemos.answer_support import AnswerSupportConfig
from mnemos.judgment import MockJudge
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.reliability import MnemosTransientError
from tests.test_retrieval_comparison import add_fact, fixture_service


class FixedEmbedding(SimpleEmbeddingProvider):
    def __init__(self):
        super().__init__(dim=2)

    def embed(self, text):
        return [1.0, 0.0]


@pytest.mark.asyncio
async def test_seeded_projects_remain_distinct_inside_a_checkout(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    memory, context = _seed_case(
        tmp_path / "run",
        {
            "id": "scope",
            "facts": [
                {"id": "local", "content": "Morgan owns billing."},
                {"id": "foreign", "content": "River owns search.", "scope": "foreign"},
            ],
        },
        FixedEmbedding(),
    )
    try:
        local = memory.engine.store.get("local")
        foreign = memory.engine.store.get("foreign")
        assert local.metadata["scope_id"] != foreign.metadata["scope_id"]
        rows = await compare_query(
            memory,
            context,
            {"id": "q", "query": "Who owns search?", "expected_fact_ids": []},
            judge=MockJudge({"support:*": "related"}),
            support_config=AnswerSupportConfig(),
        )
        assert len(rows) == 4
        assert all("foreign" not in row["baseline_ids"] for row in rows)
    finally:
        await memory.engine.aclose()


def test_grading_keeps_unavailable_results_out_of_correct_abstention():
    assert grade_selection([], [], "unavailable")["correct_abstention"] is False
    assert grade_selection([], [], "partial")["correct_abstention"] is False
    assert grade_selection([], [], "assessed")["correct_abstention"] is True
    assert grade_selection(["a"], [], "assessed")["answerable_hit"] is False
    assert grade_selection(["a"], ["a", "b"], "assessed")["unsupported_count"] == 1


@pytest.mark.asyncio
async def test_all_arms_share_snapshot_and_support_sees_only_baseline_results(tmp_path):
    memory, context = fixture_service(tmp_path)
    memory.engine._embedder = FixedEmbedding()
    store = memory.engine.store
    try:
        add_fact(memory, context, "release", "Release checks live in docs/release.md.")
        store._conn.execute(
            "INSERT INTO fact_evidence "
            "(evidence_id,fact_id,source_event_id,excerpt,source_ref,created_at) VALUES (?,?,?,?,?,?)",
            (
                "e1",
                "release",
                None,
                "Release checks live in docs/release.md.",
                "fixture:release",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        store._conn.commit()
        before = store.get("release").model_dump()
        judge = MockJudge({"support:*": "evidence:0"})
        rows = await compare_query(
            memory,
            context,
            {"id": "q", "query": "release", "expected_fact_ids": ["release"]},
            judge=judge,
            support_config=AnswerSupportConfig(),
        )
        assert len(rows) == 4
        assert len({row["snapshot_sha256"] for row in rows}) == 1
        for row in rows:
            assert row["supported_ids"] == row["baseline_ids"] == ["release"]
            assert row["support_grade"]["answerable_hit"] is True
            assert row["support_decisions"][0]["evidence_ids"] == ("e1",)
        assert store.get("release").model_dump() == before
    finally:
        store.close()


@pytest.mark.asyncio
async def test_treatments_share_one_baseline_and_bind_observed_requests(tmp_path):
    from contextvars import ContextVar
    import json

    memory, context = _seed_case(
        tmp_path / "run",
        {"id": "release", "facts": [{"id": "path", "content": "Release notes are in RELEASE.md."}]},
        FixedEmbedding(),
    )
    active = ContextVar("test_request", default={})
    bindings = []

    class BoundJudge(MockJudge):
        async def ask(self, *args, **kwargs):
            bindings.append(active.get())
            return await super().ask(*args, **kwargs)

    def observed(name, choice):
        return ObservedJudge(
            name, BoundJudge({"support:*": choice}), tmp_path / "requests.jsonl", active
        )

    treatments = [
        SupportTreatment("select", observed("select", "evidence:0"), AnswerSupportConfig()),
        SupportTreatment("decline", observed("decline", "related"), AnswerSupportConfig()),
    ]
    try:
        before = memory.engine.store.get("path").model_dump()
        rows = await compare_query(
            memory,
            context,
            {"id": "q", "query": "release", "expected_fact_ids": ["path"]},
            judge=MockJudge(),
            support_config=AnswerSupportConfig(),
            treatments=treatments,
        )
        assert len(rows) == 8
        assert len({row["snapshot_sha256"] for row in rows}) == 1
        assert all(row["baseline_ids"] == ["path"] for row in rows)
        assert all(
            row["supported_ids"] == (["path"] if row["treatment"] == "select" else [])
            for row in rows
        )
        assert memory.engine.store.get("path").model_dump() == before
        requests = [
            json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()
        ]
        assert {r["request_id"] for r in requests} == {
            key for row in rows for key in row["judge_request_ids"]
        }
        assert {r["request_id"] for r in bindings} == {r["request_id"] for r in requests}
        assert active.get() == {}
    finally:
        await memory.engine.aclose()


@pytest.mark.asyncio
async def test_comparison_records_provider_failure_as_attempted_failed_observation(tmp_path):
    memory, context = fixture_service(tmp_path)
    memory.engine._embedder = FixedEmbedding()
    store = memory.engine.store
    try:
        add_fact(memory, context, "status", "Release check passed.")
        store._conn.execute(
            "INSERT INTO fact_evidence "
            "(evidence_id,fact_id,source_event_id,excerpt,source_ref,created_at) VALUES (?,?,?,?,?,?)",
            (
                "e1",
                "status",
                None,
                "Release check passed.",
                "fixture:status",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        store._conn.commit()

        class OfflineJudge(MockJudge):
            async def ask(self, *args, **kwargs):
                raise MnemosTransientError("offline")

        rows = await compare_query(
            memory,
            context,
            {"id": "q", "query": "release", "expected_fact_ids": []},
            judge=OfflineJudge(),
            support_config=AnswerSupportConfig(),
        )
        assert len(rows) == 4
        assert all(row["support_status"] == "unavailable" for row in rows)
        assert all(not row["support_grade"]["correct_abstention"] for row in rows)
    finally:
        store.close()
