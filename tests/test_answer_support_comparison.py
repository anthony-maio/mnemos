"""The support treatment filters the exact baseline candidates without extraction."""

from datetime import datetime, timezone

import pytest

from benchmarks.local_memory.answer_support_comparison import compare_query, grade_selection
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
