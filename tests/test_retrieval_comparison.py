from datetime import datetime, timezone

import pytest

from mnemos.config import MnemosConfig
from mnemos.contracts import RetrievalRequest
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry, context_metadata
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


def fixture_service(tmp_path):
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="codex")
    config = MnemosConfig()
    for flag in type(config.modules).model_fields:
        setattr(config.modules, flag, False)
    engine = MnemosEngine(
        config=config,
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(),
        store=SQLiteStore(str(tmp_path / "source.sqlite")),
    )
    return LocalMemoryService(engine, registry), context


def add_fact(service, context, identity, content, embedding=None):
    chunk = MemoryChunk(
        id=identity,
        content=content,
        embedding=embedding or [1.0, 0.0],
        metadata={**context_metadata(context, "project"), "assertion_status": "asserted"},
    )
    service.engine.store.store(chunk)
    return chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_identity", ["user_profile_id", "project_id"])
async def test_lexical_needs_no_embedding_and_scopes_before_candidate_limit(
    tmp_path, foreign_identity
):
    from benchmarks.local_memory.retrieval_comparison import comparison_snapshots

    service, context = fixture_service(tmp_path)

    class BrokenEmbedder:
        def embed(self, text):
            raise AssertionError("Lexical search must not embed queries")

    service.engine._embedder = BrokenEmbedder()
    try:
        for i in range(40):
            add_fact(
                service,
                context.model_copy(update={foreign_identity: "foreign"}),
                f"foreign-{i}",
                "Orchid",
            )
        add_fact(service, context, "ours", "Orchid exact codeword", [0.0, 1.0])
        with comparison_snapshots(service) as (digest, variants):
            result = await variants["lexical"].retrieve(
                RetrievalRequest(
                    query="Orchid",
                    context=context,
                    top_k=1,
                )
            )
            assert [fact["id"] for fact in result["facts"]] == ["ours"]
            assert result["facts"][0]["scores"]["semantic"] is None
            assert result["facts"][0]["scores"]["ranking_method"] == "sqlite_fts5_bm25"
            assert len(digest) == 64
            assert variants["lexical"].engine.store.get("ours").access_count == 1
            assert variants["vector"].engine.store.get("ours").access_count == 0
        assert service.engine.store.get("ours").access_count == 0
        assert service.engine.store.get("ours").version == 1
    finally:
        service.engine.store.close()


def test_snapshot_corruption_fails_before_any_strategy_runs(tmp_path, monkeypatch):
    import shutil
    from benchmarks.local_memory.retrieval_comparison import comparison_snapshots

    service, context = fixture_service(tmp_path)
    add_fact(service, context, "ours", "Durable source")
    copy = shutil.copyfile

    def corrupt(source, destination):
        result = copy(source, destination)
        with open(destination, "ab") as handle:
            handle.write(b"corrupt")
        return result

    monkeypatch.setattr(shutil, "copyfile", corrupt)
    try:
        with pytest.raises(RuntimeError, match="snapshot copy changed"):
            with comparison_snapshots(service):
                pytest.fail("Corrupted snapshots must not be exposed")
        assert service.engine.store.get("ours").content == "Durable source"
        assert not list(tmp_path.glob("retrieval-*"))
    finally:
        service.engine.store.close()


@pytest.mark.asyncio
async def test_each_query_restarts_from_original_accepted_state(tmp_path):
    from benchmarks.local_memory.retrieval_comparison import comparison_snapshots

    service, context = fixture_service(tmp_path)
    fact = add_fact(service, context, "ours", "Orchid")

    class FixedEmbedder:
        def embed(self, text):
            return [1.0, 0.0]

    service.engine._embedder = FixedEmbedder()
    try:
        digests = []
        for _ in range(2):
            with comparison_snapshots(service) as (digest, variants):
                digests.append(digest)
                lexical = variants["lexical"]
                assert lexical.engine.store.get("ours").access_count == 0
                await lexical.retrieve(RetrievalRequest(query="Orchid", context=context))
                assert lexical.engine.store.get("ours").access_count == 1
                assert lexical.engine.store.get("ours").content == fact.content
                assert lexical.engine.store.get("ours").version == fact.version
                for name in ("vector", "hybrid", "full"):
                    assert variants[name].engine.store.get("ours").model_dump() == fact.model_dump()
                    result = await variants[name].retrieve(
                        RetrievalRequest(query="Orchid", context=context)
                    )
                    assert [item["id"] for item in result["facts"]] == ["ours"]
                    assert variants[name].engine.store.get("ours").access_count == 1
                    assert service.engine.store.get("ours").model_dump() == fact.model_dump()
        assert digests[0] == digests[1]
        assert service.engine.store.get("ours").model_dump() == fact.model_dump()
    finally:
        service.engine.store.close()


@pytest.mark.asyncio
async def test_vector_does_not_use_fts_and_skips_oversized_candidates(tmp_path, monkeypatch):
    from benchmarks.local_memory.retrieval_comparison import comparison_snapshots
    from mnemos.retrieval import context_token_upper_bound

    service, context = fixture_service(tmp_path)

    class FixedEmbedder:
        def embed(self, text):
            return [1.0, 0.0]

    service.engine._embedder = FixedEmbedder()
    try:
        for i in range(40):
            add_fact(
                service,
                context.model_copy(update={"user_profile_id": "foreign"}),
                f"0-foreign-{i}",
                "compact",
            )
        add_fact(service, context, "a-long", "long " * 200)
        small = add_fact(service, context, "b-short", "compact")
        add_fact(service, context, "c-unrelated", "Orchid", [0.0, 1.0])
        with comparison_snapshots(service) as (_, variants):

            def forbidden(*args, **kwargs):
                raise AssertionError("Vector arm must not query FTS")

            monkeypatch.setattr(variants["vector"].engine.store, "retrieve_lexical", forbidden)
            result = await variants["vector"].retrieve(
                RetrievalRequest(
                    query="Orchid",
                    context=context,
                    top_k=1,
                    max_context_tokens=context_token_upper_bound(small),
                )
            )
            assert [fact["id"] for fact in result["facts"]] == ["b-short"]
            assert result["facts"][0]["scores"]["semantic"] == 1.0
            assert result["context_token_upper_bound"] <= context_token_upper_bound(small)
    finally:
        service.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy,profile", [("hybrid", "baseline"), ("full", "biomimetic")])
async def test_existing_profiles_use_production_engine(tmp_path, monkeypatch, strategy, profile):
    from benchmarks.local_memory.retrieval_comparison import comparison_snapshots

    service, context = fixture_service(tmp_path)
    observed = []

    async def production(self, query, **kwargs):
        observed.append((query, kwargs["profile"], kwargs["context"]))
        return []

    monkeypatch.setattr(MnemosEngine, "retrieve", production)
    try:
        with comparison_snapshots(service) as (_, variants):
            await variants[strategy].retrieve(RetrievalRequest(query="query", context=context))
        assert observed == [("query", profile, context)]
    finally:
        service.engine.store.close()


@pytest.mark.parametrize("broken_vector", [False, True])
def test_adapter_records_all_strategies_and_preserves_missing_results(
    tmp_path, monkeypatch, broken_vector
):
    from benchmarks.local_memory.retrieval_comparison import RetrievalComparatorAdapter
    from benchmarks.local_memory.contracts import (
        SessionScenario,
        SourceEvent,
        FactRecord,
        QueryRecord,
        StatePair,
    )
    from benchmarks.local_memory.harness import DEFAULT_PROFILES

    memory, context = fixture_service(tmp_path)
    memory.engine.store.close()

    def factory(profile, run_key=None):
        return fixture_service(tmp_path)[0]

    # Keep the same registry identity across factory openings.
    def stable_factory(profile, run_key=None):
        service = factory(profile, run_key)
        service.registry = memory.registry
        return service

    if broken_vector:
        original = SQLiteStore.retrieve

        def fail_vector(self, *args, **kwargs):
            if str(self.db_path).endswith("vector.sqlite"):
                raise RuntimeError("deliberate vector failure")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(SQLiteStore, "retrieve", fail_vector)
    session = SessionScenario(
        "retrieval-dev",
        "development",
        "codex",
        context.project_id,
        (FactRecord("f", "Use Orchid", True, "project", "synthetic_seed", "supported", ("e",)),),
        (QueryRecord("q", "Orchid", "answerable", ("f",), "Use Orchid"),),
        (
            (
                StatePair(
                    "pair", "Orchid", "I feel relieved about Orchid", ("f",), "calm", "Use Orchid"
                ),
            )
            if not broken_vector
            else ()
        ),
        "synthetic_seed",
        "not_run",
        (
            SourceEvent(
                "e",
                datetime.now(timezone.utc).isoformat(),
                "user",
                "Use Orchid",
                "user_prompt",
                "explicit",
                "remember",
                ("f",),
            ),
        ),
        (),
    )
    result = RetrievalComparatorAdapter(stable_factory, user_profile_id="local").evaluate_session(
        session,
        DEFAULT_PROFILES[1],
        1,
    )
    comparison = result.raw_output["retrieval_comparisons"][0]
    assert set(comparison["strategies"]) == {"vector", "lexical", "hybrid", "full"}
    assert len(set(row["snapshot_sha256"] for row in comparison["strategies"].values())) == 1
    assert comparison["strategies"]["lexical"]["quality_hit"] is True
    assert comparison["strategies"]["lexical"]["result"]["profile"] == "lexical"
    assert comparison["strategies"]["lexical"]["details"]["retrieval_features"] == {
        "embedding": False,
        "affective": False,
        "spreading": False,
    }
    assert comparison["strategies"]["vector"]["details"]["retrieval_features"]["embedding"] is True
    assert result.verified_model_provenance is False
    if broken_vector:
        assert comparison["strategies"]["vector"]["quality_hit"] is None
        assert comparison["strategies"]["vector"]["status"] == "partial"
        assert result.execution_status == "partial"
    else:
        assert result.execution_status == "complete", result.raw_output["errors"]
        records = result.raw_output["retrieval_comparisons"]
        assert [(row["query_id"], row["quality_kind"]) for row in records] == [
            ("q", "answerable"),
            ("pair:semantic", "state_pair_semantic"),
            ("pair:state", "state_pair_state"),
        ]
        assert len({row["snapshot_sha256"] for row in records}) == 1
