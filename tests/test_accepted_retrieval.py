from datetime import datetime, timedelta, timezone

import pytest

from mnemos.contracts import RetrievalRequest
from mnemos.jobs import FactEvidence
from mnemos.scopes import ProjectIdentity, ScopeContext, ScopeRegistry, context_metadata
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.embeddings import EmbeddingProvider

from benchmarks.local_memory.accepted_retrieval import (
    AcceptedSnapshot,
    build_accepted_snapshot,
    compare_accepted_query,
)

STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


class ObservedEmbedding(EmbeddingProvider):
    dim = 2

    def __init__(self):
        self.calls = []
        self.fail_query = False

    def embed(self, text):
        raise AssertionError("Embedding roles must be explicit")

    def embed_batch(self, texts):
        raise AssertionError("Unexpected batch call")

    def embed_document(self, text):
        self.calls.append(("document", text))
        return [1.0, 0.0]

    def embed_query(self, text):
        self.calls.append(("query", text))
        if self.fail_query:
            raise RuntimeError("fixture query failure")
        return [1.0, 0.0]


def snapshot():
    registry = ScopeRegistry(
        profiles=["local"],
        projects={
            key: ProjectIdentity(
                id=key, root="fixture:" + key, user_profile_id="local", enabled=True
            )
            for key in ["ours", "foreign"]
        },
    )
    facts, sources, evidence = [], [], []
    for key, project, content in [
        ("current", "ours", "Orchid logs are retained for 30 days."),
        ("foreign", "foreign", "Orchid logs are retained for 7 days."),
    ]:
        context = ScopeContext(
            user_profile_id="local", project_id=project, session_id="s", host="codex"
        )
        metadata = context_metadata(context, "project")
        sources.append(
            EpisodicEvent(
                id=key + "-source",
                event_fingerprint=key + "-fingerprint",
                role="user",
                content=content,
                normalized_content=content,
                scope="project",
                scope_id=project,
                source="declared_fixture",
                metadata=metadata,
                status="consolidated",
                occurred_at=STAMP,
                created_at=STAMP,
            )
        )
        facts.append(
            MemoryChunk(
                id=key,
                content=content,
                version=2,
                created_at=STAMP,
                updated_at=STAMP,
                last_observed_at=STAMP,
                metadata={**metadata, "assertion_status": "asserted"},
            )
        )
        evidence.append(
            FactEvidence(
                evidence_id=key + "-evidence",
                fact_id=key,
                source_event_id=key + "-source",
                excerpt=content,
                source_ref="fixture:" + key,
                created_at=STAMP,
            )
        )
    return AcceptedSnapshot(
        snapshot_id="terminal-state",
        as_of=STAMP,
        registry=registry,
        facts=tuple(facts),
        sources=tuple(sources),
        evidence=tuple(evidence),
    )


def request():
    return RetrievalRequest(
        query="Orchid logs",
        context=ScopeContext(
            user_profile_id="local", project_id="ours", session_id="query", host="codex"
        ),
        allowed_scopes=["project"],
        top_k=5,
        max_context_tokens=1200,
    )


def test_build_preserves_declared_fact_versions_and_sources_without_jobs(tmp_path):
    fixture = snapshot()
    before = fixture.model_dump(mode="json")
    provider = ObservedEmbedding()
    prepared = build_accepted_snapshot(tmp_path / "accepted.sqlite", fixture, provider, dimension=2)
    try:
        store = prepared.service.engine.store
        assert {fact.id for fact in store.get_all()} == {"current", "foreign"}
        assert store.get("current").version == 2
        assert store.get("current").content == fixture.facts[0].content
        (support,) = store.list_fact_evidence("current")
        assert store.get_episodic_event(support.source_event_id).content == support.excerpt
        assert store.list_jobs() == []
        assert not any(prepared.service.engine.config.modules.model_dump().values())
        assert fixture.model_dump(mode="json") == before
        assert provider.calls == [("document", fact.content) for fact in fixture.facts]
    finally:
        prepared.service.engine.store.close()


@pytest.mark.asyncio
async def test_strategies_share_vectors_and_snapshot_but_not_access_writes(tmp_path):
    provider = ObservedEmbedding()
    prepared = build_accepted_snapshot(
        tmp_path / "accepted.sqlite", snapshot(), provider, dimension=2
    )
    try:
        rows = await compare_accepted_query(
            prepared, request(), query_id="q", expected_fact_ids=["current"], repetition=1
        )
        assert {row["strategy"] for row in rows} == {"vector", "lexical", "hybrid"}
        assert len({row["sqlite_snapshot_sha256"] for row in rows}) == 1
        assert (
            len({row["query_vector_sha256"] for row in rows if row["strategy"] != "lexical"}) == 1
        )
        assert all(row["selected_ids"] == ["current"] for row in rows)
        assert all(row["grade"]["answerable_hit"] for row in rows)
        assert provider.calls.count(("query", "Orchid logs")) == 1
        assert prepared.service.engine.store.get("current").access_count == 0
        again = await compare_accepted_query(
            prepared, request(), query_id="q2", expected_fact_ids=[], repetition=2
        )
        assert all(not row["grade"]["correct_abstention"] for row in again)
        assert all(row["selected_ids"] == ["current"] for row in again)
    finally:
        prepared.service.engine.store.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_source",
        "wrong_scope",
        "nonliteral",
        "future",
        "label",
        "unknown",
        "duplicate",
        "source_label",
        "fingerprint",
        "evidence_id",
        "source_scope",
        "source_id",
        "no_evidence",
        "future_fact",
        "naive",
        "disabled_project",
    ],
)
def test_invalid_snapshot_rejected_before_embedding_or_database(tmp_path, mutation):
    fixture = snapshot()
    if mutation == "missing_source":
        fixture.evidence[0].source_event_id = "missing"
    elif mutation == "wrong_scope":
        fixture.evidence[0].source_event_id = "foreign-source"
        fixture.evidence[0].excerpt = fixture.sources[1].content
    elif mutation == "nonliteral":
        fixture.evidence[0].excerpt = "invented quote"
    elif mutation == "future":
        fixture.sources[0].occurred_at = STAMP + timedelta(days=1)
    elif mutation == "label":
        fixture.facts[0].metadata["expected_answer"] = "30 days"
    elif mutation == "unknown":
        fixture.facts[0].metadata["assertion_status"] = "unknown"
    elif mutation == "duplicate":
        fixture.facts[1].id = "current"
    elif mutation == "source_label":
        fixture.sources[0].metadata["relevant_fact_ids"] = ["current"]
    elif mutation == "fingerprint":
        fixture.sources[1].event_fingerprint = fixture.sources[0].event_fingerprint
    elif mutation == "evidence_id":
        fixture.evidence[1].evidence_id = fixture.evidence[0].evidence_id
    elif mutation == "source_scope":
        fixture.sources[0].scope_id = "foreign"
    elif mutation == "source_id":
        fixture.sources[1].id = fixture.sources[0].id
    elif mutation == "no_evidence":
        fixture.evidence = fixture.evidence[1:]
    elif mutation == "future_fact":
        fixture.facts[0].last_observed_at = STAMP + timedelta(seconds=1)
    elif mutation == "naive":
        fixture.as_of = STAMP.replace(tzinfo=None)
    else:
        fixture.registry.projects["ours"].enabled = False
    provider = ObservedEmbedding()
    with pytest.raises(ValueError):
        build_accepted_snapshot(tmp_path / "accepted.sqlite", fixture, provider, dimension=2)
    assert provider.calls == []
    assert not (tmp_path / "accepted.sqlite").exists()


@pytest.mark.asyncio
async def test_query_embedding_failure_cannot_earn_abstention_credit(tmp_path):
    provider = ObservedEmbedding()
    prepared = build_accepted_snapshot(
        tmp_path / "accepted.sqlite", snapshot(), provider, dimension=2
    )
    provider.fail_query = True
    try:
        query = request().model_copy(update={"query": "Zzzabsentword"})
        rows = await compare_accepted_query(
            prepared, query, query_id="absent", expected_fact_ids=[], repetition=1
        )
        for row in rows:
            if row["strategy"] == "lexical":
                assert row["grade"]["correct_abstention"]
                assert row["grade"]["operational_success"]
            else:
                assert row["error_type"] == "RuntimeError"
                assert not row["grade"]["correct_abstention"]
                assert not row["grade"]["operational_success"]
    finally:
        prepared.service.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["content", "evidence", "config"])
async def test_changed_accepted_state_is_refused_before_query_embedding(tmp_path, mutation):
    provider = ObservedEmbedding()
    prepared = build_accepted_snapshot(
        tmp_path / "accepted.sqlite", snapshot(), provider, dimension=2
    )
    store = prepared.service.engine.store
    try:
        if mutation == "content":
            fact = store.get("current")
            fact.content = "Changed accepted fact"
            store.store(fact)
        elif mutation == "evidence":
            store._conn.execute("UPDATE fact_evidence SET excerpt='changed'")
            store._conn.commit()
        else:
            prepared.service.engine.config.modules.sleep = True
        with pytest.raises(ValueError, match="accepted state changed"):
            await compare_accepted_query(
                prepared, request(), query_id="q", expected_fact_ids=["current"], repetition=1
            )
        assert all(role == "document" for role, _ in provider.calls)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_foreign_expected_answer_rejected_before_query_embedding(tmp_path):
    provider = ObservedEmbedding()
    prepared = build_accepted_snapshot(
        tmp_path / "accepted.sqlite", snapshot(), provider, dimension=2
    )
    try:
        with pytest.raises(ValueError, match="visible accepted"):
            await compare_accepted_query(
                prepared, request(), query_id="q", expected_fact_ids=["foreign"], repetition=1
            )
        assert all(role == "document" for role, _ in provider.calls)
    finally:
        prepared.service.engine.store.close()


@pytest.mark.parametrize(
    "vector", [[1.0], [float("nan"), 0.0], [float("inf"), 0.0], [0.0, 0.0], [True, False]]
)
def test_invalid_document_vectors_leave_no_database(tmp_path, vector):
    provider = ObservedEmbedding()
    provider.embed_document = lambda text: vector
    with pytest.raises(ValueError, match="Embedding"):
        build_accepted_snapshot(tmp_path / "accepted.sqlite", snapshot(), provider, dimension=2)
    assert not (tmp_path / "accepted.sqlite").exists()


def test_destination_created_during_embedding_is_preserved(tmp_path):
    path = tmp_path / "accepted.sqlite"
    provider = ObservedEmbedding()
    original = provider.embed_document

    def concurrent_create(text):
        path.write_bytes(b"another owner")
        return original(text)

    provider.embed_document = concurrent_create
    with pytest.raises(FileExistsError):
        build_accepted_snapshot(path, snapshot(), provider, dimension=2)
    assert path.read_bytes() == b"another owner"


@pytest.mark.asyncio
async def test_conflict_stays_labeled_and_does_not_become_an_answer_label(tmp_path):
    fixture = snapshot()
    fixture.facts[0].metadata["assertion_status"] = "conflict"
    prepared = build_accepted_snapshot(
        tmp_path / "accepted.sqlite", fixture, ObservedEmbedding(), dimension=2
    )
    try:
        rows = await compare_accepted_query(
            prepared, request(), query_id="conflict", expected_fact_ids=[], repetition=1
        )
        assert all("unresolved conflict" in row["result"]["context"] for row in rows)
        assert all(row["grade"]["scope_leak_count"] == 0 for row in rows)
        assert all(not row["grade"]["correct_abstention"] for row in rows)
        with pytest.raises(ValueError, match="visible accepted"):
            await compare_accepted_query(
                prepared,
                request(),
                query_id="conflict",
                expected_fact_ids=["current"],
                repetition=1,
            )
    finally:
        prepared.service.engine.store.close()


def test_existing_database_refused_before_provider_call(tmp_path):
    path = tmp_path / "accepted.sqlite"
    path.write_bytes(b"preserve")
    provider = ObservedEmbedding()
    with pytest.raises(FileExistsError):
        build_accepted_snapshot(path, snapshot(), provider, dimension=2)
    assert path.read_bytes() == b"preserve"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_global_visibility_follows_explicit_scope_permission(tmp_path):
    fixture = snapshot()
    fixture.facts[0].metadata.update(scope="global", scope_id=None)
    fixture.sources[0].metadata.update(scope="global", scope_id=None)
    fixture.sources[0].scope = "global"
    fixture.sources[0].scope_id = None
    prepared = build_accepted_snapshot(
        tmp_path / "accepted.sqlite", fixture, ObservedEmbedding(), dimension=2
    )
    try:
        hidden = await compare_accepted_query(
            prepared, request(), query_id="project-only", expected_fact_ids=[], repetition=1
        )
        assert all(
            row["selected_ids"] == [] and row["grade"]["correct_abstention"] for row in hidden
        )
        permitted = request().model_copy(update={"allowed_scopes": ["project", "global"]})
        visible = await compare_accepted_query(
            prepared, permitted, query_id="global", expected_fact_ids=["current"], repetition=1
        )
        assert all(row["selected_ids"] == ["current"] for row in visible)
    finally:
        prepared.service.engine.store.close()


@pytest.mark.asyncio
async def test_recall_fraction_is_distinct_from_one_answer_hit(tmp_path):
    fixture = snapshot()
    fixture.facts[1].metadata.update(project_id="ours", scope_id="ours")
    fixture.sources[1].metadata.update(project_id="ours", scope_id="ours")
    fixture.sources[1].scope_id = "ours"
    prepared = build_accepted_snapshot(
        tmp_path / "accepted.sqlite", fixture, ObservedEmbedding(), dimension=2
    )
    try:
        query = request().model_copy(update={"top_k": 1})
        rows = await compare_accepted_query(
            prepared,
            query,
            query_id="partial",
            expected_fact_ids=["current", "foreign"],
            repetition=1,
        )
        assert all(row["grade"]["answerable_hit"] and row["grade"]["recall"] == 0.5 for row in rows)
        assert all(row["grade"]["precision"] == 1.0 for row in rows)
    finally:
        prepared.service.engine.store.close()


@pytest.mark.asyncio
async def test_provider_cannot_mutate_source_state_during_query_embedding(tmp_path):
    provider = ObservedEmbedding()
    prepared = build_accepted_snapshot(
        tmp_path / "accepted.sqlite", snapshot(), provider, dimension=2
    )
    store = prepared.service.engine.store
    original = provider.embed_query

    def mutate(text):
        fact = store.get("current")
        fact.version += 1
        store.store(fact)
        return original(text)

    provider.embed_query = mutate
    try:
        with pytest.raises(ValueError, match="during query embedding"):
            await compare_accepted_query(
                prepared,
                request(),
                query_id="mutation",
                expected_fact_ids=["current"],
                repetition=1,
            )
    finally:
        store.close()


def test_preflight_never_reads_lazy_provider_dimension(tmp_path):
    class LazyProvider(ObservedEmbedding):
        @property
        def dim(self):
            raise AssertionError("Reading dimension would trigger unrecorded inference")

    fixture = snapshot()
    fixture.evidence[0].excerpt = "not present"
    provider = LazyProvider()
    with pytest.raises(ValueError, match="literal"):
        build_accepted_snapshot(tmp_path / "invalid.sqlite", fixture, provider, dimension=2)
    valid = build_accepted_snapshot(tmp_path / "valid.sqlite", snapshot(), provider, dimension=2)
    valid.service.engine.store.close()
    assert len(provider.calls) == 2


def test_new_sidecar_is_preserved_and_database_is_not_created(tmp_path):
    path = tmp_path / "accepted.sqlite"
    sidecar = tmp_path / "accepted.sqlite-wal"
    provider = ObservedEmbedding()
    original = provider.embed_document

    def create_sidecar(text):
        sidecar.write_bytes(b"another owner")
        return original(text)

    provider.embed_document = create_sidecar
    with pytest.raises(FileExistsError, match="sidecar"):
        build_accepted_snapshot(path, snapshot(), provider, dimension=2)
    assert not path.exists() and sidecar.read_bytes() == b"another owner"
