"""Evidence and lifecycle contracts for mutable reconsolidation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mnemos.contracts import FactProposal
from mnemos.jobs import DurableJob
from mnemos.modules.mutable_rag import MutableRAG
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import SQLiteStore


def _event(event_id: str, content: str, profile: str = "profile-a") -> EpisodicEvent:
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=f"finger-{event_id}",
        role="user",
        content=content,
        normalized_content=" ".join(content.lower().split()),
        scope="project",
        scope_id="repo",
        source="manual",
        metadata={
            "scope": "project",
            "scope_id": "repo",
            "user_profile_id": profile,
            "source_event_id": event_id,
        },
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _fact(embedder: SimpleEmbeddingProvider) -> MemoryChunk:
    return MemoryChunk(
        id="fact-python",
        content="The project uses Python",
        embedding=embedder.embed("The project uses Python"),
        metadata={"scope": "project", "scope_id": "repo", "user_profile_id": "profile-a"},
    )


@pytest.mark.asyncio
async def test_direct_review_returns_authoritative_metadata_after_captured_merge(tmp_path):
    embedder = SimpleEmbeddingProvider(dim=32)
    store = SQLiteStore(str(tmp_path / "merged-review.sqlite"))
    try:
        target = _fact(embedder)
        target.metadata.update(source="explicit_remember", assertion_status="asserted")
        store.upsert_canonical(target)
        duplicate = target.model_copy(
            deep=True,
            update={
                "id": "captured-rust",
                "content": "The project uses Rust",
                "embedding": embedder.embed("The project uses Rust"),
                "metadata": {
                    **target.metadata,
                    "source": "captured_passthrough",
                    "assertion_status": "unknown",
                    "source_event_ids": ["captured-source"],
                },
            },
        )
        store.append_episodic_event(_event("captured-source", duplicate.content))
        store.upsert_canonical(duplicate)
        source = _event("correction-source", "The project switched from Python to Rust")
        store.append_episodic_event(source)
        rag = MutableRAG(MockLLMProvider(), embedder, store)
        result, changed = await rag.review(
            FactProposal(
                content=duplicate.content,
                evidence_ids=[source.id],
                operation="correct",
                target_fact_id=target.id,
                expected_version=1,
            )
        )
        assert changed and store.get(duplicate.id) is None
        assert result.model_dump() == store.get(target.id).model_dump()
        assert duplicate.id in result.metadata["absorbed_captured_ids"]
        assert set(result.metadata["source_event_ids"]) == {"captured-source", source.id}
    finally:
        store.close()


@pytest.mark.asyncio
async def test_retrieve_does_not_enqueue_query_as_review_evidence(tmp_path: Path) -> None:
    embedder = SimpleEmbeddingProvider(dim=32)
    store = SQLiteStore(str(tmp_path / "retrieve.sqlite"))
    try:
        fact = _fact(embedder)
        store.store(fact)
        rag = MutableRAG(MockLLMProvider(), embedder, store)
        rag.retrieve(
            embedder.embed("did we switch to Rust?"), current_context="did we switch to Rust?"
        )
        assert store.list_reconsolidation_jobs() == []
        assert store.get(fact.id).content == fact.content
    finally:
        store.close()


@pytest.mark.asyncio
async def test_typed_review_requires_existing_same_scope_evidence_and_cas(tmp_path: Path) -> None:
    embedder = SimpleEmbeddingProvider(dim=32)
    store = SQLiteStore(str(tmp_path / "review.sqlite"))
    try:
        fact = _fact(embedder)
        source = _event("source-rust", "The project switched from Python to Rust")
        store.store(fact)
        store.append_episodic_event(source)
        unrelated = store.enqueue_job(DurableJob(kind="extract", payload={"event_id": "other"}))
        rag = MutableRAG(MockLLMProvider(), embedder, store)

        query_only, changed = await rag.reconsolidate(
            fact, "did we switch to Rust?", evidence_refs=[]
        )
        assert changed is False
        assert query_only.content == fact.content

        proposal = FactProposal(
            content="The project uses Rust",
            evidence_ids=[source.id],
            operation="correct",
            target_fact_id=fact.id,
            expected_version=1,
        )
        revised, changed = await rag.review(proposal)
        assert changed is True
        assert revised.id == fact.id
        assert revised.version == 2
        assert store.get(fact.id).content == "The project uses Rust"
        assert store.list_fact_evidence(fact.id)[0].source_event_id == source.id
        assert store.list_jobs(status="completed")[0].result["decision"] == "corrected"
        assert [item.id for item in store.list_jobs(status="pending")] == [unrelated.id]

        with pytest.raises(ValueError, match="version conflict"):
            await rag.review(proposal)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_review_rejects_missing_or_foreign_evidence_without_writing(tmp_path: Path) -> None:
    embedder = SimpleEmbeddingProvider(dim=32)
    store = SQLiteStore(str(tmp_path / "reject.sqlite"))
    try:
        fact = _fact(embedder)
        store.store(fact)
        rag = MutableRAG(MockLLMProvider(), embedder, store)
        proposal = FactProposal(
            content="The project uses Rust",
            evidence_ids=["missing"],
            operation="correct",
            target_fact_id=fact.id,
            expected_version=1,
        )
        with pytest.raises(ValueError, match="does not exist"):
            await rag.review(proposal)
        assert store.get(fact.id).content == fact.content
        assert store.list_jobs() == []

        foreign = _event("foreign", "A different profile changed the project", "profile-b")
        store.append_episodic_event(foreign)
        foreign_proposal = proposal.model_copy(update={"evidence_ids": [foreign.id]})
        with pytest.raises(ValueError, match="outside the fact profile"):
            await rag.review(foreign_proposal)
        assert store.get(fact.id).content == fact.content
    finally:
        store.close()


@pytest.mark.asyncio
async def test_unchanged_review_preserves_content_timestamps_and_access(tmp_path: Path) -> None:
    embedder = SimpleEmbeddingProvider(dim=32)
    store = SQLiteStore(str(tmp_path / "unchanged.sqlite"))
    try:
        fact = _fact(embedder)
        fact.last_observed_at = datetime(2025, 12, 1, tzinfo=timezone.utc)
        store.store(fact)
        source = _event("source-same", fact.content)
        store.append_episodic_event(source)
        before = store.get(fact.id)
        proposal = FactProposal(
            content=fact.content,
            evidence_ids=[source.id],
            operation="correct",
            target_fact_id=fact.id,
            expected_version=1,
        )
        revised, changed = await MutableRAG(MockLLMProvider(), embedder, store).review(proposal)
        after = store.get(fact.id)
        assert changed is False
        assert revised == before
        assert after.updated_at == before.updated_at
        assert after.last_observed_at == before.last_observed_at
        assert after.access_count == before.access_count
        assert store.list_jobs(status="completed")[0].result["decision"] == "unchanged"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_empty_or_unparseable_review_output_leaves_fact_untouched(tmp_path: Path) -> None:
    embedder = SimpleEmbeddingProvider(dim=32)
    store = SQLiteStore(str(tmp_path / "malformed.sqlite"))
    try:
        fact = _fact(embedder)
        source = _event("source-malformed", "The project still uses Python")
        store.store(fact)
        store.append_episodic_event(source)
        rag = MutableRAG(
            MockLLMProvider(responses={"has this stored memory": "{}"}), embedder, store
        )
        result, changed = await rag.reconsolidate(fact, "ignored query", evidence_refs=[source.id])
        assert changed is False
        assert result == fact
        assert store.get(fact.id).content == fact.content
        assert store.list_jobs() == []
    finally:
        store.close()


@pytest.mark.asyncio
async def test_claimed_review_job_can_resume_after_store_reopen(tmp_path: Path) -> None:
    embedder = SimpleEmbeddingProvider(dim=32)
    db_path = tmp_path / "restart.sqlite"
    store = SQLiteStore(str(db_path))
    fact = _fact(embedder)
    source = _event("source-restart", "The project switched to Rust")
    store.store(fact)
    store.append_episodic_event(source)
    job = store.enqueue_job(
        DurableJob(
            kind="reconsolidate_review",
            payload={
                "content": "The project uses Rust",
                "evidence_ids": [source.id],
                "operation": "correct",
                "target_fact_id": fact.id,
                "expected_version": 1,
            },
        )
    )
    claimed = store.claim_jobs(owner="maintenance", job_id=job.id)[0]
    store.close()
    reopened = SQLiteStore(str(db_path))
    try:
        revised, changed = await MutableRAG(MockLLMProvider(), embedder, reopened).process_job(
            claimed
        )
        assert changed is True
        assert revised.version == 2
        assert reopened.get(fact.id).content == "The project uses Rust"
        assert reopened.list_jobs(status="completed")[0].id == job.id
    finally:
        reopened.close()
