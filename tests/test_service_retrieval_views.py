"""Raw captured evidence remains reviewable without becoming ordinary recall."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from mnemos.contracts import CaptureEvent, RetrievalRequest
from tests.test_memory_service import service


@pytest.mark.asyncio
async def test_service_separates_capture_and_memory_without_erasing_source(tmp_path):
    memory, context = service(tmp_path)
    for name in type(memory.engine.config.modules).model_fields:
        setattr(memory.engine.config.modules, name, False)
    try:
        await memory.dispatch(
            "remember",
            {"context": context.model_dump(), "content": "The release label is maple-heron."},
        )
        captured = memory.capture(
            CaptureEvent(
                event_id="old-release-conversation",
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="assistant",
                content="Earlier we used cedar-otter as the release label.",
                source_kind="conversation",
            )
        )
        await memory.maintenance.run_once(force=True)
        facts = memory.engine.store.get_all()
        assert len(facts) == 2
        for fact in facts:
            fact.embedding = memory.engine.embedder.embed_query("release label")
            memory.engine.store.update(fact.id, fact)
        ids_before = {fact.id for fact in facts}
        payload = {
            "context": context.model_dump(),
            "query": "release label",
            "top_k": 10,
            "max_context_tokens": 4000,
        }
        ordinary = await memory.dispatch("retrieve", payload)
        assert ordinary["view"] == "memory"
        assert len(ordinary["facts"]) == 1
        assert ordinary["facts"][0]["content"] == "The release label is maple-heron."
        assert ordinary["captured_evidence"] == []
        assert ordinary["evidence_context"] == ""
        assert "cedar-otter" not in ordinary["context"]

        review = await memory.dispatch("retrieve", {**payload, "view": "captured_evidence"})
        assert review["view"] == "captured_evidence"
        assert review["facts"] == [] and review["context"] == ""
        assert len(review["captured_evidence"]) == 1
        evidence = review["captured_evidence"][0]
        assert evidence["assertion_status"] == "unknown"
        assert evidence["support_available"]
        assert evidence["sources"][0]["original_source_status"] == "retained_locally"
        assert "cedar-otter" in review["evidence_context"]
        assert "captured conversation" in review["evidence_context"]
        assert "unverified" in review["evidence_context"]
        for result in (ordinary, review):
            assert result["schema_version"] == 1
            assert result["token_count_method"] == "utf8_bytes_upper_bound"
            assert result["context_token_upper_bound"] == len(
                (result["context"] + result["evidence_context"]).encode("utf-8")
            )
        assert {fact.id for fact in memory.engine.store.get_all()} == ids_before
        assert memory.engine.store.get_episodic_event(captured["event_id"]) is not None
    finally:
        memory.engine.store.close()


@pytest.mark.parametrize("view", ["all", "unknown", None, 1, True, [], {}])
def test_retrieval_request_rejects_invalid_views(tmp_path, view):
    memory, context = service(tmp_path)
    try:
        with pytest.raises(ValidationError):
            RetrievalRequest(query="release label", context=context, view=view)
    finally:
        memory.engine.store.close()
