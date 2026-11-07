"""Compact recall keeps literal facts and source references through the service boundary."""

import json

import pytest
from pydantic import ValidationError

from mnemos.contracts import RetrievalRequest
from mnemos.utils.reliability import MnemosTransientError
from tests.test_host_context_budget import service
from tests.test_retrieval_comparison import add_fact
from tests.test_retrieval_contract import FixedEmbedding


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["asserted", "conflict", "unknown"])
async def test_compact_preserves_complete_context_qualifiers_and_scope(tmp_path, status):
    memory, context = service(tmp_path, None)
    try:
        content = 'Juniper production retention is 30 days; staging remains 7. Literal "quote".\nNext line.'
        chunk = add_fact(memory, context, "ours", content)
        chunk.metadata["assertion_status"] = status
        memory.engine.store.update(chunk.id, chunk)
        add_fact(
            memory,
            context.model_copy(update={"project_id": "foreign"}),
            "foreign",
            "Juniper private",
        )
        request = RetrievalRequest(query="Juniper", context=context)
        full = await memory.retrieve(request)
        compact = await memory.retrieve(request.model_copy(update={"response_format": "compact"}))
        assert compact["context"] == full["context"]
        assert content in compact["context"]
        assert compact["references"][0]["id"] == "ours"
        assert compact["references"][0]["assertion_status"] == status
        assert compact["references"][0]["scope_id"] == full["facts"][0]["scope_id"]
        assert "facts" not in compact and "captured_evidence" not in compact
        assert (
            "content" not in compact["references"][0] and "scores" not in compact["references"][0]
        )
        assert "foreign" not in json.dumps(compact)
        assert compact["context_token_upper_bound"] == full["context_token_upper_bound"]
        assert compact["response_format"] == "compact" and "response_format" not in full
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_compact_capture_keeps_unverified_and_degraded_warnings(tmp_path):
    memory, context = service(tmp_path, None)
    try:
        chunk = add_fact(memory, context, "capture", "Juniper may use the old deployment rule.")
        chunk.metadata.update(assertion_status="unknown", source="captured_passthrough")
        memory.engine.store.update(chunk.id, chunk)

        class Offline(FixedEmbedding):
            def embed(self, text):
                raise MnemosTransientError("offline")

        memory.engine._embedder = Offline()
        result = await memory.retrieve(
            RetrievalRequest(
                query="Juniper",
                context=context,
                view="captured_evidence",
                response_format="compact",
            )
        )
        assert result["view"] == "captured_evidence" and result["context"] == ""
        assert "captured conversation; unverified, may be outdated" in result["evidence_context"]
        assert "lexical match only; semantic retrieval unavailable" in result["evidence_context"]
        assert result["degraded"] is True and result["retrieval"]
        assert result["references"][0]["assertion_status"] == "unknown"
    finally:
        await memory.close()


def test_unknown_response_format_is_not_silently_ignored(tmp_path):
    memory, context = service(tmp_path, None)
    try:
        with pytest.raises(ValidationError):
            RetrievalRequest(query="Juniper", context=context, response_format="summary")
    finally:
        memory.engine.store.close()
