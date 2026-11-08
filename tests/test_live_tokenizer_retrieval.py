"""The measurement path preserves real service packing and snapshot isolation."""

import pytest

from benchmarks.local_memory.live_tokenizer_retrieval import (
    accepted_projection,
    file_hash,
    observe_retrieval,
)
from mnemos.context_budget import HostTokenizer
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from tests.test_memory_service import service


async def scenario(tmp_path):
    memory, context = service(tmp_path)
    try:
        for name in type(memory.engine.config.modules).model_fields:
            setattr(memory.engine.config.modules, name, False)
        await memory.dispatch(
            "remember",
            {"context": context.model_dump(), "content": "The release label is maple-heron."},
        )
        await memory.maintenance.run_once(force=True)
        case = {
            "database": str(tmp_path / "memory.sqlite"),
            "registry": memory.registry.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "config": memory.engine.config.model_dump(mode="json"),
            "logical_sha256": accepted_projection(memory.engine.store),
        }
    finally:
        memory.engine.store.close()
    case["database_sha256"] = file_hash(tmp_path / "memory.sqlite")
    return case


async def test_native_count_and_source_identity_reach_service_response(tmp_path):
    case = await scenario(tmp_path)
    calls = []

    def count(text):
        calls.append(text)
        return 7

    observation = await observe_retrieval(
        case,
        tmp_path / "native.sqlite",
        query="release label",
        budget=10,
        embedder=SimpleEmbeddingProvider(),
        tokenizer=HostTokenizer("test-native", "r1", count),
    )
    result = observation["result"]
    assert len(result["facts"]) == 1 and result["facts"][0]["sources"]
    assert result["context"] == calls[-1]
    assert result["context_token_count"] == 7
    assert result["token_count_method"] == "host_tokenizer"
    assert result["tokenizer_revision"] == "r1"
    assert observation["candidate_traces"]
    assert file_hash(tmp_path / "memory.sqlite") == case["database_sha256"]


async def test_failing_callback_matches_service_byte_fallback(tmp_path):
    case = await scenario(tmp_path)

    def broken(text):
        raise RuntimeError("Injected unavailability")

    common = dict(query="release label", budget=2048, embedder=SimpleEmbeddingProvider())
    normal = await observe_retrieval(case, tmp_path / "bytes.sqlite", tokenizer=None, **common)
    failed = await observe_retrieval(
        case,
        tmp_path / "failed.sqlite",
        tokenizer=HostTokenizer("test-native", "r1", broken),
        **common,
    )
    a, b = normal["result"], failed["result"]
    assert a["context"] == b["context"] and a["facts"] == b["facts"]
    assert b["context_token_count"] is None
    assert b["tokenizer_fallback_reason"] == "tokenizer_error"


async def test_existing_destination_is_never_overwritten(tmp_path):
    case = await scenario(tmp_path)
    target = tmp_path / "existing.sqlite"
    target.write_bytes(b"preserve this")
    with pytest.raises(ValueError, match="new database"):
        await observe_retrieval(
            case,
            target,
            query="release",
            budget=2048,
            embedder=SimpleEmbeddingProvider(),
            tokenizer=None,
        )
    assert target.read_bytes() == b"preserve this"
