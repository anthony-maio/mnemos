"""Captured history must not masquerade as a current accepted assertion."""

from datetime import datetime, timezone

import pytest

from mnemos.contracts import CaptureEvent
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.retrieval import assertion_status, render_context_chunk
from mnemos.types import MemoryChunk
from mnemos.utils.storage import InMemoryStore, SQLiteStore
from tests.test_memory_service import service
from tests.test_retrieval_contract import make_engine


@pytest.mark.asyncio
async def test_corrected_fact_and_captured_history_keep_distinct_authority_after_restart(tmp_path):
    memory, context = service(tmp_path)
    for name in type(memory.engine.config.modules).model_fields:
        setattr(memory.engine.config.modules, name, False)
    old = "The release label is cedar-otter."
    new = "The release label is maple-heron."
    try:
        await memory.dispatch("remember", {"context": context.model_dump(), "content": old})
        await memory.maintenance.run_once(force=True)
        target = memory.engine.store.get_all()[0]
        captured_ids = []
        for role, text in (
            ("user", 'Please remember this quoted convention: "' + old + '"'),
            ("assistant", old),
        ):
            result = memory.capture(
                CaptureEvent(
                    event_id="captured-" + role,
                    context=context,
                    occurred_at=datetime.now(timezone.utc),
                    role=role,
                    content=text,
                    source_kind="conversation",
                )
            )
            captured_ids.append(result["event_id"])
        await memory.maintenance.run_once(force=True)
        await memory.dispatch(
            "correct",
            {
                "context": context.model_dump(),
                "fact_id": target.id,
                "expected_version": 1,
                "content": new,
            },
        )
        await memory.maintenance.run_once(force=True)
        # A new explicit assertion containing the old text remains legitimate.
        await memory.dispatch(
            "remember", {"context": context.model_dump(), "content": "Archive convention: " + old}
        )
        await memory.maintenance.run_once(force=True)
        engine = memory.engine
        engine.store.close()
        memory = LocalMemoryService(
            MnemosEngine(
                config=engine.config,
                store=SQLiteStore(str(tmp_path / "memory.sqlite")),
                llm=engine.llm,
                embedder=engine.embedder,
            ),
            memory.registry,
        )
        # Fix candidate similarities only; exercise actual service/storage/rendering.
        for fact in memory.engine.store.get_all():
            fact.embedding = memory.engine.embedder.embed_query("release label")
            memory.engine.store.update(fact.id, fact)
        result = await memory.dispatch(
            "retrieve",
            {
                "context": context.model_dump(),
                "query": "release label",
                "top_k": 10,
                "max_context_tokens": 4000,
            },
        )
        returned = {fact["id"]: fact for fact in result["facts"]}
        assert returned[target.id]["content"] == new
        assert returned[target.id]["version"] == 2
        assert returned[target.id]["assertion_status"] == "asserted"
        assert len(returned) == 4
        raw = [
            fact
            for fact in returned.values()
            if fact["id"] != target.id and not fact["content"].startswith("Archive convention:")
        ]
        assert len(raw) == 2
        assert all(fact["assertion_status"] == "unknown" for fact in raw)
        for fact in raw:
            line = next(line for line in result["context"].splitlines() if fact["id"] in line)
            assert "captured conversation" in line and "unverified" in line
            assert fact["support_available"]
        assert any(
            f["content"].startswith("Archive convention:") and f["assertion_status"] == "asserted"
            for f in returned.values()
        )
        for event_id in captured_ids:
            assert memory.engine.store.get_episodic_event(event_id) is not None
        assert result["context_token_upper_bound"] == len(result["context"].encode("utf-8"))
    finally:
        memory.engine.store.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("explicit_first", [True, False])
def test_identical_capture_and_explicit_fact_retain_explicit_authority(
    tmp_path, backend, explicit_first
):
    store = InMemoryStore() if backend == "memory" else SQLiteStore(str(tmp_path / "merge.sqlite"))
    scope = {"scope": "project", "scope_id": "a", "user_profile_id": "p"}
    explicit = MemoryChunk(
        content="Use SQLite",
        embedding=[1.0, 0.0],
        version=2 if explicit_first else 1,
        metadata={
            **scope,
            "source": "explicit_correction" if explicit_first else "explicit_remember",
            "assertion_status": "asserted",
            "source_event_ids": ["explicit-event"],
            "source_roles": ["user"],
            "encoding_diagnostics": {},
            "host": "claude-code",
            "session_id": "correction-session",
        },
    )
    captured = MemoryChunk(
        content="Use SQLite",
        embedding=[1.0, 0.0],
        metadata={
            **scope,
            "source": "captured_passthrough",
            "assertion_status": "unknown",
            "source_event_ids": ["captured-event"],
            "source_roles": ["user"],
            "encoding_diagnostics": {"surprisal_status": "disabled"},
            "host": "codex",
            "session_id": "captured-session",
        },
    )
    first, second = (explicit, captured) if explicit_first else (captured, explicit)
    try:
        stored = store.upsert_canonical(first)
        merged = store.upsert_canonical(second)
        assert merged.id == stored.id
        assert len(store.get_all()) == 1
        assert assertion_status(merged) == "asserted"
        assert merged.metadata["source"] == (
            "explicit_correction" if explicit_first else "explicit_remember"
        )
        assert merged.version == (2 if explicit_first else 1)
        assert set(merged.metadata["source_event_ids"]) == {"explicit-event", "captured-event"}
        assert merged.metadata["source_roles"] == ["user"]
        assert merged.metadata["encoding_diagnostics"] == {}
        assert merged.metadata["host"] == "claude-code"
        assert merged.metadata["session_id"] == "correction-session"
        assert "unverified" not in render_context_chunk(merged)
    finally:
        if isinstance(store, SQLiteStore):
            store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["captured_passthrough", "legacy"])
async def test_unknown_context_qualification_counts_toward_retrieval_budget(source):
    engine = make_engine()
    engine.store.clear()
    fact = MemoryChunk(
        id="unreviewed",
        content="Use Python",
        embedding=[1.0, 0.0],
        metadata={
            "scope": "project",
            "scope_id": "default",
            "source": source,
            "assertion_status": "unknown",
        },
    )
    engine.store.store(fact)
    unqualified = f"[{fact.id} version {fact.version}] {fact.content}\n"
    assert not await engine.retrieve("Python", max_context_tokens=len(unqualified.encode()))
    assert "unverified" in render_context_chunk(fact)
    assert await engine.retrieve(
        "Python", max_context_tokens=len(render_context_chunk(fact).encode())
    )
