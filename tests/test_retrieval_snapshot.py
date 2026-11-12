"""Recall must not emit a revision forgotten or changed before response assembly."""

import pytest

from mnemos.context_budget import HostTokenizer
from mnemos.contracts import RetrievalRequest
from mnemos.utils.storage import SQLiteStore
from tests.test_related_forget import _service, _seed_overlap


@pytest.mark.asyncio
@pytest.mark.parametrize("view", ["memory", "captured_evidence"])
@pytest.mark.parametrize("response_format", ["full", "compact"])
async def test_forget_after_engine_selection_cannot_return_deleted_content(
    tmp_path, monkeypatch, view, response_format
):
    memory, context, _, _ = _service(tmp_path)
    writer = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        assertion, target, captured = await _seed_overlap(memory, context)
        selected = target if view == "memory" else captured
        original = memory.engine.retrieve

        async def retrieve_then_forget(*args, **kwargs):
            chunks = await original(*args, **kwargs)
            assert selected.id in [c.id for c in chunks]
            assert writer.forget_fact(selected.id, expected_version=selected.version)
            return chunks

        monkeypatch.setattr(memory.engine, "retrieve", retrieve_then_forget)
        result = await memory.retrieve(
            RetrievalRequest(
                query=assertion, context=context, view=view, response_format=response_format
            )
        )
        records = result.get("references", result.get("facts") or result.get("captured_evidence"))
        assert records == []
        assert result["context"] == result["evidence_context"] == ""
        assert result["context_byte_count"] == result["context_token_upper_bound"] == 0
        assert not memory.engine.store._conn.in_transaction
    finally:
        writer.close()
        await memory.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["correction", "profile", "project", "view", "status", "version_only"]
)
async def test_changed_candidate_cannot_reuse_earlier_rank_or_context(
    tmp_path, monkeypatch, change
):
    memory, context, registry, _ = _service(tmp_path)
    writer_memory, _, _, _ = _service(tmp_path, registry)
    try:
        assertion, target, _ = await _seed_overlap(memory, context)
        original = memory.engine.retrieve

        async def retrieve_then_change(*args, **kwargs):
            chunks = await original(*args, **kwargs)
            assert target.id in [c.id for c in chunks]
            if change == "correction":
                await writer_memory.dispatch(
                    "correct",
                    dict(
                        context=context.model_dump(mode="json"),
                        fact_id=target.id,
                        expected_version=target.version,
                        content="The current release channel is corrected-new.",
                        event_id="recall-race-correction",
                    ),
                )
                await writer_memory.dispatch("maintenance", {})
            else:
                changed = writer_memory.engine.store.get(target.id)
                if change == "profile":
                    changed.metadata["user_profile_id"] = "other-profile"
                elif change == "project":
                    changed.metadata["scope_id"] = "other-project"
                elif change == "view":
                    changed.metadata["source"] = "captured_passthrough"
                elif change == "status":
                    changed.metadata["assertion_status"] = "conflict"
                else:
                    changed.version += 1
                assert writer_memory.engine.store.update(target.id, changed)
            return chunks

        monkeypatch.setattr(memory.engine, "retrieve", retrieve_then_change)
        result = await memory.retrieve(RetrievalRequest(query=assertion, context=context))
        assert result["facts"] == []
        assert result["context"] == ""
        assert result["context_byte_count"] == result["context_token_upper_bound"] == 0
    finally:
        await writer_memory.close()
        await memory.close()


@pytest.mark.asyncio
async def test_forget_during_assembly_cannot_mix_fact_with_newer_empty_evidence(
    tmp_path, monkeypatch
):
    memory, context, _, _ = _service(tmp_path)
    writer = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        assertion, target, _ = await _seed_overlap(memory, context)
        request = RetrievalRequest(query=assertion, context=context)
        before = await memory.retrieve(request)
        assert before["facts"][0]["sources"]
        original_retrieve, original_get = memory.engine.retrieve, memory.engine.store.get
        selected = False
        deleted = []

        async def mark_selected(*args, **kwargs):
            nonlocal selected
            result = await original_retrieve(*args, **kwargs)
            selected = True
            return result

        def get_then_forget(identifier):
            chunk = original_get(identifier)
            if selected and identifier == target.id and not deleted:
                deleted.append(writer.forget_fact(target.id, expected_version=target.version))
            return chunk

        monkeypatch.setattr(memory.engine, "retrieve", mark_selected)
        monkeypatch.setattr(memory.engine.store, "get", get_then_forget)
        after = await memory.retrieve(request)
        assert deleted == [True]
        assert after["facts"] == before["facts"]
        assert after["context"] == before["context"]
        assert original_get(target.id) is None
        assert not memory.engine.store._conn.in_transaction
        assert (await memory.retrieve(request))["facts"] == []
    finally:
        writer.close()
        await memory.close()


@pytest.mark.asyncio
async def test_failed_assembly_releases_snapshot_without_committing_caller_work(
    tmp_path, monkeypatch
):
    memory, context, _, _ = _service(tmp_path)
    observer = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        assertion, _, _ = await _seed_overlap(memory, context)
        conn = memory.engine.store._conn
        conn.execute("CREATE TABLE caller_work (value TEXT)")
        conn.commit()
        original = memory.engine.retrieve

        async def retrieve_with_pending_work(*args, **kwargs):
            chunks = await original(*args, **kwargs)
            conn.execute("INSERT INTO caller_work VALUES ('pending')")
            return chunks

        def fail_evidence(_):
            assert conn.in_transaction
            raise RuntimeError("evidence unavailable")

        monkeypatch.setattr(memory.engine, "retrieve", retrieve_with_pending_work)
        monkeypatch.setattr(memory.engine.store, "list_fact_evidence", fail_evidence)
        with pytest.raises(RuntimeError, match="evidence unavailable"):
            await memory.retrieve(RetrievalRequest(query=assertion, context=context))
        assert conn.in_transaction
        assert conn.execute("SELECT value FROM caller_work").fetchall() == [("pending",)]
        assert observer._conn.execute("SELECT value FROM caller_work").fetchall() == []
        conn.rollback()
        assert not conn.in_transaction
    finally:
        observer.close()
        await memory.close()


@pytest.mark.asyncio
async def test_filtering_stale_fact_recounts_remaining_host_token_context(tmp_path, monkeypatch):
    from tests.test_host_context_budget import service
    from tests.test_retrieval_comparison import add_fact

    memory, context = service(
        tmp_path, HostTokenizer("words", "v1", lambda text: len(text.split()))
    )
    writer = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        add_fact(memory, context, "a", "Juniper uses Python for its backend.")
        add_fact(memory, context, "b", "Juniper uses SQLite for local storage.")
        original = memory.engine.retrieve

        async def retrieve_then_forget(*args, **kwargs):
            chunks = await original(*args, **kwargs)
            assert [c.id for c in chunks] == ["a", "b"]
            assert writer.forget_fact("a", expected_version=1)
            return chunks

        monkeypatch.setattr(memory.engine, "retrieve", retrieve_then_forget)
        result = await memory.retrieve(
            RetrievalRequest(query="Juniper", context=context, max_context_tokens=20)
        )
        assert [f["id"] for f in result["facts"]] == ["b"]
        assert result["context_token_count"] == len(result["context"].split())
        assert result["context_byte_count"] == len(result["context"].encode())
        assert result["token_count_method"] == "host_tokenizer"
    finally:
        writer.close()
        await memory.close()
