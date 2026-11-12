"""Evidence inspection must represent one database state during other writes."""

from copy import deepcopy
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from mnemos.utils.storage import SQLiteStore
from mnemos.jobs import FactNotFoundError
from mnemos.types import MemoryChunk
from tests.test_related_forget import _service, _seed_overlap


@pytest.mark.asyncio
async def test_concurrent_forget_cannot_strip_evidence_from_an_older_inspection(
    tmp_path, monkeypatch
):
    memory, context, _, _ = _service(tmp_path)
    writer = None
    try:
        _, target, _ = await _seed_overlap(memory, context)
        payload = {"context": context.model_dump(mode="json"), "fact_id": target.id}
        before = await memory.dispatch("inspect", payload)
        assert before["evidence"] and before["original_source_status"] == "retained_locally"
        writer = SQLiteStore(str(tmp_path / "memory.sqlite"))
        original_get = memory.engine.store.get
        deleted = []

        def get_then_forget(identifier):
            fact = original_get(identifier)
            if identifier == target.id and not deleted:
                deleted.append(writer.forget_fact(target.id, expected_version=target.version))
            return fact

        monkeypatch.setattr(memory.engine.store, "get", get_then_forget)
        inspected = await memory.dispatch("inspect", deepcopy(payload))
        assert deleted == [True]
        assert inspected == before
        assert original_get(target.id) is None
        assert not memory.engine.store._conn.in_transaction
    finally:
        if writer is not None:
            writer.close()
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_concurrent_correction_cannot_mix_fact_and_revision_versions(tmp_path, monkeypatch):
    memory, context, registry, _ = _service(tmp_path)
    writer_memory = None
    try:
        _, target, _ = await _seed_overlap(memory, context)
        payload = {"context": context.model_dump(mode="json"), "fact_id": target.id}
        before = await memory.dispatch("inspect", payload)
        writer_memory, _, _, _ = _service(tmp_path, registry)
        original_get = memory.engine.store.get
        corrected = []

        async def correct():
            await writer_memory.dispatch(
                "correct",
                {
                    **payload,
                    "expected_version": target.version,
                    "content": "The project release channel is corrected-current.",
                    "event_id": "concurrent-inspection-correction",
                },
            )
            await writer_memory.dispatch("maintenance", {})

        def get_then_correct(identifier):
            fact = original_get(identifier)
            if identifier == target.id and not corrected:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(lambda: asyncio.run(correct())).result(timeout=5)
                corrected.append(True)
            return fact

        monkeypatch.setattr(memory.engine.store, "get", get_then_correct)
        inspected = await memory.dispatch("inspect", payload)
        assert corrected == [True]
        assert inspected == before
        assert original_get(target.id).version == target.version + 1
        assert not memory.engine.store._conn.in_transaction
    finally:
        if writer_memory is not None:
            writer_memory.engine.store.close()
        memory.engine.store.close()


@pytest.mark.parametrize("fail", [False, True])
def test_snapshot_preserves_caller_transaction_on_success_and_failure(tmp_path, fail):
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    observer = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        store._conn.execute("CREATE TABLE caller_work (value TEXT)")
        store._conn.commit()
        store._conn.execute("INSERT INTO caller_work VALUES ('pending')")
        assert store._conn.in_transaction

        def inspect():
            with store.read_snapshot():
                with store.read_snapshot():
                    assert store._conn.execute("SELECT value FROM caller_work").fetchall() == [
                        ("pending",)
                    ]
                    if fail:
                        raise RuntimeError("inspection failed")

        if fail:
            with pytest.raises(RuntimeError, match="inspection failed"):
                inspect()
        else:
            inspect()
        assert store._conn.in_transaction
        assert store._conn.execute("SELECT value FROM caller_work").fetchall() == [("pending",)]
        assert observer._conn.execute("SELECT value FROM caller_work").fetchall() == []
        store._conn.rollback()
        assert store._conn.execute("SELECT value FROM caller_work").fetchall() == []
    finally:
        store.close()
        observer.close()


@pytest.mark.asyncio
async def test_failed_inspection_releases_snapshot_and_keeps_scope_guard(tmp_path, monkeypatch):
    memory, context, _, _ = _service(tmp_path)
    try:
        _, target, _ = await _seed_overlap(memory, context)
        payload = {"context": context.model_dump(mode="json"), "fact_id": target.id}

        def fail_evidence(identifier):
            assert memory.engine.store._conn.in_transaction
            raise RuntimeError("evidence read failed")

        monkeypatch.setattr(memory.engine.store, "list_fact_evidence", fail_evidence)
        with pytest.raises(RuntimeError, match="evidence read failed"):
            await memory.dispatch("inspect", payload)
        assert not memory.engine.store._conn.in_transaction
        with pytest.raises(FactNotFoundError):
            await memory.dispatch("inspect", {**payload, "fact_id": "absent"})
        assert not memory.engine.store._conn.in_transaction
        for index, scope in enumerate(
            [
                {"user_profile_id": "other-profile", "scope_id": context.project_id},
                {"user_profile_id": context.user_profile_id, "scope_id": "other-project"},
            ]
        ):
            foreign = MemoryChunk(
                id=f"foreign-{index}",
                content="Private foreign fact.",
                metadata={"scope": "project", **scope},
            )
            memory.engine.store.store(foreign)
            with pytest.raises(FactNotFoundError):
                await memory.dispatch("inspect", {**payload, "fact_id": foreign.id})
            assert not memory.engine.store._conn.in_transaction
        assert memory.engine.store.get(target.id) is not None
    finally:
        memory.engine.store.close()
