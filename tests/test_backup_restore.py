from pathlib import Path

import pytest

from mnemos.backup import create_backup, restore_backup
from mnemos.utils.storage import SQLiteStore
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.reindex import build_generation, activate_generation
from mnemos.types import MemoryChunk
from tests.test_memory_service import service


@pytest.mark.asyncio
async def test_offline_reindex_requires_exclusive_owner_and_backs_up_activation(
    tmp_path, monkeypatch
):
    from mnemos.reindex import reindex_configured_store
    from mnemos.service import _ProcessLock
    from mnemos.settings import AppSettings, save_settings, load_settings
    from mnemos.runtime import embedding_identity, build_mnemos_config_from_settings

    monkeypatch.delenv("MNEMOS_SQLITE_PATH", raising=False)  # the config names the database

    database = tmp_path / "memory.sqlite"
    config = tmp_path / "mnemos.toml"
    settings = AppSettings(
        embedding={"provider": "simple", "dim": 8},
        storage={"sqlite_path": str(database)},
        modules={"surprisal": False, "sleep": False},
        retrieval={"profile": "biomimetic", "relevance_floor": 0.4},
    )
    save_settings(settings, config, scope="global")
    resolved = load_settings(env={"MNEMOS_CONFIG_PATH": str(config)}, cwd=tmp_path)
    runtime_config = build_mnemos_config_from_settings(resolved.settings)
    assert runtime_config.modules.surprisal is False
    assert runtime_config.modules.sleep is False
    assert runtime_config.retrieval.profile == "biomimetic"
    assert runtime_config.retrieval.relevance_floor == 0.4
    monkeypatch.setenv("MNEMOS_CONFIG_PATH", str(config))
    monkeypatch.setenv("MNEMOS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMOS_RUNTIME_DESCRIPTOR", str(tmp_path / "runtime.json"))
    monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(tmp_path / "scopes.json"))
    store = SQLiteStore(str(database))
    store.bind_embedding_identity({"provider": "old-test-provider"})
    store.store(MemoryChunk(content="Preserve this fact", embedding=[1.0, 0.0]))
    store.close()
    lock = _ProcessLock(tmp_path / "runtime.lock")
    assert lock.acquire()
    try:
        with pytest.raises(ValueError, match="ownership lock"):
            await reindex_configured_store(activate=True)
    finally:
        lock.release()
    staged = await reindex_configured_store()
    assert staged["ready"] and not staged["activated"]
    store = SQLiteStore(str(database))
    assert len(store.get_all()[0].embedding) == 2
    store.close()
    activated = await reindex_configured_store(activate=True)
    assert Path(activated["backup"]).is_file()
    store = SQLiteStore(str(database))
    try:
        store.bind_embedding_identity(embedding_identity(settings))
        assert len(store.get_all()[0].embedding) == 8
    finally:
        store.close()


@pytest.mark.asyncio
async def test_embedding_identity_and_resumable_activation_preserve_previous_index(tmp_path):
    store = SQLiteStore(str(tmp_path / "reindex.sqlite"))
    old_identity = {"provider": "fixture", "model": "old"}
    new_identity = {"provider": "fixture", "model": "new-same-dimension"}
    old, new = SimpleEmbeddingProvider(dim=8, seed=1), SimpleEmbeddingProvider(dim=8, seed=2)
    try:
        store.bind_embedding_identity(old_identity)
        for index in range(3):
            content = f"Fact {index}"
            store.store(MemoryChunk(id=str(index), content=content, embedding=old.embed(content)))
        with pytest.raises(ValueError, match="identity changed"):
            store.bind_embedding_identity(new_identity)
        before = store.get("0").embedding
        partial = await build_generation(store, new, new_identity, batch_size=1, max_batches=1)
        assert partial["missing"] == 2
        assert store.get("0").embedding == before
        with pytest.raises(ValueError, match="incomplete"):
            activate_generation(store, partial["generation_id"])
        assert store.get("0").embedding == before
        resumed = await build_generation(store, new, new_identity, batch_size=1, max_batches=1)
        assert resumed["missing"] == 1
        ready = await build_generation(store, new, new_identity)
        assert ready["ready"] is True
        # A concurrent revision invalidates its staged vector without changing the old index.
        changed = store.get("2").reconsolidate("A corrected fact")
        store.update("2", changed)
        with pytest.raises(ValueError, match="facts changed"):
            activate_generation(store, ready["generation_id"])
        await build_generation(store, new, new_identity)
        assert activate_generation(store, ready["generation_id"])["activated"] is True
        assert store.get("0").embedding == new.embed("Fact 0")
        store.bind_embedding_identity(new_identity)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_wal_backup_restore_preserves_jobs_sources_and_newer_forget(tmp_path: Path):
    memory, context = service(tmp_path)
    try:
        await memory.dispatch(
            "remember", {"context": context.model_dump(), "content": "Use SQLite"}
        )
        await memory.maintenance.run_once(force=True)
        fact = memory.engine.store.get_all()[0]
        archive = tmp_path / "snapshot.zip"
        manifest = create_backup(memory.engine.store, archive, memory.registry)
        assert manifest["facts"] == 1
        first_path = tmp_path / "first-restore.sqlite"
        restore_backup(archive, first_path, current_registry=memory.registry)
        first = SQLiteStore(str(first_path))
        assert first.get(fact.id).content == fact.content
        assert first.list_fact_evidence(fact.id)
        assert first.list_jobs()[0].status == "completed"
        first.close()
        await memory.dispatch(
            "forget", {"context": context.model_dump(), "fact_id": fact.id, "expected_version": 1}
        )
        second_path = tmp_path / "second-restore.sqlite"
        restored_manifest, _ = restore_backup(
            archive,
            second_path,
            current_store=memory.engine.store,
            current_registry=memory.registry,
        )
        assert restored_manifest["current_suppressions_merged"] is True
        second = SQLiteStore(str(second_path))
        try:
            assert not second.get_all()
            assert not second.list_episodic_events()
            assert second.is_suppressed(fact_id=fact.id)
            assert second._conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        finally:
            second.close()
        with pytest.raises(FileExistsError):
            restore_backup(archive, second_path)
        registry = memory.registry.model_copy(deep=True)
        registry.projects[context.project_id].root = "unrelated-project"
        with pytest.raises(ValueError, match="project identity"):
            restore_backup(archive, tmp_path / "must-not-exist.sqlite", current_registry=registry)
        assert not (tmp_path / "must-not-exist.sqlite").exists()
    finally:
        memory.engine.store.close()
