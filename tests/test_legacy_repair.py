from __future__ import annotations

from pathlib import Path

from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore
import mnemos.cli as cli_module


def test_repair_preserves_unassigned_rows_and_profile_boundaries(tmp_path: Path) -> None:
    source_path = tmp_path / "source.sqlite"
    target_path = tmp_path / "target.sqlite"
    source = SQLiteStore(db_path=str(source_path))
    try:
        source.store(MemoryChunk(id="a-legacy", content="same fact", metadata={}))
        source.store(MemoryChunk(id="b-legacy", content="same fact", metadata={}))
        source.store(
            MemoryChunk(
                id="c-profiled",
                content="same fact",
                metadata={"scope": "global", "user_profile_id": "alice"},
            )
        )
        source.store(
            MemoryChunk(
                id="d-invalid-scope", content="unassigned detail", metadata={"scope": "bogus"}
            )
        )
    finally:
        source.close()

    summary = cli_module._repair_sqlite_db(str(source_path), str(target_path), dry_run=False)

    repaired = SQLiteStore(db_path=str(target_path))
    try:
        chunks = {chunk.id: chunk for chunk in repaired.get_all()}
        assert summary["source_chunks"] == 4
        assert summary["canonical_chunks"] == 3
        assert set(chunks) == {"a-legacy", "c-profiled", "d-invalid-scope"}
        assert chunks["a-legacy"].metadata.get("user_profile_id") is None
        assert chunks["a-legacy"].metadata.get("scope") is None
        assert chunks["d-invalid-scope"].metadata["scope"] == "bogus"
        assert chunks["c-profiled"].metadata["user_profile_id"] == "alice"
        assert chunks["a-legacy"].canonical_key != chunks["c-profiled"].canonical_key
        assert [chunk.id for chunk in repaired.list_legacy_chunks(limit=10)] == [
            "a-legacy",
            "d-invalid-scope",
        ]
    finally:
        repaired.close()


def test_repair_does_not_merge_legacy_owners_or_promote_invalid_scope(tmp_path: Path) -> None:
    source_path = tmp_path / "legacy-owners.sqlite"
    target_path = tmp_path / "repaired.sqlite"
    ownership = {
        "alias-alice": {"profile_id": "alice", "scope": "global"},
        "alias-bob": {"profile_id": "bob", "scope": "global"},
        "alias-case-sensitive": {"profile_id": "Alice", "scope": "global"},
        "unassigned-project-a": {"scope": "project", "scope_id": "a"},
        "unassigned-project-b": {"scope": "project", "scope_id": "b"},
        "invalid-alice": {"user_profile_id": "alice", "scope": "invalid"},
        "global-alice": {"user_profile_id": "alice", "scope": "global"},
    }
    source = SQLiteStore(db_path=str(source_path))
    try:
        for fact_id, metadata in ownership.items():
            source.store(MemoryChunk(id=fact_id, content="same fact", metadata=metadata))
    finally:
        source.close()
    summary = cli_module._repair_sqlite_db(str(source_path), str(target_path), dry_run=False)
    assert summary["duplicates_collapsed"] == 0
    repaired = SQLiteStore(db_path=str(target_path))
    try:
        chunks = repaired.get_all()
        assert len(chunks) == len(ownership)
        assert len({chunk.canonical_key for chunk in chunks}) == len(ownership)
        for chunk in chunks:
            assert {key: chunk.metadata[key] for key in ownership[chunk.id]} == ownership[chunk.id]
        assert {chunk.id for chunk in repaired.list_legacy_chunks()} == {
            "unassigned-project-a",
            "unassigned-project-b",
        }
    finally:
        repaired.close()
