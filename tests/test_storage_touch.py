"""
tests/test_storage_touch.py — Fast access-count persistence helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mnemos.types import MemoryChunk
from mnemos.utils.storage import InMemoryStore, SQLiteStore


@pytest.mark.parametrize("store_factory", [InMemoryStore, SQLiteStore])
def test_store_touch_persists_access_count_and_updated_at(store_factory, tmp_path) -> None:
    if store_factory is SQLiteStore:
        store = store_factory(str(tmp_path / "touch.sqlite"))
    else:
        store = store_factory()

    chunk = MemoryChunk(
        id="alpha",
        content="project memory",
        embedding=[1.0, 0.0, 0.0],
        metadata={"scope": "project", "scope_id": "repo-alpha"},
    )
    store.store(chunk)

    touched_at = datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)
    assert store.touch(chunk.id, access_count=3, updated_at=touched_at) is True

    touched = store.get(chunk.id)
    assert touched is not None
    assert touched.access_count == 3
    assert touched.updated_at == touched_at

    close = getattr(store, "close", None)
    if callable(close):
        close()
