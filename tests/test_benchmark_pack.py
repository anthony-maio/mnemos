"""Dataset resolution contracts; no ranking evaluation is performed."""

from pathlib import Path

import pytest

from mnemos import benchmark


def test_claim_driving_pack_loads_documents_and_queries() -> None:
    resolved = benchmark._resolve_dataset_pack("claim-driving")
    assert [name for name, _ in resolved] == [
        "contradiction-update-v1",
        "preference-drift-v1",
        "cross-project-scope-v1",
    ]
    for _, path in resolved:
        documents = benchmark.load_documents(path)
        queries = benchmark._load_queries(path, documents)
        assert documents and queries
        ids = {document.id for document in documents}
        assert all(query.relevant_ids and query.relevant_ids <= ids for query in queries)


def test_missing_pack_file_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(benchmark, "BENCHMARK_DATASET_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="contradiction-update-v1.jsonl"):
        benchmark._resolve_dataset_pack("claim-driving")


def test_unknown_pack_is_reported() -> None:
    with pytest.raises(ValueError, match="Unknown dataset pack"):
        benchmark._resolve_dataset_pack("not-a-pack")
