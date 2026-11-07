"""A prospective study must reject changed labels before contacting a provider."""

from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.local_memory import support_temporal_comparison as temporal


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["labels", "duplicate_dataset"])
async def test_temporal_study_rejects_dataset_drift_before_provider_setup(
    tmp_path, monkeypatch, change
):
    root = Path(__file__).resolve().parents[1]
    original = root / "benchmarks/local_memory/fixtures/answer-support-development-v1.json"
    extra = root / "benchmarks/local_memory/fixtures/answer-support-temporal-development-v1.json"
    if change == "labels":
        changed = tmp_path / "changed.json"
        changed.write_text(
            extra.read_text().replace(
                '"expected_fact_ids": ["previous-owner"]', '"expected_fact_ids": []'
            ),
            encoding="utf-8",
        )
        inputs = [original, changed]
        error = "Dataset digest"
    else:
        inputs = [original, original]
        error = "two declared datasets"

    def no_provider(*args, **kwargs):
        pytest.fail("Dataset validation must finish before any provider setup")

    monkeypatch.setattr(temporal.httpx, "AsyncClient", no_provider)
    monkeypatch.setattr(temporal, "FastembedProvider", no_provider)
    with pytest.raises(ValueError, match=error):
        await temporal.run(
            Namespace(
                source_root=str(root),
                output=str(tmp_path / "attempt"),
                dataset=[str(p) for p in inputs],
                endpoint="http://127.0.0.1:11436",
                protocol=str(root / "docs/plans/2026-11-06-answer-support-temporal-v4.md"),
                embedding_cache=str(tmp_path / "unused-cache"),
            )
        )
