"""A prospective study must reject changed labels before contacting a provider."""

from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.local_memory import support_temporal_comparison as temporal


def test_contrast_uses_one_comparison_batch_and_preserves_prior_treatments():
    contrast = temporal.support_configuration("contrast_v5")
    assert contrast.protocol == "contrast_v5" and contrast.batching == "all"
    assert contrast.min_probability == 0.8 and contrast.timeout_seconds == 15
    for protocol in ("joint_v1", "qualified_v3", "checked_v4"):
        assert temporal.support_configuration(protocol).batching == "candidate"


@pytest.mark.asyncio
async def test_study_rejects_unknown_candidate_before_output_or_provider_setup(tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="Unsupported candidate support protocol"):
        await temporal.run(
            Namespace(
                endpoint="http://127.0.0.1:11437",
                candidate_protocol="unreviewed",
                source_root=str(Path(__file__).resolve().parents[1]),
                output=str(output),
                dataset=[],
            )
        )
    assert not output.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["labels", "duplicate_dataset"])
@pytest.mark.parametrize("protocol", ["qualified_v3", "contrast_v5"])
async def test_temporal_study_rejects_dataset_drift_before_provider_setup(
    tmp_path, monkeypatch, change, protocol
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
                candidate_protocol=protocol,
                source_root=str(root),
                output=str(tmp_path / "attempt"),
                dataset=[str(p) for p in inputs],
                endpoint="http://127.0.0.1:11436",
                protocol=str(root / "docs/plans/2026-11-06-answer-support-temporal-v4.md"),
                embedding_cache=str(tmp_path / "unused-cache"),
            )
        )
