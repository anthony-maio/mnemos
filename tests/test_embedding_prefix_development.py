from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.local_memory.embedding_prefix_development_v1 import (
    CASE_IDS,
    MANIFEST_PATH,
    _arm,
    _rank,
    load_cases,
    run_paired_diagnostic,
)
from mnemos.utils.embeddings import EmbeddingProvider, PrefixEmbeddingProvider


class RecordingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def dim(self) -> int:
        return 3

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        # Stable, role-sensitive test vectors without a model request.
        value = float(sum(ord(char) for char in text) % 97)
        return [value, float(len(text)), 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def test_prefix_provider_forwards_role_prefix_once() -> None:
    provider = RecordingProvider()
    wrapped = PrefixEmbeddingProvider(
        provider,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )

    wrapped.embed_query("same text")
    wrapped.embed_document("same text")

    assert provider.calls == ["search_query: same text", "search_document: same text"]


def test_paired_report_has_five_cases_scope_controls_and_rank_arithmetic(tmp_path: Path) -> None:
    output = tmp_path / "paired.json"
    providers: list[RecordingProvider] = []

    def factory(_label: str) -> RecordingProvider:
        provider = RecordingProvider()
        providers.append(provider)
        return provider

    report = run_paired_diagnostic(
        factory,
        output,
        model_metadata={"provider": "fake", "model": "offline-test"},
        code_sha="test-code",
    )

    assert report["dataset"]["case_ids"] == list(CASE_IDS)
    assert {arm["label"] for arm in report["arms"]} == {"unprefixed", "nomic_task"}
    assert len(providers) == 2
    assert any(call.startswith("search_query: ") for call in providers[1].calls)
    assert any(call.startswith("search_document: ") for call in providers[1].calls)
    assert all(not call.startswith("search_") for call in providers[0].calls)
    for arm in report["arms"]:
        assert len(arm["cases"]) == 5
        for case in arm["cases"]:
            assert case["eligible_fact_count"] == 36
            assert case["target"]["rank"] == case["ranked_fact_ids"].index(case["target"]["id"]) + 1
            assert case["seed"]["rank"] == case["ranked_fact_ids"].index(case["seed"]["id"]) + 1
            assert case["bridge"]["rank"] == case["ranked_fact_ids"].index(case["bridge"]["id"]) + 1
            assert case["scope_controls"]["present_in_ranked"] == []
            assert len(case["scored_rows"]) >= 36
            assert all("cosine" in row for row in case["scored_rows"])
    assert json.loads(output.read_text(encoding="utf-8"))["binding"] == report["binding"]


def test_changed_binding_and_resume_are_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "paired.json"
    kwargs: dict[str, Any] = {
        "model_metadata": {"provider": "fake", "model": "offline-test"},
        "code_sha": "test-code",
    }
    run_paired_diagnostic(lambda _label: RecordingProvider(), output, **kwargs)
    with pytest.raises(ValueError, match="output exists"):
        run_paired_diagnostic(lambda _label: RecordingProvider(), output, **kwargs)
    with pytest.raises(ValueError, match="output exists"):
        run_paired_diagnostic(
            lambda _label: RecordingProvider(),
            output,
            model_metadata={"provider": "fake", "model": "changed"},
            code_sha="test-code",
        )


def test_harness_uses_fixed_development_manifest(tmp_path: Path) -> None:
    assert MANIFEST_PATH.is_file()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["graph_cases"][0]["split"] = "heldout"
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="heldout"):
        load_cases(altered)

    manifest["graph_cases"][0]["split"] = "development"
    manifest["graph_cases"][0]["query"] = "changed query"
    altered.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset changed"):
        load_cases(altered)


def test_rank_uses_cosine_then_id_tie_break() -> None:
    assert _rank([1.0, 0.0], {"b": [0.0, 1.0], "a": [1.0, 0.0]}) == ["a", "b"]
    assert _rank([1.0, 0.0], {"b": [1.0, 0.0], "a": [1.0, 0.0]}) == ["a", "b"]


def test_foreign_exact_match_is_scored_but_cannot_enter_ranking() -> None:
    _, _, cases = load_cases()
    case = next(c for c in cases if c.get("scope_controls"))
    foreign_texts = {c["content"] for c in case["scope_controls"]}

    class ExactForeignProvider(RecordingProvider):
        def embed(self, text: str) -> list[float]:
            return (
                [1.0, 0.0, 0.0]
                if text in foreign_texts or text == case["query"]
                else [0.0, 1.0, 0.0]
            )

    result = _arm(
        ExactForeignProvider(),
        [case],
        label="scope-test",
        prefixes={"query_prefix": "", "document_prefix": "", "symmetric_prefix": ""},
    )
    observed = result["cases"][0]
    foreign_ids = {c["id"] for c in case["scope_controls"]}
    assert not foreign_ids.intersection(observed["ranked_fact_ids"])
    assert all(
        row["cosine"] == 1.0 and row["rank"] is None
        for row in observed["scored_rows"]
        if row["id"] in foreign_ids
    )
    assert all(row["cosine"] == 0.0 for row in observed["scored_rows"] if row["eligible"])
    assert observed["ranked_fact_ids"] == sorted(fact["id"] for fact in case["facts"])

    case["scope_controls"][0]["scope"] = dict(case["facts"][0]["scope"])
    with pytest.raises(ValueError, match="unexpectedly eligible scope control"):
        _arm(
            ExactForeignProvider(),
            [case],
            label="scope-test",
            prefixes={"query_prefix": "", "document_prefix": "", "symmetric_prefix": ""},
        )


class InvalidProvider(RecordingProvider):
    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.0, float("nan"), 0.0]


def test_invalid_vectors_are_rejected_before_scoring(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite and nonzero"):
        run_paired_diagnostic(
            lambda _label: InvalidProvider(),
            tmp_path / "invalid.json",
            model_metadata={"provider": "fake"},
            code_sha="test-code",
        )
