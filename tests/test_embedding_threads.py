"""Local embedding resources are explicit without changing vector identity."""

import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mnemos.runtime import build_embedder_from_settings, embedding_identity
from mnemos.settings import AppSettings, load_settings, save_settings
from mnemos.utils.embeddings import FastembedProvider


def stub_fastembed(monkeypatch):
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(embed=lambda texts: iter([[1.0, 0.0] for _ in texts]))

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=create))
    return seen


def test_default_local_embedding_does_not_create_an_automatic_worker_pool(monkeypatch):
    seen = stub_fastembed(monkeypatch)
    provider = FastembedProvider()
    assert provider.embed("retained source") == [1.0, 0.0]
    assert seen[0]["threads"] == 1


@pytest.mark.parametrize("threads", [0, 1, 4])
def test_operator_thread_budget_reaches_the_onnx_provider(monkeypatch, threads):
    seen = stub_fastembed(monkeypatch)
    settings = AppSettings.model_validate(
        {"embedding": {"provider": "fastembed", "threads": threads}}
    )
    provider = build_embedder_from_settings(settings)
    assert provider.embed("retained source") == [1.0, 0.0]
    assert seen[0]["threads"] == threads


@pytest.mark.parametrize("threads", [-1, True, 1.5, "4", None])
def test_invalid_direct_budget_fails_before_model_initialization(monkeypatch, threads):
    seen = stub_fastembed(monkeypatch)
    with pytest.raises(ValueError):
        FastembedProvider(threads=threads)
    assert seen == []
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"embedding": {"threads": threads}})


def test_global_project_and_environment_budget_precedence_survives_save(tmp_path):
    global_path = tmp_path / "global.toml"
    global_path.write_text('[embedding]\nprovider="fastembed"\nthreads=2\n')
    project_path = tmp_path / ".mnemos/mnemos.toml"
    project_path.parent.mkdir()
    project_path.write_text("[embedding]\nthreads=3\n")
    resolved = load_settings(
        global_config_path=global_path, cwd=tmp_path, env={"MNEMOS_EMBEDDING_THREADS": "4"}
    )
    assert resolved.settings.embedding.threads == 4
    save_settings(resolved.settings, global_path, scope="global")
    assert (
        load_settings(
            global_config_path=global_path, cwd=tmp_path, env={}
        ).settings.embedding.threads
        == 3
    )
    project_path.unlink()
    assert (
        load_settings(
            global_config_path=global_path, cwd=tmp_path, env={}
        ).settings.embedding.threads
        == 4
    )


def test_thread_budget_does_not_relabel_existing_embeddings():
    single = AppSettings.model_validate({"embedding": {"provider": "fastembed", "threads": 1}})
    automatic = AppSettings.model_validate({"embedding": {"provider": "fastembed", "threads": 0}})
    assert embedding_identity(single) == embedding_identity(automatic)
