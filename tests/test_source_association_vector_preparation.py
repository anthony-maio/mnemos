import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, cast

import pytest

from benchmarks.local_memory.prepare_source_association_vectors import (
    DIMENSION,
    DIGEST,
    MODEL,
    prepare_source_association_vectors,
)


def _manifest() -> dict[str, Any]:
    return {
        "dataset_version": "source-association-development-v1",
        "cases": [
            {
                "id": "case-1",
                "query": "Where did the label come from?",
                "facts": [{"id": "fact-1", "content": "The label is cedar."}],
                "events": [{"id": "event-1", "content": "The label is cedar."}],
                "scope_controls": [
                    {"id": "foreign-1", "content": "A foreign project has another label."}
                ],
                "lifecycle_actions": [
                    {"operation": "correct", "new_content": "The label is maple."}
                ],
            }
        ],
    }


class FakeOllama:
    def __init__(
        self,
        *,
        bad_count: bool = False,
        bad_vector: str | None = None,
        changed_manifest: Path | None = None,
        wrong_digest: bool = False,
        show_drift: bool = False,
        show_modelfile_drift: bool = False,
        fail_embed: bool = False,
        tags_drift_after: bool = False,
    ):
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.bad_count = bad_count
        self.bad_vector = bad_vector
        self.changed_manifest = changed_manifest
        self.wrong_digest = wrong_digest
        self.show_drift = show_drift
        self.show_modelfile_drift = show_modelfile_drift
        self.fail_embed = fail_embed
        self.tags_drift_after = tags_drift_after
        self.show_calls = 0
        self.tags_calls = 0

    def __call__(
        self, method: str, url: str, body: bytes | None, timeout: int
    ) -> tuple[int, dict[str, str], bytes]:
        payload = cast(dict[str, Any] | None, json.loads(body) if body else None)
        self.calls.append((method, url, payload))
        if url.endswith("/api/version"):
            return 200, {}, b'{"version":"0.17.0"}'
        if url.endswith("/api/tags"):
            self.tags_calls += 1
            digest = (
                "wrong"
                if self.wrong_digest or (self.tags_drift_after and self.tags_calls == 2)
                else DIGEST
            )
            return 200, {}, json.dumps({"models": [{"name": MODEL, "digest": digest}]}).encode()
        if url.endswith("/api/show"):
            self.show_calls += 1
            family = "wrong-family" if self.show_drift and self.show_calls == 2 else "nomic-bert"
            return (
                200,
                {},
                json.dumps(
                    {
                        "model": MODEL,
                        "details": {"family": family},
                        "modelfile": (
                            "FROM changed"
                            if self.show_modelfile_drift and self.show_calls == 2
                            else "FROM pinned"
                        ),
                        "model_info": {"nomic-bert.embedding_length": DIMENSION},
                        "parameters": "num_ctx 2048",
                    }
                ).encode(),
            )
        if url.endswith("/api/ps"):
            return 200, {}, b'{"models":[]}'
        if url.endswith("/api/embed"):
            if self.fail_embed:
                raise RuntimeError("simulated transport failure")
            assert payload is not None
            assert payload["truncate"] is False
            if self.changed_manifest is not None:
                self.changed_manifest.write_text('{"changed":true}', encoding="utf-8")
                self.changed_manifest = None
            count = len(payload["input"]) - 1 if self.bad_count else len(payload["input"])
            vectors = [[1.0] * DIMENSION for _ in range(max(0, count))]
            if self.bad_vector == "dimension":
                vectors[0] = [1.0]
            if self.bad_vector == "nonfinite":
                vectors[0][0] = float("nan")
            if self.bad_vector == "zero":
                vectors[0] = [0.0] * DIMENSION
            return 200, {}, json.dumps({"embeddings": vectors, "model": MODEL}).encode()
        raise AssertionError(url)


def _run(tmp_path: Path, fake: FakeOllama) -> dict[str, Any]:
    repo, manifest, helper = _git_fixture(tmp_path)
    return prepare_source_association_vectors(
        manifest,
        tmp_path / "artifact",
        transport=fake,
        repo_root=repo,
        helper_path=helper,
    )


def _git_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = repo / "manifest.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    helper = repo / "prepare.py"
    shutil.copy2(
        Path(__file__).parents[1] / "benchmarks/local_memory/prepare_source_association_vectors.py",
        helper,
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "manifest.json", "prepare.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    return repo, manifest, helper


def test_preparation_caches_texts_and_preserves_role_bindings(tmp_path: Path) -> None:
    fake = FakeOllama()
    artifact = _run(tmp_path, fake)
    assert artifact["status"] == "complete"
    assert len(artifact["vectors"]) == 4
    assert {row["role"] for row in artifact["bindings"]} == {
        "query",
        "fact",
        "event",
        "scope_control",
        "correction_new_content",
    }
    assert sum(url.endswith("/api/embed") for _, url, _ in fake.calls) == 1
    assert all(len(row["vector"]) == DIMENSION for row in artifact["vectors"].values())
    assert artifact["embedding_text_contract"] == {
        "query_prefix": "",
        "document_prefix": "",
        "symmetric_prefix": "",
        "symmetry": "symmetric",
    }
    assert "python" in artifact["runtime"] and "platform" in artifact["runtime"]


@pytest.mark.parametrize("bad_vector", ["dimension", "nonfinite", "zero"])
def test_invalid_vectors_leave_failed_partial_artifact(tmp_path: Path, bad_vector: str) -> None:
    fake = FakeOllama(bad_vector=bad_vector)
    with pytest.raises(ValueError, match="embedding"):
        _run(tmp_path, fake)
    report = json.loads((tmp_path / "artifact" / "artifact.json").read_text())
    assert report["status"] == "failed"
    assert (tmp_path / "artifact" / "raw").is_dir()


def test_count_model_and_input_changes_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="count"):
        _run(tmp_path / "count", FakeOllama(bad_count=True))

    repo, manifest, helper = _git_fixture(tmp_path / "changed")
    with pytest.raises(ValueError, match="working input differs"):
        prepare_source_association_vectors(
            manifest,
            tmp_path / "changed" / "artifact",
            transport=FakeOllama(changed_manifest=manifest),
            repo_root=repo,
            helper_path=helper,
        )


def test_output_refuses_overwrite_and_model_digest_is_checked(tmp_path: Path) -> None:
    fake = FakeOllama()
    _run(tmp_path, fake)
    repo = tmp_path / "repo"
    manifest = repo / "manifest.json"
    helper = repo / "prepare.py"
    with pytest.raises(FileExistsError):
        prepare_source_association_vectors(
            manifest,
            tmp_path / "artifact",
            transport=FakeOllama(),
            repo_root=repo,
            helper_path=helper,
        )
    with pytest.raises(ValueError, match="digest"):
        _run(tmp_path / "wrong-digest", FakeOllama(wrong_digest=True))


def test_preflight_requires_clean_committed_inputs_and_loopback(tmp_path: Path) -> None:
    repo, manifest, helper = _git_fixture(tmp_path / "dirty")
    manifest.write_text(json.dumps({"changed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="working input differs"):
        prepare_source_association_vectors(
            manifest,
            tmp_path / "dirty" / "artifact",
            transport=FakeOllama(),
            repo_root=repo,
            helper_path=helper,
        )
    assert not (tmp_path / "dirty" / "artifact").exists()
    repo, manifest, helper = _git_fixture(tmp_path / "dirty-helper")
    helper.write_text(helper.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    fake = FakeOllama()
    with pytest.raises(ValueError, match="working input differs"):
        prepare_source_association_vectors(
            manifest,
            tmp_path / "dirty-helper" / "artifact",
            transport=fake,
            repo_root=repo,
            helper_path=helper,
        )
    assert fake.calls == []
    repo, manifest, helper = _git_fixture(tmp_path / "endpoint")
    with pytest.raises(ValueError, match="loopback"):
        prepare_source_association_vectors(
            manifest,
            tmp_path / "endpoint" / "artifact",
            endpoint="http://192.0.2.1:11434",
            transport=FakeOllama(),
            repo_root=repo,
            helper_path=helper,
        )
    with pytest.raises(ValueError, match="git repository"):
        prepare_source_association_vectors(
            manifest,
            tmp_path / "endpoint" / "artifact-no-repo",
            transport=FakeOllama(),
            repo_root=tmp_path / "no-repo",
            helper_path=helper,
        )
    with pytest.raises(ValueError, match="outside the repository"):
        prepare_source_association_vectors(
            manifest,
            repo / "artifact-inside-repo",
            transport=FakeOllama(),
            repo_root=repo,
            helper_path=helper,
        )


@pytest.mark.parametrize("untracked", ["manifest", "helper"])
def test_preflight_rejects_untracked_input_without_transport(
    tmp_path: Path, untracked: str
) -> None:
    repo, manifest, helper = _git_fixture(tmp_path / untracked)
    if untracked == "manifest":
        manifest = repo / "new-manifest.json"
        manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    else:
        helper = repo / "new-helper.py"
        shutil.copy2(repo / "prepare.py", helper)
    fake = FakeOllama()
    with pytest.raises(ValueError, match="not committed"):
        prepare_source_association_vectors(
            manifest,
            tmp_path / untracked / "artifact",
            transport=fake,
            repo_root=repo,
            helper_path=helper,
        )
    assert fake.calls == []
    assert not (tmp_path / untracked / "artifact").exists()


def test_show_drift_and_transport_failure_are_recorded(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="embedding family"):
        _run(tmp_path / "show", FakeOllama(show_drift=True))
    repo, manifest, helper = _git_fixture(tmp_path / "transport")
    with pytest.raises(RuntimeError, match="simulated"):
        prepare_source_association_vectors(
            manifest,
            tmp_path / "transport" / "artifact",
            transport=FakeOllama(fail_embed=True),
            repo_root=repo,
            helper_path=helper,
        )
    report = json.loads((tmp_path / "transport" / "artifact" / "artifact.json").read_text())
    assert report["status"] == "failed"
    assert report["requests"][-1]["status"] == "transport_error"
    assert (tmp_path / "transport" / "artifact" / "raw" / "request-0005.request.bin").is_file()
    with pytest.raises(RuntimeError, match="show metadata"):
        _run(tmp_path / "show-config", FakeOllama(show_modelfile_drift=True))
    with pytest.raises(ValueError, match="digest"):
        _run(tmp_path / "tags-after", FakeOllama(tags_drift_after=True))
