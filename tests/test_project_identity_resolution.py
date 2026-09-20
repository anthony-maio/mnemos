import os
import subprocess

import pytest

from mnemos.scopes import ScopeRegistry, canonical_project_root


def test_transient_git_timeout_does_not_register_fallback_identity(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    original = subprocess.run
    failed = False

    def flaky(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise subprocess.TimeoutExpired(args[0], 5)
        return original(*args, **kwargs)

    monkeypatch.setattr("mnemos.scopes.subprocess.run", flaky)
    registry = ScopeRegistry()
    project = registry.register_project(repo, user_profile_id="local", enabled=True)
    assert project.root == canonical_project_root(repo)
    assert (
        registry.resolve(repo, user_profile_id="local", session_id="new", host="manual").project_id
        == project.id
    )


@pytest.mark.parametrize("failure", ["timeout", "missing", "error"])
def test_git_failure_cannot_register_or_alias_a_repository(tmp_path, monkeypatch, failure):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: elsewhere")
    nested = repo / "src"
    nested.mkdir()

    def unavailable(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 5)
        if failure == "missing":
            raise FileNotFoundError("git unavailable")
        return subprocess.CompletedProcess(args[0], 128, "", "fatal: repository unavailable")

    monkeypatch.setattr("mnemos.scopes.subprocess.run", unavailable)
    registry = ScopeRegistry()
    with pytest.raises(ValueError, match="project identity"):
        registry.register_project(nested, user_profile_id="local", enabled=True)
    assert registry.projects == {}


def test_plain_directory_without_git_remains_supported(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr("mnemos.scopes.subprocess.run", missing)
    assert canonical_project_root(tmp_path) == os.path.normcase(str(tmp_path.resolve()))


@pytest.mark.parametrize("output", ["", "relative/path", "/nonexistent/mnemos-git-directory"])
def test_invalid_git_output_cannot_register_repository(tmp_path, monkeypatch, output):
    (tmp_path / ".git").mkdir()

    def invalid(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, output, "")

    monkeypatch.setattr("mnemos.scopes.subprocess.run", invalid)
    registry = ScopeRegistry()
    with pytest.raises(ValueError, match="project identity"):
        registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    assert registry.projects == {}


def test_resolution_timeout_preserves_registered_identity(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    registry = ScopeRegistry()
    project = registry.register_project(repo, user_profile_id="local", enabled=True)
    before = registry.model_dump()
    original = subprocess.run

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 5)

    monkeypatch.setattr("mnemos.scopes.subprocess.run", timeout)
    with pytest.raises(ValueError, match="project identity"):
        registry.resolve(repo, user_profile_id="local", session_id="new", host="manual")
    with pytest.raises(ValueError, match="project identity"):
        registry.add_alias(project.id, repo)
    assert registry.model_dump() == before
    monkeypatch.setattr("mnemos.scopes.subprocess.run", original)
    assert (
        registry.resolve(repo, user_profile_id="local", session_id="new", host="manual").project_id
        == project.id
    )


def test_unrelated_clone_keeps_distinct_identity(tmp_path):
    repo, clone = tmp_path / "repo", tmp_path / "clone"
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "clone", str(repo), str(clone)], check=True, capture_output=True)
    registry = ScopeRegistry()
    first = registry.register_project(repo, user_profile_id="local", enabled=True)
    second = registry.register_project(clone, user_profile_id="local", enabled=True)
    assert first.id != second.id
    assert first.root != second.root


@pytest.mark.parametrize("variable", ["GIT_DIR", "GIT_COMMON_DIR"])
def test_inherited_git_environment_cannot_select_another_projects_identity(
    tmp_path, monkeypatch, variable
):
    first, second = tmp_path / "first", tmp_path / "second"
    for directory in (first, second):
        subprocess.run(["git", "init", str(directory)], check=True, capture_output=True)
    expected = canonical_project_root(first)
    foreign = canonical_project_root(second)
    monkeypatch.setenv(variable, str(second / ".git"))
    assert canonical_project_root(first) == expected
    assert canonical_project_root(first) != foreign


def test_plain_directory_with_normal_git_nonrepo_response(tmp_path, monkeypatch):
    def nonrepo(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0], 128, "", "fatal: not a git repository (or any parent)"
        )

    monkeypatch.setattr("mnemos.scopes.subprocess.run", nonrepo)
    assert canonical_project_root(tmp_path) == os.path.normcase(str(tmp_path.resolve()))


def test_bare_repository_requires_git_identity_on_failure(tmp_path, monkeypatch):
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    assert canonical_project_root(bare) == os.path.normcase(str(bare.resolve()))

    def missing(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr("mnemos.scopes.subprocess.run", missing)
    with pytest.raises(ValueError, match="project identity"):
        canonical_project_root(bare)


def test_dangling_git_marker_cannot_become_plain_directory(tmp_path, monkeypatch):
    marker = tmp_path / ".git"
    try:
        marker.symlink_to(tmp_path / "missing-git-directory", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")

    def nonrepo(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 128, "", "fatal: not a git repository")

    monkeypatch.setattr("mnemos.scopes.subprocess.run", nonrepo)
    with pytest.raises(ValueError, match="project identity"):
        canonical_project_root(tmp_path)
