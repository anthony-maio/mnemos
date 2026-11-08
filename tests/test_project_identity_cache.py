import os
from pathlib import Path
import subprocess

import pytest

from mnemos import project_identity_cache as module
from mnemos.errors import ScopeUnavailable
from mnemos.scopes import ScopeRegistry, canonical_project_root


def git(*args):
    return subprocess.run(
        ["git", *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.upper().startswith("GIT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "absent-global"))
    root = tmp_path / "repo"
    root.mkdir()
    git("init", "-q", root)
    git(
        "-C",
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty",
        "-qm",
        "initial",
    )
    return root


def recorder():
    calls = []

    def resolve(path):
        calls.append(path)
        return canonical_project_root(path)

    return module.ProjectIdentityCache(resolve), calls


def test_missing_policy_generation_never_caches(repo):
    cache, calls = recorder()
    assert cache.resolve(repo) == cache.resolve(repo)
    assert len(calls) == 2


def test_stable_repository_reuses_identity_but_not_ordinary_file_edits(repo):
    cache, calls = recorder()
    first = cache.resolve(repo, policy_generation="reviewed-policy-1")
    (repo / "ordinary.txt").write_text("user work")
    assert cache.resolve(repo, policy_generation="reviewed-policy-1") == first
    assert len(calls) == 1


def test_nested_repository_invalidates_cached_parent_identity(repo):
    child = repo / "child"
    child.mkdir()
    cache, calls = recorder()
    parent = cache.resolve(child, policy_generation="policy")
    git("init", "-q", child)
    nested = cache.resolve(child, policy_generation="policy")
    assert nested != parent and nested == canonical_project_root(child)
    assert len(calls) == 2


def test_worktree_pointer_retarget_invalidates_even_with_preserved_mtime(repo, tmp_path):
    tree = tmp_path / "linked"
    git("-C", repo, "worktree", "add", "--detach", "-q", tree, "HEAD")
    other = tmp_path / "other"
    other.mkdir()
    git("init", "-q", other)
    cache, calls = recorder()
    original = cache.resolve(tree, policy_generation="policy")
    assert original == canonical_project_root(repo)
    marker = tree / ".git"
    saved = marker.stat()
    # Git marks this file hidden on Windows; update the existing file handle.
    with marker.open("r+", encoding="utf-8") as handle:
        handle.seek(0)
        handle.write(f"gitdir: {other / '.git'}\n")
        handle.truncate()
    os.utime(marker, ns=(saved.st_atime_ns, saved.st_mtime_ns))
    assert cache.resolve(tree, policy_generation="policy") == canonical_project_root(other)
    assert len(calls) == 2


def test_commondir_change_invalidates_linked_worktree(repo, tmp_path):
    tree = tmp_path / "linked"
    git("-C", repo, "worktree", "add", "--detach", "-q", tree, "HEAD")
    cache, calls = recorder()
    cache.resolve(tree, policy_generation="policy")
    gitdir = Path((tree / ".git").read_text().strip()[8:])
    (gitdir / "commondir").write_text(str(tmp_path / "missing") + "\n")
    with pytest.raises(ValueError):
        cache.resolve(tree, policy_generation="policy")
    assert len(calls) == 2 and cache._cached is None


def test_policy_and_environment_changes_invalidate(repo, monkeypatch):
    cache, calls = recorder()
    cache.resolve(repo, policy_generation="policy-1")
    cache.resolve(repo, policy_generation="policy-2")
    monkeypatch.setenv("MNEMOS_IDENTITY_TEST", "changed")
    cache.resolve(repo, policy_generation="policy-2")
    assert len(calls) == 3


def test_git_disappearance_does_not_return_cached_identity(repo, monkeypatch):
    cache, calls = recorder()
    cache.resolve(repo, policy_generation="policy")
    monkeypatch.setattr(module.shutil, "which", lambda name: None)

    def unavailable(path):
        raise ValueError("Git unavailable")

    cache._resolver = unavailable
    with pytest.raises(ValueError, match="Git unavailable"):
        cache.resolve(repo, policy_generation="policy")
    assert cache._cached is None


def test_routing_change_during_lookup_is_not_cached(repo):
    def changing(path):
        result = canonical_project_root(path)
        with (repo / ".git/config").open("a") as handle:
            handle.write("\n# changed during lookup\n")
        return result

    cache = module.ProjectIdentityCache(changing)
    assert cache.resolve(repo, policy_generation="policy") == canonical_project_root(repo)
    assert cache._cached is None


def test_cached_identity_does_not_cache_membership(repo, monkeypatch):
    registry = ScopeRegistry()
    project = registry.register_project(repo, user_profile_id="owner", enabled=True)
    cache, calls = recorder()
    monkeypatch.setattr(
        "mnemos.scopes.canonical_project_root",
        lambda path: cache.resolve(path, policy_generation="policy"),
    )
    args = dict(user_profile_id="owner", session_id="session", host="codex")
    assert registry.resolve(repo, **args).project_id == project.id
    registry.projects[project.id].enabled = False
    with pytest.raises(ScopeUnavailable):
        registry.resolve(repo, **args)
    assert len(calls) == 1


def test_external_object_directory_loss_cannot_reuse_identity(repo, tmp_path, monkeypatch):
    objects = tmp_path / "external-objects"
    objects.mkdir()
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(objects))
    cache, calls = recorder()
    cache.resolve(repo, policy_generation="unchanged-config-policy")
    # Remove only an empty directory created by this test.
    objects.rmdir()
    with pytest.raises(ValueError):
        canonical_project_root(repo)
    with pytest.raises(ValueError):
        cache.resolve(repo, policy_generation="unchanged-config-policy")


def test_unknown_git_selector_bypasses_cache(repo, monkeypatch):
    monkeypatch.setenv("GIT_FUTURE_ROUTING_OPTION", "external-resource")
    cache, calls = recorder()
    assert cache.resolve(repo, policy_generation="policy") == cache.resolve(
        repo, policy_generation="policy"
    )
    assert len(calls) == 2 and cache._cached is None


@pytest.mark.skipif(os.name != "nt", reason="Windows security descriptor invalidation")
def test_acl_only_change_invalidates_repository_identity(repo):
    from mnemos.service import _protect_private_path

    cache, calls = recorder()
    first = cache.resolve(repo, policy_generation="unchanged-external-policy")
    # Change only the synthetic repository's DACL. No live project is touched.
    _protect_private_path(repo / ".git")
    assert cache.resolve(repo, policy_generation="unchanged-external-policy") == first
    assert len(calls) == 2


def test_unreadable_security_metadata_delegates_without_caching(repo, monkeypatch):
    cache, calls = recorder()
    cache.resolve(repo, policy_generation="policy")

    def unavailable(path, info):
        raise PermissionError("Cannot inspect security metadata")

    monkeypatch.setattr(module, "_security_identity", unavailable)
    assert cache.resolve(repo, policy_generation="policy") == canonical_project_root(repo)
    assert cache.resolve(repo, policy_generation="policy") == canonical_project_root(repo)
    assert len(calls) == 3 and cache._cached is None


def test_security_change_during_authoritative_resolution_is_not_cached(repo, monkeypatch):
    original = module._security_identity
    generation = [0]

    def security(path, info):
        return (original(path, info), generation[0])

    def changing(path):
        result = canonical_project_root(path)
        generation[0] += 1
        return result

    monkeypatch.setattr(module, "_security_identity", security)
    cache = module.ProjectIdentityCache(changing)
    assert cache.resolve(repo, policy_generation="policy") == canonical_project_root(repo)
    assert cache._cached is None
