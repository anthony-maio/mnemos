import os
from pathlib import Path

import pytest

from mnemos import git_policy_owner as module
from mnemos.scopes import canonical_project_root
from tests.test_project_identity_cache import repo, recorder  # noqa: F401


def setup(repo):
    owner = module.GitPolicyOwner(repo, runtime_files=())
    cache, calls = recorder()
    return owner, cache, calls


def test_unchanged_policy_reuses_closure_and_identity(repo, monkeypatch):
    owner, cache, calls = setup(repo)
    original = module.discover_config_dependencies
    discoveries = []

    def discover(*args):
        discoveries.append(args[0])
        return original(*args)

    monkeypatch.setattr(module, "discover_config_dependencies", discover)
    assert cache.resolve(repo, policy_probe=owner.probe) == canonical_project_root(repo)
    assert cache.resolve(repo, policy_probe=owner.probe) == canonical_project_root(repo)
    assert len(calls) == len(discoveries) == 1


def test_new_global_file_forces_authoritative_lookup(repo):
    owner, cache, calls = setup(repo)
    cache.resolve(repo, policy_probe=owner.probe)
    assert cache._cached is not None
    Path(os.environ["GIT_CONFIG_GLOBAL"]).write_text("[user]\nname = New\n")
    cache.resolve(repo, policy_probe=owner.probe)
    assert len(calls) == 2


def test_changed_include_with_preserved_mtime_invalidates_identity(repo):
    global_path = Path(os.environ["GIT_CONFIG_GLOBAL"])
    global_path.write_text("[include]\npath = child\n")
    child = global_path.parent / "child"
    child.write_text("[user]\nname = one\n")
    owner, cache, calls = setup(repo)
    cache.resolve(repo, policy_probe=owner.probe)
    assert cache._cached is not None
    saved = child.stat()
    child.write_text("[user]\nname = two\n")
    os.utime(child, ns=(saved.st_atime_ns, saved.st_mtime_ns))
    cache.resolve(repo, policy_probe=owner.probe)
    assert len(calls) == 2


def test_unsupported_runtime_config_delegates_without_reuse(repo, monkeypatch):
    owner, cache, calls = setup(repo)
    cache.resolve(repo, policy_probe=owner.probe)
    assert cache._cached is not None
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.name")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "Test")
    assert cache.resolve(repo, policy_probe=owner.probe) == canonical_project_root(repo)
    assert len(calls) == 2 and cache._cached is None


@pytest.mark.skipif(os.name != "nt", reason="Windows wrapper HOME selection")
def test_home_selection_observes_creation_removal_and_explicit_override(tmp_path):
    candidate = tmp_path / "home"
    environment = {"HOMEDRIVE": candidate.drive, "HOMEPATH": str(candidate)[len(candidate.drive):]}
    absent = module._home_selection_observation(environment)
    candidate.mkdir()
    assert module._home_selection_observation(environment) != absent
    candidate.rmdir()
    assert module._home_selection_observation(environment) == absent
    environment["HOME"] = str(tmp_path)
    assert module._home_selection_observation(environment) == ()
    environment.pop("HOME")
    environment["HOMEDRIVE"] = ""
    environment["HOMEPATH"] = "relative"
    with pytest.raises(ValueError, match="home selection"):
        module._home_selection_observation(environment)


def test_unreadable_dependency_forces_authoritative_fallback(repo, monkeypatch):
    owner, cache, calls = setup(repo)
    cache.resolve(repo, policy_probe=owner.probe)
    assert cache._cached is not None

    def fail(*args):
        raise PermissionError("Cannot observe config")

    monkeypatch.setattr(module, "observe_config_path", fail)
    assert cache.resolve(repo, policy_probe=owner.probe) == canonical_project_root(repo)
    assert len(calls) == 2 and cache._cached is None


def test_declared_runtime_dependency_content_change_invalidates(repo, tmp_path):
    binary = tmp_path / "runtime.bin"
    binary.write_bytes(b"first")
    owner = module.GitPolicyOwner(repo, runtime_files=(binary,))
    cache, calls = recorder()
    cache.resolve(repo, policy_probe=owner.probe)
    assert cache._cached is not None
    before = binary.stat()
    binary.write_bytes(b"other")
    os.utime(binary, ns=(before.st_atime_ns, before.st_mtime_ns))
    cache.resolve(repo, policy_probe=owner.probe)
    assert len(calls) == 2


def test_policy_change_inside_resolver_does_not_publish_identity(repo):
    owner = module.GitPolicyOwner(repo, runtime_files=())
    assert owner.probe() is not None
    from mnemos.project_identity_cache import ProjectIdentityCache

    def resolve(path):
        result = canonical_project_root(path)
        Path(os.environ["GIT_CONFIG_GLOBAL"]).write_text("[user]\nname = Changed\n")
        return result

    cache = ProjectIdentityCache(resolve)
    assert cache.resolve(repo, policy_probe=owner.probe) == canonical_project_root(repo)
    assert cache._cached is None


def test_new_invalid_policy_cannot_hide_authoritative_git_failure(repo):
    owner, cache, calls = setup(repo)
    cache.resolve(repo, policy_probe=owner.probe)
    assert cache._cached is not None
    Path(os.environ["GIT_CONFIG_GLOBAL"]).write_text("[broken\n")
    with pytest.raises(ValueError):
        cache.resolve(repo, policy_probe=owner.probe)
    assert len(calls) == 2 and cache._cached is None


def test_new_include_edge_is_rediscovered_before_later_target_change(repo):
    owner, cache, calls = setup(repo)
    cache.resolve(repo, policy_probe=owner.probe)
    assert cache._cached is not None
    global_path = Path(os.environ["GIT_CONFIG_GLOBAL"])
    global_path.write_text("[include]\npath = newly-included\n")
    cache.resolve(repo, policy_probe=owner.probe)
    child = global_path.parent / "newly-included"
    assert child in owner._dependencies.paths
    child.write_text("[user]\nname = New\n")
    cache.resolve(repo, policy_probe=owner.probe)
    assert len(calls) == 3 and cache._cached is not None


@pytest.mark.skipif(os.name != "nt", reason="Windows wrapper HOME selection")
def test_new_home_directory_cannot_hide_authoritative_git_failure(repo, tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    roaming = tmp_path / "new-roaming-home"
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)
    monkeypatch.setenv("USERPROFILE", str(profile))
    monkeypatch.setenv("HOMEDRIVE", roaming.drive)
    monkeypatch.setenv("HOMEPATH", str(roaming)[len(roaming.drive):])
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    owner, cache, calls = setup(repo)
    cache.resolve(repo, policy_probe=owner.probe)
    assert cache._cached is not None
    roaming.mkdir()
    (roaming / ".gitconfig").write_text("[broken\n")
    with pytest.raises(ValueError):
        canonical_project_root(repo)
    with pytest.raises(ValueError):
        cache.resolve(repo, policy_probe=owner.probe)
    assert len(calls) == 2 and cache._cached is None
