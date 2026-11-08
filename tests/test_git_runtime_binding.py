from dataclasses import replace
import hashlib
import os
from pathlib import Path
import shutil

import pytest

from mnemos.git_policy_owner import GitPolicyOwner, PinnedGitRuntime
from mnemos.scopes import canonical_project_root
from tests.test_project_identity_cache import repo, recorder  # noqa: F401


def prepared(repo, tmp_path):
    executable = Path(shutil.which("git")).absolute()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    library = runtime / "dependency.dll"
    library.write_bytes(b"original")
    files = (executable, library)
    binding = PinnedGitRuntime(
        executable,
        tuple((p, hashlib.sha256(p.read_bytes()).hexdigest()) for p in files),
        tuple(
            (p, tuple(sorted(c.name for c in p.iterdir()))) for p in (executable.parent, runtime)
        ),
    )
    owner = GitPolicyOwner(repo, runtime_binding=binding)
    cache, calls = recorder()
    assert cache.resolve(repo, policy_probe=owner.probe) == canonical_project_root(repo)
    assert cache._cached is not None
    return binding, owner, cache, calls, library


def test_pinned_runtime_allows_unchanged_identity_reuse(repo, tmp_path):
    _, owner, cache, calls, _ = prepared(repo, tmp_path)
    cache.resolve(repo, policy_probe=owner.probe)
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["bytes", "added_file", "removed_file"])
def test_runtime_change_delegates_without_automatic_reapproval(repo, tmp_path, change):
    _, owner, cache, calls, library = prepared(repo, tmp_path)
    if change == "bytes":
        prior = library.stat()
        library.write_bytes(b"replaced")
        os.utime(library, ns=(prior.st_atime_ns, prior.st_mtime_ns))
    elif change == "added_file":
        (library.parent / "new.dll").write_bytes(b"new dependency")
    else:
        library.unlink()
    for _ in range(2):
        assert cache.resolve(repo, policy_probe=owner.probe) == canonical_project_root(repo)
        assert cache._cached is None and owner._context is None
    assert len(calls) == 3


def test_different_selected_git_path_is_not_approved_by_same_bytes(repo, tmp_path, monkeypatch):
    from mnemos import git_policy_owner as module

    binding, owner, _, _, _ = prepared(repo, tmp_path)
    copy = tmp_path / "other-git.exe"
    shutil.copyfile(binding.selected, copy)
    monkeypatch.setattr(module.shutil, "which", lambda name: str(copy))
    assert owner.probe() is None


def test_manifest_requires_selected_executable_and_directory_coverage(repo, tmp_path):
    binding, _, _, _, _ = prepared(repo, tmp_path)
    with pytest.raises(ValueError, match="Incomplete"):
        replace(binding, hashes=binding.hashes[1:])
    with pytest.raises(ValueError, match="parents"):
        replace(binding, directories=binding.directories[:1])


def test_bad_hash_and_mixed_unpinned_runtime_are_rejected(repo, tmp_path):
    binding, _, _, _, library = prepared(repo, tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        replace(binding, hashes=((binding.selected, "bad"), *binding.hashes[1:]))
    with pytest.raises(ValueError, match="not both"):
        GitPolicyOwner(repo, runtime_files=(library,), runtime_binding=binding)
