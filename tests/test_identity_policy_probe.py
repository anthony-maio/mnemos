import pytest

from mnemos.project_identity_cache import ProjectIdentityCache
from mnemos.scopes import canonical_project_root
from tests.test_project_identity_cache import repo, recorder  # noqa: F401


def test_live_policy_change_invalidates_a_hit(repo):
    cache, calls = recorder()
    generation = ["first"]
    probe = lambda: generation[0]
    assert cache.resolve(repo, policy_probe=probe) == cache.resolve(repo, policy_probe=probe)
    assert len(calls) == 1
    generation[0] = "second"
    assert cache.resolve(repo, policy_probe=probe) == canonical_project_root(repo)
    assert len(calls) == 2


def test_policy_change_during_authoritative_resolution_is_not_cached(repo):
    generation = ["first"]

    def resolve(path):
        result = canonical_project_root(path)
        generation[0] = "second"
        return result

    cache = ProjectIdentityCache(resolve)
    assert cache.resolve(repo, policy_probe=lambda: generation[0]) == canonical_project_root(repo)
    assert cache._cached is None


def test_policy_change_during_hit_validation_forces_authoritative_lookup(repo):
    cache, calls = recorder()
    cache.resolve(repo, policy_probe=lambda: "first")
    samples = iter(["first", "second"])
    assert cache.resolve(repo, policy_probe=lambda: next(samples)) == canonical_project_root(repo)
    assert len(calls) == 2 and cache._cached is None


@pytest.mark.parametrize("failure", [None, "", " ", 1, PermissionError("unreadable")])
def test_unavailable_policy_discards_old_entry_and_delegates(repo, failure):
    cache, calls = recorder()
    cache.resolve(repo, policy_probe=lambda: "first")

    def probe():
        if isinstance(failure, Exception):
            raise failure
        return failure

    assert cache.resolve(repo, policy_probe=probe) == canonical_project_root(repo)
    assert len(calls) == 2 and cache._cached is None


def test_unavailable_policy_after_resolution_does_not_publish_entry(repo):
    cache, calls = recorder()
    samples = iter(["first", None])
    cache.resolve(repo, policy_probe=lambda: next(samples))
    assert len(calls) == 1 and cache._cached is None


def test_static_and_live_policy_cannot_be_combined(repo):
    cache, calls = recorder()
    with pytest.raises(ValueError, match="not both"):
        cache.resolve(repo, policy_generation="static", policy_probe=lambda: "live")
    assert not calls and cache._cached is None


@pytest.mark.parametrize("existing_entry", [False, True])
def test_environment_change_during_final_probe_prevents_reuse(repo, monkeypatch, existing_entry):
    cache, calls = recorder()
    if existing_entry:
        cache.resolve(repo, policy_probe=lambda: "first")
    count = 0

    def probe():
        nonlocal count
        count += 1
        if count == 2:
            monkeypatch.setenv("MNEMOS_POLICY_TEST_CHANGE", "changed")
        return "first"

    assert cache.resolve(repo, policy_probe=probe) == canonical_project_root(repo)
    assert len(calls) == (2 if existing_entry else 1)
    assert cache._cached is None
