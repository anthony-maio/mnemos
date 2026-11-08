import os
from pathlib import Path
import shutil

import pytest

from mnemos.git_policy_dependencies import (
    discover_config_dependencies,
    observe_config_path,
    PolicyUnavailable,
)
from tests.test_project_identity_cache import repo  # noqa: F401


def discover(repo):
    return discover_config_dependencies(repo, Path(shutil.which("git")), dict(os.environ))


def test_missing_global_and_repository_paths_are_observed(repo):
    result = discover(repo)
    global_path = Path(os.environ["GIT_CONFIG_GLOBAL"])
    assert global_path in result.paths
    assert repo / ".git/config" in result.paths
    assert repo / ".git/config.worktree" in result.paths
    prior = result.observations[result.paths.index(global_path)]
    global_path.write_text("[user]\nname = Test\n")
    assert observe_config_path(global_path) != prior


def test_relative_transitive_and_missing_includes_are_tracked(repo):
    global_path = Path(os.environ["GIT_CONFIG_GLOBAL"])
    global_path.write_text('[include]\npath = "first config"\n')
    first = global_path.parent / "first config"
    first.write_text("[include]\npath = missing\n")
    result = discover(repo)
    assert {global_path, first, global_path.parent / "missing"} <= set(result.paths)


def test_inactive_conditional_include_is_still_tracked(repo):
    global_path = Path(os.environ["GIT_CONFIG_GLOBAL"])
    global_path.write_text('[includeIf "onbranch:never-this-branch"]\npath = future\n')
    assert global_path.parent / "future" in discover(repo).paths


def test_malformed_configuration_cannot_produce_dependencies(repo):
    Path(os.environ["GIT_CONFIG_GLOBAL"]).write_text("[bad\n")
    with pytest.raises(PolicyUnavailable):
        discover(repo)


def test_inactive_include_cycle_is_rejected(repo):
    global_path = Path(os.environ["GIT_CONFIG_GLOBAL"])
    global_path.write_text('[includeIf "onbranch:never-this-branch"]\npath = loop\n')
    (global_path.parent / "loop").write_text(f"[include]\npath = {global_path.name}\n")
    with pytest.raises(PolicyUnavailable, match="Cyclic"):
        discover(repo)


def test_runtime_injected_configuration_requires_authoritative_fallback(repo, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "include.path")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "outside")
    with pytest.raises(PolicyUnavailable, match="overrides"):
        discover(repo)


def test_parent_traversal_is_not_lexically_normalized_past_possible_links(repo):
    global_path = Path(os.environ["GIT_CONFIG_GLOBAL"])
    global_path.write_text('[includeIf "onbranch:never-this-branch"]\npath = child/../outside\n')
    with pytest.raises(PolicyUnavailable, match="Parent traversal"):
        discover(repo)


def test_multiline_global_selector_cannot_be_split_into_wrong_dependencies(repo, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(repo / "one\ntwo"))
    with pytest.raises(PolicyUnavailable, match="Multiline"):
        discover(repo)


def test_configuration_change_during_discovery_is_rejected(repo, monkeypatch):
    from mnemos import git_policy_dependencies as module

    path = Path(os.environ["GIT_CONFIG_GLOBAL"])
    path.write_text("[user]\nname = first\n")
    original = module.observe_config_path
    count = 0

    def changed(target):
        nonlocal count
        if target == path:
            count += 1
            if count == 2:
                path.write_text("[user]\nname = second\n")
        return original(target)

    monkeypatch.setattr(module, "observe_config_path", changed)
    with pytest.raises(PolicyUnavailable, match="changed"):
        discover(repo)
