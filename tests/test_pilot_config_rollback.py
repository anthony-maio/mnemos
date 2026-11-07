"""Rehearsal preserves exact bytes and refuses to overwrite subsequent edits."""

import pytest

from tests.test_installed_artifact import load_script


@pytest.fixture
def runner():
    return load_script("scripts/check-pilot-config-rollback.py")


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_two_insertions_preserve_original_bytes(runner, newline):
    original = newline.join([b"first", b"last", b""])
    expected = newline.join([b"first", b"inserted", b"last", b"tail", b""])
    patch = "--- before\n+++ after\n@@ -1,0 +2,1 @@\n+inserted\n@@ -2,0 +4,1 @@\n+tail\n"
    assert runner.apply_additions(original, patch, runner.digest(expected)) == expected


def test_patch_hash_or_removals_fail_closed(runner):
    with pytest.raises(ValueError, match="reviewed proposal"):
        runner.apply_additions(None, "--- a\n+++ a\n@@ -0,0 +1,1 @@\n+new\n", "0" * 64)
    with pytest.raises(ValueError, match="addition hunks"):
        runner.apply_additions(b"old\n", "--- a\n+++ a\n@@ -1,1 +1,1 @@\n-old\n+new\n", "0" * 64)


@pytest.mark.parametrize("original", [None, b"original\r\n"])
def test_drift_blocks_apply_and_rollback_without_mutating_file(runner, tmp_path, original):
    path = tmp_path / "config"
    if original is not None:
        path.write_bytes(original)
    proposed = b"new\n"
    runner.guarded_replace(path, expected=original, replacement=proposed, root=tmp_path)
    path.write_bytes(b"user changed it\n")
    with pytest.raises(ValueError, match="Configuration drift"):
        runner.guarded_replace(path, expected=proposed, replacement=original, root=tmp_path)
    assert path.read_bytes() == b"user changed it\n"
    runner.guarded_replace(path, expected=b"user changed it\n", replacement=original, root=tmp_path)
    assert runner.read(path) == original and path.exists() == (original is not None)


def test_rehearsal_cannot_write_outside_root(runner, tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = tmp_path / "live-config"
    target.write_bytes(b"keep")
    with pytest.raises(ValueError, match="escaped"):
        runner.guarded_replace(target, expected=b"keep", replacement=None, root=sandbox)
    assert target.read_bytes() == b"keep"


def test_home_reference_allows_fresh_temp_but_preserves_host_and_project_boundaries(
    runner, tmp_path
):
    home = tmp_path / "user"
    project = tmp_path / "project"
    watched = {
        home / ".claude.json": b"reference",
        home / ".codex/config.toml": b"host",
        project / "AGENTS.md": None,
    }
    temporary = home / "AppData/Local/Temp/unique-rehearsal"
    assert runner.validate_output_root(temporary, watched, home) == temporary.resolve()
    for forbidden in (home / ".codex/new", project / "new", home / ".claude.json"):
        with pytest.raises(ValueError):
            runner.validate_output_root(forbidden, watched, home)


def test_existing_output_is_never_reused(runner, tmp_path):
    with pytest.raises(ValueError, match="fresh disposable"):
        runner.validate_output_root(tmp_path, {}, tmp_path / "home")
