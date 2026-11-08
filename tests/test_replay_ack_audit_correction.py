"""Git identities bind the common directory, not the working directory."""

from pathlib import Path
import runpy

import pytest

AUDIT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "benchmarks/native_hosts/replay_ack_audit_v2.py")
)


def test_prepared_git_common_directory_is_the_identity(tmp_path):
    common = tmp_path / "project/.git"
    common.mkdir(parents=True)
    AUDIT["validate_project_identity"](tmp_path, {"root": str(common)})


@pytest.mark.parametrize("wrong", ["project", "unrelated/.git"])
def test_other_directory_is_rejected(tmp_path, wrong):
    (tmp_path / "project/.git").mkdir(parents=True)
    with pytest.raises(AssertionError):
        AUDIT["validate_project_identity"](tmp_path, {"root": str(tmp_path / wrong)})


def test_missing_git_directory_is_rejected(tmp_path):
    with pytest.raises(AssertionError):
        AUDIT["validate_project_identity"](tmp_path, {"root": str(tmp_path / "project/.git")})
