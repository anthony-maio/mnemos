from pathlib import Path
import runpy

import pytest

HELPER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "benchmarks/native_hosts/claude_isolation.py")
)


def test_case_insensitive_auth_removal_and_profile_override(tmp_path):
    home = tmp_path / "private"
    home.mkdir()
    inherited = {
        name.lower(): "outside-or-secret" for name in HELPER["REMOVED"] | HELPER["OVERRIDDEN"]
    }
    inherited["MNEMOS_PROJECT_ROOT"] = "disposable-project"
    env = HELPER["environment"](inherited, home, tmp_path)
    assert not any(key.upper() in HELPER["REMOVED"] for key in env)
    assert "outside-or-secret" not in env.values()
    assert env["MNEMOS_PROJECT_ROOT"] == "disposable-project"
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "CLAUDE_CONFIG_DIR"):
        assert Path(env[key]).resolve().is_relative_to(home)
    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"


def test_external_home_rejected_before_directory_creation(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    with pytest.raises(ValueError, match="private child"):
        HELPER["environment"]({}, tmp_path, root)
    assert not (tmp_path / "userprofile").exists()


def test_unreviewed_binary_rejected(tmp_path):
    binary = tmp_path / "claude.exe"
    binary.write_bytes(b"changed executable")
    with pytest.raises(ValueError, match="reviewed isolation"):
        HELPER["verify_binary"](binary)
