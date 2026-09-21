from __future__ import annotations

from pathlib import Path

import pytest

import mnemos.settings as settings_module

# Bound before any test patches it, so opt-in tests can reach the real lookup.
_REAL_PERSISTENT_ENV_VALUE = settings_module._persistent_env_value


@pytest.fixture(autouse=True)
def isolate_user_mnemos_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Keep every test away from the developer's live Mnemos profile.

    Environment variables reach spawned service processes, which do not see
    monkeypatches. The patched config directory covers in-process defaults,
    which Windows resolves through the Known Folder API instead of APPDATA.
    Both roots live outside tmp_path so tests may inspect their own directory.
    """
    user_root = tmp_path_factory.mktemp("mnemos-user")
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("MNEMOS_CONFIG_PATH", str(tmp_path / "global-mnemos.toml"))
    monkeypatch.setenv("MNEMOS_ROUTING_REVIEW_DIR", str(tmp_path / "routing-review"))
    monkeypatch.setenv("MNEMOS_AUTO_START", "0")
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(user_root / "memory.sqlite"))
    monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(user_root / "scopes.json"))
    monkeypatch.setenv("MNEMOS_RUNTIME_DIR", str(user_root / "runtime"))
    monkeypatch.delenv("MNEMOS_RUNTIME_DESCRIPTOR", raising=False)
    monkeypatch.delenv("MNEMOS_DB_PATH", raising=False)
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        settings_module, "_user_config_dir", lambda app_name: user_root / "config" / app_name
    )
    monkeypatch.setattr(settings_module, "_persistent_env_value", lambda name: None)


@pytest.fixture
def real_persistent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the real registry lookup; callers must supply a fake winreg."""
    monkeypatch.setattr(settings_module, "_persistent_env_value", _REAL_PERSISTENT_ENV_VALUE)
