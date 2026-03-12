from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_user_mnemos_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MNEMOS_CONFIG_PATH", str(tmp_path / "global-mnemos.toml"))
