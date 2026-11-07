"""Sleep configuration loading, persistence, and runtime wiring."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mnemos.cli import _build_engine
from mnemos.config import SleepConfig
from mnemos.mcp_server import _build_config
from mnemos.runtime import build_mnemos_config_from_env, build_mnemos_config_from_settings
from mnemos.settings import AppSettings, load_settings, save_settings

SLEEP_VALUES = {
    "consolidation_interval_seconds": 120,
    "min_episodes_before_consolidation": 3,
    "max_episodes_per_consolidation": 25,
    "enable_proceduralization": True,
    "recall_gated_plasticity_enabled": True,
    "recall_min_supporting_episodes": 4,
    "recall_similarity_threshold": 0.9,
    "consolidation_prompt": "Extract facts from {episodes}",
    "proceduralization_prompt": "Find repeated tasks in {episodes}",
}


@pytest.mark.parametrize("source", ["toml", "env"])
def test_cli_stats_and_mcp_config_enable_recall_gating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    config_path = tmp_path / "mnemos.toml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MNEMOS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "mock")
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "simple")
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "memory")
    if source == "toml":
        monkeypatch.delenv("MNEMOS_SLEEP_RECALL_GATED_PLASTICITY_ENABLED", raising=False)
        config_path.write_text(
            "[sleep]\nrecall_gated_plasticity_enabled = true\n", encoding="utf-8"
        )
    else:
        monkeypatch.setenv("MNEMOS_SLEEP_RECALL_GATED_PLASTICITY_ENABLED", "true")

    engine = _build_engine()

    assert engine.get_stats()["sleep_daemon"]["recall_gated_plasticity_enabled"] is True
    assert _build_config().sleep.recall_gated_plasticity_enabled is True


@pytest.mark.parametrize("source", ["toml", "env"])
def test_all_sleep_settings_reach_runtime(tmp_path: Path, source: str) -> None:
    config_path = tmp_path / "mnemos.toml"
    env = {"MNEMOS_CONFIG_PATH": str(config_path)}
    if source == "toml":
        lines = ["[sleep]"]
        for name, value in SLEEP_VALUES.items():
            literal = f'"{value}"' if isinstance(value, str) else str(value).lower()
            lines.append(f"{name} = {literal}")
        config_path.write_text("\n".join(lines), encoding="utf-8")
    else:
        env.update(
            {f"MNEMOS_SLEEP_{name.upper()}": str(value) for name, value in SLEEP_VALUES.items()}
        )

    config = build_mnemos_config_from_env(env=env, cwd=tmp_path)

    assert config.sleep.model_dump() == SLEEP_VALUES


def test_sleep_settings_defaults_match_engine(tmp_path: Path) -> None:
    resolved = load_settings(env={}, cwd=tmp_path, global_config_path=tmp_path / "missing.toml")

    assert resolved.settings.sleep.model_dump() == SleepConfig().model_dump()
    assert build_mnemos_config_from_settings(resolved.settings).sleep == SleepConfig()


def test_sleep_settings_precedence(tmp_path: Path) -> None:
    global_config = tmp_path / "global.toml"
    global_config.write_text(
        "[sleep]\nconsolidation_interval_seconds = 120\n"
        "min_episodes_before_consolidation = 5\nrecall_gated_plasticity_enabled = true\n",
        encoding="utf-8",
    )
    project = tmp_path / "repo"
    project_config = project / ".mnemos" / "mnemos.toml"
    project_config.parent.mkdir(parents=True)
    project_config.write_text(
        "[sleep]\nmin_episodes_before_consolidation = 3\n"
        "recall_gated_plasticity_enabled = true\nrecall_min_supporting_episodes = 4\n",
        encoding="utf-8",
    )

    resolved = load_settings(
        env={
            "MNEMOS_SLEEP_RECALL_GATED_PLASTICITY_ENABLED": "false",
            "MNEMOS_SLEEP_RECALL_MIN_SUPPORTING_EPISODES": "6",
            "MNEMOS_SLEEP_CONSOLIDATION_INTERVAL_SECONDS": "",
        },
        cwd=project,
        global_config_path=global_config,
    )

    sleep = build_mnemos_config_from_settings(resolved.settings).sleep
    assert sleep.consolidation_interval_seconds == 120
    assert sleep.min_episodes_before_consolidation == 3
    assert sleep.recall_gated_plasticity_enabled is False
    assert sleep.recall_min_supporting_episodes == 6
    assert sleep.max_episodes_per_consolidation == 100


@pytest.mark.parametrize("source", ["toml", "env"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("consolidation_interval_seconds", -1),
        ("min_episodes_before_consolidation", 0),
        ("max_episodes_per_consolidation", 0),
        ("recall_min_supporting_episodes", 0),
        ("recall_similarity_threshold", -1.1),
        ("recall_similarity_threshold", 1.1),
        ("recall_gated_plasticity_enabled", "invalid"),
    ],
)
def test_invalid_sleep_settings_are_rejected(
    tmp_path: Path, source: str, field: str, value: Any
) -> None:
    config_path = tmp_path / "mnemos.toml"
    env = {"MNEMOS_CONFIG_PATH": str(config_path)}
    if source == "toml":
        literal = f'"{value}"' if isinstance(value, str) else str(value)
        config_path.write_text(f"[sleep]\n{field} = {literal}\n", encoding="utf-8")
    else:
        env[f"MNEMOS_SLEEP_{field.upper()}"] = str(value)

    with pytest.raises(ValidationError, match=f"sleep.{field}"):
        load_settings(env=env, cwd=tmp_path)


@pytest.mark.parametrize("scope", ["global", "project"])
@pytest.mark.parametrize("custom", [False, True])
def test_sleep_settings_round_trip(tmp_path: Path, scope: str, custom: bool) -> None:
    settings = AppSettings.model_validate({"sleep": SLEEP_VALUES} if custom else {})
    if custom:
        settings.sleep.consolidation_prompt = 'Facts:\n\t"{episodes}"\r\nC:\\memory'
        settings.sleep.proceduralization_prompt = 'Tasks:\n{episodes}\nReturn "NO_PATTERN"'
    global_config = tmp_path / "global.toml"
    config_path = global_config if scope == "global" else tmp_path / ".mnemos" / "mnemos.toml"

    save_settings(settings, config_path, scope=scope)
    loaded = load_settings(env={}, cwd=tmp_path, global_config_path=global_config)

    assert loaded.settings.sleep.model_dump() == settings.sleep.model_dump()
    assert build_mnemos_config_from_settings(loaded.settings).sleep.model_dump() == (
        settings.sleep.model_dump()
    )
