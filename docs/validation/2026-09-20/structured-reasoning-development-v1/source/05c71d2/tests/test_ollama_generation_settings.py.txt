"""Managed local inference must honor explicit generation and reasoning budgets."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mnemos.runtime import build_llm_from_settings
from mnemos.settings import AppSettings, load_settings, save_settings
from mnemos.utils.llm import OllamaProvider


def test_managed_ollama_generation_settings_round_trip_and_reach_provider(tmp_path: Path) -> None:
    path = tmp_path / "mnemos.toml"
    path.write_text(
        '[llm]\nprovider = "ollama"\nmodel = "qwen3.5:9b"\n'
        "[providers.ollama]\nthink = true\nmax_tokens = 8192\ntimeout = 180.0\ntemperature = 0.0\n",
        encoding="utf-8",
    )
    settings = load_settings(env={"MNEMOS_CONFIG_PATH": str(path)}, cwd=tmp_path).settings
    provider = build_llm_from_settings(settings)
    assert isinstance(provider, OllamaProvider)
    assert (provider.think, provider.max_tokens, provider.timeout, provider.temperature) == (
        True,
        8192,
        180.0,
        0.0,
    )
    saved = tmp_path / "saved.toml"
    save_settings(settings, saved, scope="global")
    reloaded = load_settings(env={"MNEMOS_CONFIG_PATH": str(saved)}, cwd=tmp_path).settings
    assert reloaded.providers.ollama == settings.providers.ollama


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", 0),
        ("max_tokens", -1),
        ("max_tokens", True),
        ("max_tokens", 1.5),
        ("timeout", 0),
        ("timeout", -1),
        ("timeout", float("inf")),
        ("timeout", True),
        ("temperature", -1),
        ("temperature", float("nan")),
        ("temperature", True),
    ],
)
def test_invalid_managed_generation_settings_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"providers": {"ollama": {field: value}}})


def test_managed_generation_defaults_match_existing_provider_defaults() -> None:
    settings = AppSettings.model_validate({"llm": {"provider": "ollama"}})
    provider = build_llm_from_settings(settings)
    default = OllamaProvider()
    assert isinstance(provider, OllamaProvider)
    assert (provider.max_tokens, provider.timeout, provider.temperature, provider.think) == (
        default.max_tokens,
        default.timeout,
        default.temperature,
        default.think,
    )
