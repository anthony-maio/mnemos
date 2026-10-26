"""Ollama thinking-field and final-response contract tests."""

from __future__ import annotations

from datetime import datetime, timezone
import json

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import mnemos.utils.llm as llm_module
from mnemos.modules.sleep import SleepDaemon
from mnemos.runtime import build_llm_from_settings
from mnemos.settings import AppSettings, save_settings, load_settings
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import MnemosExternalServiceError, OllamaProvider
from mnemos.utils.storage import InMemoryStore


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class _Client:
    payloads: list[dict[str, object]] = []
    response_payload: object = {"response": "ok"}

    def __init__(self, *, timeout: float, verify: object = True) -> None:
        _ = timeout, verify

    async def __aenter__(self) -> "_Client":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = exc_type, exc, tb

    async def post(self, url: str, *, json: dict[str, object]) -> _Response:
        _ = url
        self.payloads.append(json)
        return _Response(self.response_payload)


def _install_client(monkeypatch: pytest.MonkeyPatch, response: object) -> None:
    _Client.payloads = []
    _Client.response_payload = response
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", _Client)


def test_ollama_think_setting_round_trips_and_default_is_omitted(tmp_path: Path) -> None:
    config_path = tmp_path / "mnemos.toml"
    config_path.write_text(
        '[llm]\nprovider = "ollama"\nmodel = "qwen3.5:9b"\n' "[providers.ollama]\nthink = false\n",
        encoding="utf-8",
    )
    resolved = load_settings(
        env={"MNEMOS_CONFIG_PATH": str(config_path)},
        cwd=tmp_path,
    )
    assert resolved.settings.providers.ollama.think is False

    saved_path = tmp_path / "saved.toml"
    save_settings(resolved.settings, saved_path, scope="global")
    saved = tomllib.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["providers"]["ollama"]["think"] is False

    default_path = tmp_path / "default.toml"
    save_settings(AppSettings(), default_path, scope="global")
    default = tomllib.loads(default_path.read_text(encoding="utf-8"))
    assert "think" not in default.get("providers", {}).get("ollama", {})


@pytest.mark.parametrize("value", ["false", "auto", 1])
def test_ollama_think_setting_rejects_non_documented_values(value: object) -> None:
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"providers": {"ollama": {"think": value}}})


def test_runtime_wires_ollama_think_setting(tmp_path: Path) -> None:
    settings = AppSettings.model_validate(
        {
            "llm": {"provider": "ollama", "model": "qwen3.5:9b"},
            "providers": {"ollama": {"base_url": "http://ollama", "think": "low"}},
            "storage": {"sqlite_path": str(tmp_path / "memory.sqlite")},
        }
    )
    provider = build_llm_from_settings(settings)
    assert isinstance(provider, OllamaProvider)
    assert provider.think == "low"


@pytest.mark.asyncio
@pytest.mark.parametrize("think", [None, False, True, "low", "medium", "high", "max"])
async def test_ollama_payload_omits_or_carries_think(
    monkeypatch: pytest.MonkeyPatch,
    think: bool | str | None,
) -> None:
    _install_client(monkeypatch, {"response": "final", "thinking": "private reasoning"})
    provider = OllamaProvider(think=think)
    assert await provider.predict("prompt") == "final"
    payload = _Client.payloads[-1]
    if think is None:
        assert "think" not in payload
    else:
        assert payload["think"] == think


@pytest.mark.asyncio
async def test_ollama_per_call_think_override_is_validated_and_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_client(monkeypatch, {"response": "final"})
    provider = OllamaProvider(think=False)
    await provider.predict("prompt", think="high")
    assert _Client.payloads[-1]["think"] == "high"

    with pytest.raises(ValueError, match="think must be"):
        OllamaProvider(think="false")
    with pytest.raises(ValueError, match="think must be"):
        await provider.predict("prompt", think="auto")


@pytest.mark.asyncio
async def test_ollama_preserves_legacy_positional_observer_and_returns_final_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {"response": "final", "thinking": "private reasoning"}
    _install_client(monkeypatch, response)
    observed: list[dict[str, Any]] = []
    provider = OllamaProvider("model", "http://ollama", 10.0, 0.0, 32, observed.append, think=False)

    assert await provider.predict("prompt") == "final"
    assert observed == [response]
    assert _Client.payloads[-1]["think"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    ["", {"facts": [{"content": "private reasoning"}]}],
)
async def test_ollama_rejects_empty_or_non_string_final_response_without_thinking_leak(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
) -> None:
    secret = '{"facts":[{"content":"private reasoning"}]}'
    _install_client(monkeypatch, {"response": response, "thinking": secret})
    observed: list[dict[str, Any]] = []
    provider = OllamaProvider(response_observer=observed.append)
    with pytest.raises(MnemosExternalServiceError) as failure:
        await provider.predict("extract")
    assert "final response" in str(failure.value)
    assert secret not in str(failure.value)
    assert observed[0]["thinking"] == secret


@pytest.mark.asyncio
async def test_extraction_provider_failure_creates_no_fact_from_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_client(
        monkeypatch,
        {
            "response": "",
            "thinking": json.dumps(
                {
                    "decision": "facts",
                    "facts": [{"content": "should never be durable"}],
                }
            ),
        },
    )
    event = EpisodicEvent(
        id="source-1",
        event_fingerprint="fingerprint-1",
        role="user",
        content="The source event is an extraction input.",
        normalized_content="the source event is an extraction input",
        scope="project",
        scope_id="project-1",
        metadata={"user_profile_id": "profile-1"},
        occurred_at=datetime.now(timezone.utc),
    )
    store = InMemoryStore()
    provider = OllamaProvider()
    with pytest.raises(MnemosExternalServiceError):
        await SleepDaemon(store).propose_facts([event], provider)
    assert store.get_all() == []
