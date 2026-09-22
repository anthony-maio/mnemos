"""Judgment is optional: off by default, explicit when on, never a silent cloud path."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from mnemos.health import run_health_checks
from mnemos.judgment import UnavailableJudge
from mnemos.runtime import build_configured_engine, build_judge_from_settings
from mnemos.settings import AppSettings, load_settings
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.ollama_judge import OllamaJudge
from mnemos.utils.typesafe import TypeSafeJudge
from mnemos.memory_safety import MemoryWriteFirewall
from mnemos.config import MemorySafetyConfig


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {
        "MNEMOS_CONFIG_PATH": str(tmp_path / "missing.toml"),
        "MNEMOS_SQLITE_PATH": str(tmp_path / "memory.db"),
        **extra,
    }


def test_judgment_is_off_by_default_and_the_engine_gets_an_unavailable_judge(
    tmp_path: Path,
) -> None:
    settings = load_settings(env=_env(tmp_path)).settings
    assert settings.judgment.provider == "none"
    assert settings.judgment.model is None
    assert settings.sleep.extraction_mode == "generated"

    engine = build_configured_engine(env=_env(tmp_path), cwd=tmp_path)
    try:
        assert isinstance(engine.llm, UnavailableLLMProvider)
        assert isinstance(engine.judge, UnavailableJudge)
    finally:
        asyncio.run(engine.aclose())


def test_env_overrides_select_the_judge_and_its_key(tmp_path: Path) -> None:
    settings = load_settings(
        env=_env(
            tmp_path,
            MNEMOS_JUDGMENT_PROVIDER="TypeSafe",
            MNEMOS_TYPESAFE_API_KEY="key-1",
            MNEMOS_TYPESAFE_URL="https://example.test/v1",
        )
    ).settings
    assert settings.judgment.provider == "typesafe"
    assert settings.judgment.model == "jev-latest"
    assert settings.api_key_for("typesafe") == "key-1"
    assert settings.base_url_for("typesafe") == "https://example.test/v1"

    local = load_settings(
        env=_env(
            tmp_path,
            MNEMOS_LLM_PROVIDER="ollama",
            MNEMOS_LLM_MODEL="qwen3.5:9b",
            MNEMOS_JUDGMENT_PROVIDER="ollama",
        )
    ).settings
    assert local.judgment.model == "qwen3.5:9b"
    assert local.embedding.provider == "ollama"


def test_the_builder_makes_each_backend_and_refuses_a_missing_key() -> None:
    firewall = MemoryWriteFirewall(MemorySafetyConfig())
    typesafe = AppSettings.model_validate(
        {"judgment": {"provider": "typesafe"}, "providers": {"typesafe": {"api_key": "k"}}}
    )
    judge = build_judge_from_settings(typesafe, firewall=firewall)
    assert isinstance(judge, TypeSafeJudge) and judge.model == "jev-latest"

    local = AppSettings.model_validate({"judgment": {"provider": "ollama", "model": "m"}})
    assert isinstance(build_judge_from_settings(local, firewall=firewall), OllamaJudge)

    assert isinstance(build_judge_from_settings(AppSettings(), firewall=firewall), UnavailableJudge)

    missing = AppSettings.model_validate({"judgment": {"provider": "typesafe"}})
    with pytest.raises(ValueError, match="MNEMOS_TYPESAFE_API_KEY"):
        build_judge_from_settings(missing, firewall=firewall)


def test_judged_extraction_needs_a_judge(tmp_path: Path) -> None:
    config = tmp_path / "mnemos.toml"
    config.write_text('[sleep]\nextraction_mode = "judged"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="extraction_mode=judged needs a judgment provider"):
        build_configured_engine(
            env={"MNEMOS_CONFIG_PATH": str(config), "MNEMOS_SQLITE_PATH": str(tmp_path / "m.db")},
            cwd=tmp_path,
        )


def test_a_configured_judge_is_owned_and_closed_with_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[str] = []

    async def record_close(self: TypeSafeJudge) -> None:
        closed.append(self.model)

    monkeypatch.setattr(TypeSafeJudge, "aclose", record_close)
    engine = build_configured_engine(
        env=_env(
            tmp_path,
            MNEMOS_LLM_PROVIDER="ollama",
            MNEMOS_JUDGMENT_PROVIDER="typesafe",
            MNEMOS_TYPESAFE_API_KEY="key-1",
        ),
        cwd=tmp_path,
    )
    try:
        assert isinstance(engine.judge, TypeSafeJudge)
    finally:
        asyncio.run(engine.aclose())
    assert closed == ["jev-latest"]


def test_health_reports_judgment_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    off = run_health_checks(env=_env(tmp_path))
    assert off["provider_state"]["judgment"]["status"] == "off"
    assert not [c for c in off["checks"] if c["name"].startswith("judgment.")]

    missing_key = run_health_checks(env=_env(tmp_path, MNEMOS_JUDGMENT_PROVIDER="typesafe"))
    (check,) = [c for c in missing_key["checks"] if c["name"] == "judgment.typesafe"]
    assert check["status"] == "fail" and "MNEMOS_TYPESAFE_API_KEY" in check["message"]

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        assert url == "https://api.typesafe.ai/v1/systemone"
        return httpx.Response(
            200,
            json={"model": "jev-1.13.0", "answers": {"ready": {"type": "noul", "noul": 0.99}}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    probed = run_health_checks(
        env=_env(tmp_path, MNEMOS_JUDGMENT_PROVIDER="typesafe", MNEMOS_TYPESAFE_API_KEY="k"),
        probe=True,
    )
    state = probed["provider_state"]["judgment"]
    assert state["status"] == "ready" and state["observed_model"] == "jev-1.13.0"
    (check,) = [c for c in probed["checks"] if c["name"] == "judgment.typesafe"]
    assert check["status"] == "pass" and "leave the machine" in check["message"]


def test_novelty_surprisal_needs_a_judge_and_flows_into_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="surprisal_mode=novelty needs a judgment provider"):
        build_configured_engine(env=_env(tmp_path, MNEMOS_SURPRISAL_MODE="novelty"), cwd=tmp_path)

    config = tmp_path / "mnemos.toml"
    config.write_text(
        '[judgment]\nprovider = "ollama"\n[judgment.thresholds]\nkind_min_probability = 0.7\n',
        encoding="utf-8",
    )
    engine = build_configured_engine(
        env={
            "MNEMOS_CONFIG_PATH": str(config),
            "MNEMOS_SQLITE_PATH": str(tmp_path / "m.db"),
            "MNEMOS_LLM_PROVIDER": "ollama",
            "MNEMOS_SURPRISAL_MODE": "novelty",
        },
        cwd=tmp_path,
    )
    try:
        assert engine.config.surprisal.mode == "novelty"
        assert engine.config.judgment.kind_min_probability == 0.7
        assert isinstance(engine.judge, OllamaJudge)
    finally:
        asyncio.run(engine.aclose())
