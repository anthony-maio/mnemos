"""Regressions for the truthful application runtime factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemos.engine import MnemosEngine
from mnemos.runtime import build_configured_engine
from mnemos.types import Interaction
from mnemos.utils.llm import MockLLMProvider, UnavailableLLMProvider


def test_configured_engine_uses_truthful_pending_llm_when_no_provider_is_configured(
    tmp_path: Path,
) -> None:
    engine = build_configured_engine(
        env={
            "MNEMOS_CONFIG_PATH": str(tmp_path / "missing.toml"),
            "MNEMOS_SQLITE_PATH": str(tmp_path / "memory.db"),
        },
        cwd=tmp_path,
    )

    assert isinstance(engine, MnemosEngine)
    assert isinstance(engine.llm, UnavailableLLMProvider)
    assert engine.store.db_path == str(tmp_path / "memory.db")


def test_configured_engine_requires_explicit_opt_in_for_mock(
    tmp_path: Path,
) -> None:
    env = {
        "MNEMOS_CONFIG_PATH": str(tmp_path / "missing.toml"),
        "MNEMOS_SQLITE_PATH": str(tmp_path / "memory.db"),
        "MNEMOS_LLM_PROVIDER": "mock",
    }

    with pytest.raises(ValueError, match="allow_mock"):
        build_configured_engine(env=env, cwd=tmp_path)

    engine = build_configured_engine(env=env, cwd=tmp_path, allow_mock=True)
    assert isinstance(engine.llm, MockLLMProvider)


@pytest.mark.asyncio
async def test_missing_model_captures_episode_without_fabricating_fact(tmp_path: Path) -> None:
    engine = build_configured_engine(
        env={
            "MNEMOS_CONFIG_PATH": str(tmp_path / "missing.toml"),
            "MNEMOS_SQLITE_PATH": str(tmp_path / "memory.db"),
        },
        cwd=tmp_path,
    )

    result = await engine.process(
        Interaction(role="user", content="I prefer Rust for service development.")
    )

    assert result.stored is False
    assert "unavailable" in result.reason.lower()
    assert engine.store.get_all() == []
    engine.store.close()
