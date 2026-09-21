"""Hermes setup changes memory.provider and leaves every other setting as written."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mnemos.hosts import _merge_hermes_config, apply_host_integration


def _settings(text: str) -> object:
    return yaml.safe_load(text)


@pytest.mark.parametrize(
    ("original", "merged"),
    [
        (
            # A comment on the memory key used to hide the block, and a second
            # memory key was appended that dropped every setting in the first.
            "model:\n  default: anthropic/claude-opus-4\n"
            "memory:  # built-in memory settings\n"
            "  memory_enabled: true\n  user_char_limit: 1375\n  provider: honcho\n",
            "model:\n  default: anthropic/claude-opus-4\n"
            "memory:  # built-in memory settings\n"
            "  memory_enabled: true\n  user_char_limit: 1375\n  provider: mnemos\n",
        ),
        (
            # The inserted line used to be two spaces deep in a four space block.
            "memory:\n    memory_enabled: true\n    memory_char_limit: 2200\n",
            "memory:\n    provider: mnemos\n    memory_enabled: true\n    memory_char_limit: 2200\n",
        ),
        (
            # A nested provider belongs to embeddings, not to memory.
            "memory:\n  embeddings:\n    provider: openai\n  memory_enabled: true\n",
            "memory:\n  provider: mnemos\n  embeddings:\n    provider: openai\n"
            "  memory_enabled: true\n",
        ),
        (
            "memory:\n  embeddings:\n    provider: openai\n  provider: builtin  # default\n",
            "memory:\n  embeddings:\n    provider: openai\n  provider: mnemos  # default\n",
        ),
        (
            "# Hermes settings\ndisplay:\n  streaming: true\n",
            "# Hermes settings\ndisplay:\n  streaming: true\n\nmemory:\n  provider: mnemos\n",
        ),
        ("", "memory:\n  provider: mnemos\n"),
    ],
)
def test_only_memory_provider_changes(original: str, merged: str) -> None:
    result = _merge_hermes_config(original)

    assert result == merged
    before = _settings(original) or {}
    assert isinstance(before, dict)
    memory = before.get("memory") or {}
    assert _settings(result) == {**before, "memory": {**memory, "provider": "mnemos"}}
    assert _merge_hermes_config(result) == result


@pytest.mark.parametrize(
    "original",
    [
        "memory: {memory_enabled: true}\n",
        "memory: false\n",
        "memory:\n  provider: 'a#b'\n",
        "- not\n- a mapping\n",
        "memory: [unclosed\n",
    ],
)
def test_layouts_it_cannot_edit_are_refused(original: str) -> None:
    with pytest.raises(ValueError, match="nothing was written"):
        _merge_hermes_config(original)


def test_a_flow_mapping_already_pointing_at_mnemos_is_left_alone() -> None:
    original = "memory: {provider: mnemos, memory_enabled: true}\n"
    assert _merge_hermes_config(original) == original


def test_a_refused_setup_leaves_the_file_untouched(tmp_path: Path) -> None:
    home = tmp_path / "home"
    config = home / ".hermes" / "config.yaml"
    config.parent.mkdir(parents=True)
    original = b"memory: {memory_enabled: true, user_char_limit: 1375}\n"
    config.write_bytes(original)

    with pytest.raises(ValueError, match="nothing was written"):
        apply_host_integration(
            "hermes", mnemos_config_path=tmp_path / "Mnemos" / "mnemos.toml", home=home
        )

    assert config.read_bytes() == original
    assert list(config.parent.iterdir()) == [config]
