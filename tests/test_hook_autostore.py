"""
tests/test_hook_autostore.py — Hook payload auto-store decision tests.
"""

from __future__ import annotations

from mnemos.hook_autostore import decide_autostore, parse_hook_payload


def test_parse_hook_payload_handles_invalid_json() -> None:
    assert parse_hook_payload("not-json") == {}


def test_decide_autostore_user_prompt_stores_high_signal() -> None:
    decision = decide_autostore(
        event="UserPromptSubmit",
        payload={
            "prompt": "For this repository, use uv and mypy for Python quality checks.",
            "cwd": "/tmp/repo-alpha",
        },
    )
    assert decision.should_store is True
    assert decision.interaction is not None
    assert decision.scope == "project"
    assert decision.scope_id == "repo-alpha"


def test_decide_autostore_user_prompt_skips_low_signal() -> None:
    decision = decide_autostore(
        event="UserPromptSubmit",
        payload={"prompt": "ok"},
    )
    assert decision.should_store is False
    assert "low-signal" in decision.reason


def test_decide_autostore_user_prompt_skips_sensitive_content() -> None:
    decision = decide_autostore(
        event="UserPromptSubmit",
        payload={"prompt": "api_key=supersecretvalue"},
    )
    assert decision.should_store is False
    assert "safety policy" in decision.reason.lower()


def test_decide_autostore_post_tool_use_stores_failures() -> None:
    decision = decide_autostore(
        event="PostToolUse",
        payload={
            "tool_name": "Bash",
            "output": "Command failed with error: pip install timed out",
            "cwd": "/tmp/repo-beta",
        },
    )
    assert decision.should_store is True
    assert decision.interaction is not None
    assert "Tool failure [Bash]" in decision.interaction.content
    assert decision.scope_id == "repo-beta"


def test_decide_autostore_post_tool_use_skips_non_failure() -> None:
    decision = decide_autostore(
        event="PostToolUse",
        payload={
            "tool_name": "Bash",
            "output": "Command completed successfully",
        },
    )
    assert decision.should_store is False
    assert "non-failure" in decision.reason
