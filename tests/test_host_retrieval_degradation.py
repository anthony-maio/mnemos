"""Host adapters preserve degraded retrieval status without losing recall text."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from mnemos.adapters.claude_code import ClaudeCodeAdapter
from mnemos.adapters.hermes import MnemosHermesProvider
from mnemos.scopes import ProjectIdentity, ScopeRegistry, canonical_project_root


def _registry(path: Path) -> ScopeRegistry:
    return ScopeRegistry(
        projects={
            "project-a": ProjectIdentity(
                id="project-a",
                root=canonical_project_root(path),
                user_profile_id="profile-a",
                enabled=True,
            )
        }
    )


class _DegradedClient:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(self.result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        {"degraded": True, "context": "[fact-1] Existing recall\n"},
        {"status": "degraded", "context": ""},
    ],
)
async def test_claude_retrieve_preserves_context_and_degraded_status(
    tmp_path: Path, result: dict[str, Any]
) -> None:
    adapter = ClaudeCodeAdapter(
        client=_DegradedClient(result),
        registry=_registry(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )

    hook_result = await adapter.retrieve_context(
        "UserPromptSubmit",
        {"session_id": "session-a", "cwd": str(tmp_path), "prompt": "Recall this"},
    )

    assert hook_result.status == "degraded"
    assert hook_result.context == result.get("context", "")


@pytest.mark.parametrize(
    "result",
    [
        {"degraded": True, "context": "[fact-1] Existing recall\n"},
        {"status": "degraded", "context": ""},
    ],
)
def test_hermes_prefetch_records_content_free_degraded_retrieval(
    tmp_path: Path, result: dict[str, Any]
) -> None:
    registry = _registry(tmp_path)
    provider = MnemosHermesProvider(
        client=_DegradedClient(result),
        registry=registry,
        user_profile_id="profile-a",
    )
    provider.initialize("hermes-session", hermes_home=str(tmp_path / "hermes"), cwd=str(tmp_path))
    try:
        rendered = provider.prefetch("private query text")

        assert rendered == result.get("context", "")
        diagnostic = provider.get_lifecycle_diagnostics()[-1]
        assert diagnostic["action"] == "retrieve"
        assert diagnostic["status"] == "retrieval_degraded"
        assert diagnostic["target_digest"] == hashlib.sha256(b"hermes-session").hexdigest()
        assert diagnostic["target_digest"] != hashlib.sha256(b"private query text").hexdigest()
    finally:
        provider.shutdown()
