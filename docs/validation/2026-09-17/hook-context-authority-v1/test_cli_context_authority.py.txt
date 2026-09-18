from mnemos.cli import _format_retrieved_context
from mnemos.types import MemoryChunk


def test_cli_context_preserves_unknown_authority_and_scope() -> None:
    captured = MemoryChunk(
        id="captured-id",
        version=2,
        content="Assistant reported: use maple-heron for release labels.",
        metadata={
            "scope": "project",
            "scope_id": "project-a",
            "source": "captured_passthrough",
            "assertion_status": "unknown",
        },
    )

    rendered = _format_retrieved_context([captured], "UserPromptSubmit")

    assert "Mnemos memory (UserPromptSubmit) -- 1 relevant prior memories:" in rendered
    assert "[project:project-a]" in rendered
    assert "captured conversation; unverified, may be outdated" in rendered
    assert "version 2" in rendered
    assert captured.content in rendered
