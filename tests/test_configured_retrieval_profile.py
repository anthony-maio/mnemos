"""Hosts retrieve with the configured profile unless a caller asks for one."""

from __future__ import annotations

from pathlib import Path

from tests.test_codex_adapter import RecordingClient, make_adapter
from tests.test_suppression_ordering import _remember, _service, _settle


async def _retrieve(tmp_path: Path, **extra: str) -> dict[str, object]:
    memory, context = _service(tmp_path)
    memory.engine.config.retrieval.profile = "biomimetic"
    try:
        await _remember(memory, context, "deploys", "Deploys run on Thursdays.")
        await _settle(memory)
        return await memory.dispatch(
            "retrieve", {"context": context, "query": "When do deploys run?", **extra}
        )
    finally:
        memory.engine.store.close()


async def test_retrieval_without_a_profile_uses_the_configured_one(tmp_path: Path) -> None:
    response = await _retrieve(tmp_path)
    assert response["profile"] == "biomimetic"
    facts = response["facts"]
    assert isinstance(facts, list) and facts
    assert facts[0]["scores"]["profile"] == "biomimetic"


async def test_an_explicit_profile_still_overrides_the_configured_one(tmp_path: Path) -> None:
    response = await _retrieve(tmp_path, profile="baseline")
    assert response["profile"] == "baseline"
    facts = response["facts"]
    assert isinstance(facts, list) and facts
    assert facts[0]["scores"]["profile"] == "baseline"


async def test_codex_prompt_recall_leaves_the_profile_to_configuration(tmp_path: Path) -> None:
    client = RecordingClient()
    adapter, payload = make_adapter(tmp_path, client)
    await adapter.handle(
        "UserPromptSubmit",
        {**payload, "prompt": "Which deploy day?", "hook_event_name": "UserPromptSubmit"},
    )
    assert client.retrieves and "profile" not in client.retrieves[0]
