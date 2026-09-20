"""Structured reasoning uses Ollama's final chat channel, never its scratchpad."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from mnemos.utils.llm import OllamaProvider
from mnemos.utils.reliability import MnemosExternalServiceError


def install_client(
    monkeypatch: pytest.MonkeyPatch, response: object
) -> list[tuple[str, dict[str, Any]]]:
    requests: list[tuple[str, dict[str, Any]]] = []

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> object:
            return response

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def post(self, url: str, *, json: dict[str, Any]) -> Response:
            requests.append((url, json))
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    return requests


@pytest.mark.asyncio
@pytest.mark.parametrize("think", [True, "low", "medium", "high", "max"])
async def test_structured_reasoning_uses_chat_final_with_original_options(
    monkeypatch: pytest.MonkeyPatch,
    think: Any,
) -> None:
    raw = {
        "message": {"role": "assistant", "thinking": "not an answer", "content": ' {"ok":true} '}
    }
    requests = install_client(monkeypatch, raw)
    observed: list[dict[str, Any]] = []
    provider = OllamaProvider(
        "default", "http://local", response_observer=observed.append, think=think
    )
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    result = await provider.predict(
        "extract", model="classifier", format=schema, temperature=0, max_tokens=8192
    )
    assert result == '{"ok":true}'
    assert observed == [raw]
    assert requests == [
        (
            "http://local/api/chat",
            {
                "model": "classifier",
                "messages": [{"role": "user", "content": "extract"}],
                "stream": False,
                "think": think,
                "format": schema,
                "options": {"temperature": 0, "num_predict": 8192},
            },
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        None,
        [],
        {},
        {"content": None},
        {"content": ""},
        {"content": "   "},
        {"content": {"ok": True}},
    ],
)
async def test_structured_reasoning_rejects_missing_or_invalid_final(
    monkeypatch: pytest.MonkeyPatch,
    message: object,
) -> None:
    raw = {"response": '{"ok":true}', "thinking": "top-level scratchpad", "message": message}
    requests = install_client(monkeypatch, raw)
    provider = OllamaProvider(think=True)
    with pytest.raises(MnemosExternalServiceError, match="final response"):
        await provider.predict("extract", format={"type": "object"})
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_per_call_reasoning_mode_controls_route_without_changing_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OllamaProvider(think=False)
    requests = install_client(monkeypatch, {"message": {"content": "answer"}})
    assert await provider.predict("extract", think=True, format={"type": "object"}) == "answer"
    assert requests[0][0].endswith("/api/chat")
    assert provider.think is False
    requests = install_client(monkeypatch, {"response": "answer"})
    assert await provider.predict("extract", format={"type": "object"}) == "answer"
    assert requests[0][0].endswith("/api/generate")


@pytest.mark.asyncio
@pytest.mark.parametrize("think,output_format", [(None, {}), (False, {}), (True, None)])
async def test_other_requests_keep_generate_contract(
    monkeypatch: pytest.MonkeyPatch,
    think: Any,
    output_format: object,
) -> None:
    requests = install_client(monkeypatch, {"response": "answer"})
    assert await OllamaProvider(think=think).predict("extract", format=output_format) == "answer"
    assert requests[0][0].endswith("/api/generate")
    assert requests[0][1]["prompt"] == "extract"
