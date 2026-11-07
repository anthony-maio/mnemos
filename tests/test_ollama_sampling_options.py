"""Explicit sampling controls reach both routes without changing defaults."""

import pytest

from mnemos.utils.llm import OllamaProvider
from tests.test_ollama_structured_reasoning import install_client


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
async def test_sampling_is_copied_forwarded_and_replaced_per_call(monkeypatch, chat):
    settings = {
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0,
        "repeat_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "seed": 42,
    }
    original = dict(settings)
    provider = OllamaProvider(sampling_options=settings, force_chat=chat)
    settings["seed"] = 99
    requests = install_client(monkeypatch, {"response": "ok", "message": {"content": "ok"}})
    await provider.predict("text")
    await provider.predict("text", sampling_options={"seed": 7})
    await provider.predict("text", sampling_options=None)
    assert requests[0][1]["options"] == {"temperature": 0.1, "num_predict": 4096, **original}
    assert requests[1][1]["options"] == {"temperature": 0.1, "num_predict": 4096, "seed": 7}
    assert requests[2][1]["options"] == {"temperature": 0.1, "num_predict": 4096}
    assert provider.sampling_options == original
    assert all(url.endswith("/chat" if chat else "/generate") for url, _ in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        [],
        "seed",
        {"num_ctx": 8192},
        {"temperature": 0},
        {"num_predict": 1},
        {"seed": True},
        {"seed": 2.0},
        {"seed": -2},
        {"seed": 2**32},
        {"top_k": False},
        {"top_k": -1},
        {"top_k": 2.5},
        {"top_p": -0.1},
        {"top_p": 1.1},
        {"min_p": -0.1},
        {"min_p": 1.1},
        {"repeat_penalty": -1},
        {"presence_penalty": float("inf")},
        {"frequency_penalty": float("nan")},
        {"top_p": "0.9"},
        {"min_p": True},
    ],
)
async def test_invalid_sampling_stops_before_io(monkeypatch, settings):
    requests = install_client(monkeypatch, {"response": "ok"})
    with pytest.raises(ValueError, match="sampling"):
        OllamaProvider(sampling_options=settings)
    with pytest.raises(ValueError, match="sampling"):
        await OllamaProvider().predict("text", sampling_options=settings)
    assert requests == []


@pytest.mark.asyncio
async def test_sampling_boundary_values_remain_numeric(monkeypatch):
    settings = {
        "seed": -1,
        "top_k": 0,
        "top_p": 0,
        "min_p": 1,
        "repeat_penalty": 0,
        "presence_penalty": -2,
        "frequency_penalty": 2,
    }
    requests = install_client(monkeypatch, {"response": "ok"})
    await OllamaProvider(sampling_options=settings).predict("text")
    assert all(requests[0][1]["options"][k] == v for k, v in settings.items())
