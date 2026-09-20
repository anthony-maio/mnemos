from __future__ import annotations

from typing import Any

import pytest

from mnemos.modules.affective import AffectiveRouter
from mnemos.types import Interaction, MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import LLMProvider


class _ResponseProvider(LLMProvider):
    def __init__(self, response: str) -> None:
        self.response = response

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        return self.response

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        return {label: 0.0 for label in labels}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("-.3, .8, .6", (-0.3, 0.8, 0.6)),
        (" +1.2e-1, 8E-1, 6e-1 ", (0.12, 0.8, 0.6)),
        ("-2, 2, 3", (-1.0, 1.0, 1.0)),
    ],
)
async def test_classify_state_preserves_signed_decimal_and_scientific_values(
    response: str, expected: tuple[float, float, float]
) -> None:
    router = AffectiveRouter(_ResponseProvider(response), SimpleEmbeddingProvider(dim=8))

    state = await router.classify_state(Interaction(role="user", content="hello"))

    assert state is not None
    assert (state.valence, state.arousal, state.complexity) == expected
    assert len(router._state_history) == 1
    chunk = router.tag_chunk(MemoryChunk(content="hello"), state)
    assert chunk.cognitive_state_status == "measured"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "valence=-0.3, arousal=0.8, complexity=0.6",
        "-.3, .8",
        "-.3, .8, .6, 0.1",
        "NaN, .8, .6",
        "-.3, inf, .6",
        "1e999, .8, .6",
        "1, 2, 3 because this is a score",
    ],
)
async def test_malformed_or_ambiguous_response_is_unavailable(response: str) -> None:
    provider = _ResponseProvider("-.2, .4, .6")
    router = AffectiveRouter(provider, SimpleEmbeddingProvider(dim=8))

    prior = await router.classify_state(Interaction(role="user", content="prior"))
    assert prior is not None
    provider.response = response

    state = await router.classify_state(Interaction(role="user", content="hello"))
    chunk = router.tag_chunk(MemoryChunk(content="hello"), state)

    assert state is None
    assert list(router._state_history) == [prior]
    assert chunk.cognitive_state is None
    assert chunk.cognitive_state_status == "provider_unavailable"
    assert "cognitive_state" not in chunk.metadata
