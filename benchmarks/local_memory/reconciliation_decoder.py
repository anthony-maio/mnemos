"""Benchmark-only decoder controls; response validation remains in reconciliation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from mnemos.utils.llm import LLMProvider

DecoderMode = Literal["bounded", "reversed", "json", "none"]


class DecoderProvider(LLMProvider):
    """Change only the outbound format, leaving prompts and responses untouched."""

    supports_structured_output = True

    def __init__(self, provider: LLMProvider, *, mode: DecoderMode) -> None:
        if mode not in {"bounded", "reversed", "json", "none"}:
            raise ValueError("Unknown decoder mode")
        self.provider = provider
        self.mode = mode

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        forwarded = dict(kwargs)
        if self.mode == "reversed":
            schema = deepcopy(forwarded["format"])
            schema["properties"]["decisions"]["items"]["oneOf"].reverse()
            forwarded["format"] = schema
        elif self.mode == "json":
            forwarded["format"] = "json"
        elif self.mode == "none":
            forwarded.pop("format", None)
        return await self.provider.predict(prompt, **forwarded)

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        return await self.provider.classify(prompt, labels)
