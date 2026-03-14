"""
mnemos/utils/llm.py — LLM provider abstractions for the Mnemos memory system.

The LLM providers are used by multiple Mnemos modules:
- SurprisalGate: predicts next user intent for prediction error calculation
- AffectiveRouter: classifies emotional/cognitive state of interactions
- MutableRAG: evaluates whether retrieved facts need reconsolidation
- SleepDaemon: extracts permanent facts from episodic memory buffer
- SleepDaemon: generates procedural tool code from repeated patterns

The MockLLMProvider enables fully offline development and deterministic testing.
The OllamaProvider connects to locally-running Ollama (no API costs).
The OpenAIProvider supports any OpenAI-compatible API endpoint.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..observability import log_event
from .reliability import MnemosConfigurationError, RetryPolicy, call_with_async_retry


def _provider_name_from_base_url(base_url: str) -> str:
    lower = base_url.lower()
    if "openrouter" in lower:
        return "openrouter"
    if "openclaw" in lower:
        return "openclaw"
    return "openai"


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement:
    - predict(): generate a text completion for a given prompt
    - classify(): score a text against a set of labels (soft classification)

    Both methods are async to support concurrent background processing
    (e.g., SleepDaemon consolidation running while main processing continues).
    """

    @abstractmethod
    async def predict(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a text completion for the given prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)

        Returns:
            The generated text response as a string.
        """
        ...

    @abstractmethod
    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        """
        Soft-classify a prompt against a set of labels, returning confidence scores.

        Args:
            prompt: The text to classify.
            labels: List of candidate labels.

        Returns:
            Dict mapping each label to a confidence score in [0, 1].
            Scores should sum to approximately 1.0 but this is not strictly required.
        """
        ...


class MockLLMProvider(LLMProvider):
    """
    Deterministic mock LLM provider for testing and development.

    Returns pre-configured responses based on simple keyword matching.
    This enables running the full Mnemos pipeline without any LLM backend,
    making all tests fast and fully offline.

    Args:
        responses: Optional dict mapping keywords to responses.
            If a prompt contains a keyword (case-insensitive), the corresponding
            response is returned. Falls back to default_response.
        default_response: Response to return when no keyword matches.
        predict_call_log: If True, appends each prompt to self.calls for inspection.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str = "This is a mock response.",
        record_calls: bool = False,
    ) -> None:
        self._responses = responses or {}
        self._default = default_response
        self._record = record_calls
        self.calls: list[str] = []  # Populated when record_calls=True

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        """
        Return a deterministic response based on keyword matching.

        Special handling for common Mnemos prompt patterns:
        - Surprisal prediction: returns a plausible next-turn prediction
        - Reconsolidation check: returns UNCHANGED by default (conservative)
        - Consolidation: returns a numbered list of mock facts
        - Affective classification: returns neutral scores (0.0, 0.5, 0.5)
        """
        if self._record:
            self.calls.append(prompt)

        prompt_lower = prompt.lower()

        # Check custom keyword responses first
        for keyword, response in self._responses.items():
            if keyword.lower() in prompt_lower:
                return response

        # Built-in pattern matching for Mnemos-specific prompts
        if "predict what" in prompt_lower or "next intent" in prompt_lower:
            return "The user will likely ask about their current project or task."

        if (
            "has this stored memory" in prompt_lower
            or "has this information changed" in prompt_lower
        ):
            return "UNCHANGED"

        if "extract permanent facts" in prompt_lower or "consolidation" in prompt_lower:
            return (
                "1. The user is working on a software project.\n"
                "2. The user prefers concise, technical responses.\n"
                "3. The user values code examples over explanations."
            )

        if "valence" in prompt_lower and "arousal" in prompt_lower:
            # Affective classification — return neutral scores
            return "0.0, 0.5, 0.5"

        if "repeated reasoning" in prompt_lower or "proceduralize" in prompt_lower:
            return "NO_PATTERN"

        return self._default

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        """
        Return equal-weight classification across all labels.

        For testing, this provides a neutral baseline where no label is preferred.
        """
        if self._record:
            self.calls.append(f"CLASSIFY: {prompt}")

        if not labels:
            return {}

        weight = 1.0 / len(labels)
        return {label: weight for label in labels}


class OllamaProvider(LLMProvider):
    """
    LLM provider using a locally-running Ollama server.

    Ollama (https://ollama.ai) allows running open-weight models like Llama 3,
    Mistral, and Gemma locally with zero API cost. This is the recommended
    provider for Mnemos' background cognitive processing (surprisal prediction,
    sleep consolidation) since continuous processing on proprietary APIs is
    cost-prohibitive.

    Args:
        model: Ollama model name (e.g., "llama3", "mistral", "gemma2").
        base_url: Ollama API base URL (default: http://localhost:11434).
        timeout: HTTP request timeout in seconds.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum tokens to generate per response.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion using Ollama's /api/generate endpoint.

        Args:
            prompt: Input prompt string.
            **kwargs: Override temperature, max_tokens, or other options.

        Returns:
            Generated text (stripped of leading/trailing whitespace).

        Raises:
            httpx.HTTPStatusError: If Ollama returns an error status.
            httpx.ConnectError: If Ollama is not running.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async def _request() -> str:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("response", "")
                if not isinstance(text, str):
                    text = str(text)
                return text.strip()

        try:
            return await call_with_async_retry(
                provider="ollama",
                operation="predict",
                fn=_request,
                policy=RetryPolicy(),
            )
        except Exception as exc:
            log_event(
                "mnemos.provider_failure",
                level=logging.ERROR,
                provider="ollama",
                operation="predict",
                error=str(exc),
            )
            raise

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        """
        Classify text by asking the LLM to score each label.

        Constructs a prompt asking the model to rate the relevance of each label
        and parses the response into a score dict.

        Args:
            prompt: Text to classify.
            labels: Candidate labels to score.

        Returns:
            Dict mapping labels to confidence scores in [0, 1].
        """
        label_list = ", ".join(f'"{l}"' for l in labels)
        classify_prompt = (
            f"Rate the relevance of each label to the following text on a scale of 0.0 to 1.0.\n"
            f"Text: {prompt}\n"
            f"Labels: {label_list}\n"
            f'Reply with JSON only, like: {{"label": score, ...}}'
        )

        try:
            result = await self.predict(classify_prompt)
            # Try to parse JSON from the response
            json_match = re.search(r"\{[^}]+\}", result)
            if json_match:
                scores = json.loads(json_match.group())
                # Normalize and filter to known labels
                return {label: float(scores.get(label, 0.0)) for label in labels}
        except Exception:
            pass

        # Fallback: equal weights
        weight = 1.0 / len(labels) if labels else 0.0
        return {label: weight for label in labels}


class OpenAIProvider(LLMProvider):
    """
    LLM provider using any OpenAI-compatible chat completion API.

    Works with:
    - OpenAI (api.openai.com)
    - Azure OpenAI
    - Groq (api.groq.com/openai)
    - Together AI
    - Any other OpenAI-compatible endpoint

    Args:
        api_key: API authentication key.
        model: Model name (e.g., "gpt-4o-mini", "gpt-3.5-turbo").
        base_url: API base URL (default: OpenAI's production endpoint).
        timeout: HTTP request timeout in seconds.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens per response.
        system_prompt: Optional system prompt prepended to all requests.
    """

    def __init__(
        self,
        api_key: str,
        api_key_fallback: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        temperature: float = 0.1,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_key_fallback = api_key_fallback
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant with expertise in memory management and information synthesis."
        )

    def _headers(self, api_key: str | None = None) -> dict[str, str]:
        key = self.api_key if api_key is None else api_key
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _fallback_key_for_auth_retry(self, provider_name: str) -> str | None:
        fallback_key = self.api_key_fallback
        if provider_name != "openrouter":
            return None
        if fallback_key in (None, "", self.api_key):
            return None
        return fallback_key

    async def _predict_with_key(
        self,
        *,
        prompt: str,
        payload: dict[str, Any],
        api_key: str,
        provider_name: str,
    ) -> str:
        headers = self._headers(api_key)

        async def _request() -> str:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    content = str(content)
                return content.strip()

        _ = prompt
        return await call_with_async_retry(
            provider=provider_name,
            operation="predict",
            fn=_request,
            policy=RetryPolicy(),
        )

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion using the OpenAI chat completions API.

        Args:
            prompt: Input prompt string (sent as user message).
            **kwargs: Override temperature, max_tokens, or model.

        Returns:
            Generated assistant message content.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        model = kwargs.get("model", self.model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        provider_name = _provider_name_from_base_url(self.base_url)

        try:
            return await self._predict_with_key(
                prompt=prompt,
                payload=payload,
                api_key=self.api_key,
                provider_name=provider_name,
            )
        except MnemosConfigurationError as exc:
            error: Exception = exc
            fallback_key = self._fallback_key_for_auth_retry(provider_name)
            if fallback_key is not None:
                try:
                    content = await self._predict_with_key(
                        prompt=prompt,
                        payload=payload,
                        api_key=fallback_key,
                        provider_name=provider_name,
                    )
                    self.api_key = fallback_key
                    return content
                except Exception as retry_exc:
                    error = retry_exc
            log_event(
                "mnemos.provider_failure",
                level=logging.ERROR,
                provider=provider_name,
                operation="predict",
                error=str(error),
            )
            raise error
        except Exception as exc:
            log_event(
                "mnemos.provider_failure",
                level=logging.ERROR,
                provider=provider_name,
                operation="predict",
                error=str(exc),
            )
            raise

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        """
        Classify text using the OpenAI API with JSON mode.

        Args:
            prompt: Text to classify.
            labels: Candidate labels to score.

        Returns:
            Dict mapping labels to confidence scores in [0, 1].
        """
        label_list = ", ".join(f'"{l}"' for l in labels)
        classify_prompt = (
            f"Rate the relevance of each label to this text on a scale of 0.0 to 1.0.\n"
            f"Text: {prompt}\n"
            f"Labels: {label_list}\n"
            f'Reply ONLY with JSON: {{"label": score}}'
        )

        try:
            result = await self.predict(classify_prompt)
            json_match = re.search(r"\{[^}]+\}", result, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                return {label: float(scores.get(label, 0.0)) for label in labels}
        except Exception:
            pass

        weight = 1.0 / len(labels) if labels else 0.0
        return {label: weight for label in labels}
