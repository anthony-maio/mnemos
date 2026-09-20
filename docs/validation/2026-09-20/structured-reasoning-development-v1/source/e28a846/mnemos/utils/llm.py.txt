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
import os
import re
import ssl
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal, cast

import httpx

from ..observability import log_event
from .reliability import (
    MnemosConfigurationError,
    MnemosExternalServiceError,
    RetryPolicy,
    call_with_async_retry,
)


def _provider_name_from_base_url(base_url: str) -> str:
    lower = base_url.lower()
    if "openrouter" in lower:
        return "openrouter"
    if "openclaw" in lower:
        return "openclaw"
    return "openai"


def _request_model(default: str, options: dict[str, Any]) -> str:
    model = options.get("model", default)
    if not isinstance(model, str) or not model.strip():
        raise MnemosConfigurationError("A nonempty model name is required")
    return model.strip()


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement:
    - predict(): generate a text completion for a given prompt
    - classify(): score a text against a set of labels (soft classification)

    Both methods are async to support concurrent background processing
    (e.g., SleepDaemon consolidation running while main processing continues).
    """

    # Providers that can enforce a response schema override this capability.
    # Callers keep prompt-only behavior for providers that do not advertise it.
    supports_structured_output = False
    # An explicit selection must never disappear in a provider's **kwargs.
    supports_model_override = False

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


class UnavailableLLMProvider(LLMProvider):
    """An honest provider for a runtime without a configured local model.

    The engine can still capture an episode and keep existing recall available
    while cognitive enrichment waits for setup.  This provider deliberately
    raises instead of returning placeholder text, so a missing model can never
    manufacture a user preference or durable fact.
    """

    def __init__(self, reason: str = "No real LLM provider is configured") -> None:
        self.reason = reason

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        _ = prompt, kwargs
        raise MnemosConfigurationError(self.reason)

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        _ = prompt, labels
        raise MnemosConfigurationError(self.reason)


_OLLAMA_THINK_LEVELS = frozenset({"low", "medium", "high", "max"})


def _validate_ollama_think(
    value: Any,
) -> bool | Literal["low", "medium", "high", "max"] | None:
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str) and value in _OLLAMA_THINK_LEVELS:
        return cast(Literal["low", "medium", "high", "max"], value)
    raise ValueError("think must be a boolean or one of low, medium, high, max")


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
        think: Optional Ollama reasoning mode. ``None`` leaves the model default
            unchanged; booleans and documented reasoning levels are supported.
    """

    supports_structured_output = True
    supports_model_override = True

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        temperature: float = 0.1,
        max_tokens: int = 512,
        response_observer: Callable[[dict[str, Any]], None] | None = None,
        *,
        think: bool | Literal["low", "medium", "high", "max"] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.think = _validate_ollama_think(think)
        self.response_observer = response_observer
        self._default_trust_context: ssl.SSLContext | None = None
        self._trust_context_lock = threading.Lock()

    def _tls_verification(self) -> ssl.SSLContext | bool:
        """Reuse default CA loading without retaining an event-loop-bound client.

        Custom trust overrides still use HTTPX's per-request loading, including
        changes to a certificate file at the same path. The default CA context
        lasts for this provider's lifetime, as it would with a persistent client.
        """
        if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
            return True
        with self._trust_context_lock:
            if self._default_trust_context is None:
                # Cache only the default bundle, never a transient environment
                # override. AsyncClient itself keeps trust_env's default True.
                self._default_trust_context = httpx.create_ssl_context(verify=True, trust_env=False)
            return self._default_trust_context

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion, using chat for explicitly enabled structured reasoning.

        Args:
            prompt: Input prompt string.
            **kwargs: Override temperature, max_tokens, or other options.

        Returns:
            Generated text (stripped of leading/trailing whitespace).

        Raises:
            httpx.HTTPStatusError: If Ollama returns an error status.
            httpx.ConnectError: If Ollama is not running.
            MnemosExternalServiceError: If Ollama does not return a final text response.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        think = _validate_ollama_think(kwargs.get("think", self.think))
        response_format = kwargs.get("format")
        # On reasoning models, generate can constrain the reasoning channel
        # itself and terminate without a final answer. Chat keeps the final
        # structured response separate. Preserve the existing route when
        # reasoning is disabled or left to the model's unspecified default.
        use_chat = response_format is not None and think is not None and think is not False

        payload = {
            "model": _request_model(self.model, kwargs),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if use_chat:
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt
        if response_format is not None:
            payload["format"] = response_format
        if think is not None:
            payload["think"] = think

        async def _request() -> str:
            async with httpx.AsyncClient(
                timeout=self.timeout, verify=self._tls_verification()
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/{'chat' if use_chat else 'generate'}",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise MnemosExternalServiceError("Ollama returned a non-object response")
                if self.response_observer is not None:
                    try:
                        self.response_observer(data)
                    except Exception:
                        log_event(
                            "mnemos.provider_observer_failure",
                            level=logging.WARNING,
                            provider="ollama",
                            operation="predict_observer",
                        )
                if use_chat:
                    message = data.get("message")
                    text = message.get("content") if isinstance(message, dict) else None
                else:
                    text = data.get("response")
                if not isinstance(text, str):
                    raise MnemosExternalServiceError("Ollama returned a non-string final response")
                final = text.strip()
                if not final:
                    raise MnemosExternalServiceError("Ollama returned an empty final response")
                return final

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
                error_type=type(exc).__name__,
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

    supports_model_override = True

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
        model = _request_model(self.model, kwargs)

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
                error_type=type(error).__name__,
            )
            raise error
        except Exception as exc:
            log_event(
                "mnemos.provider_failure",
                level=logging.ERROR,
                provider=provider_name,
                operation="predict",
                error_type=type(exc).__name__,
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
