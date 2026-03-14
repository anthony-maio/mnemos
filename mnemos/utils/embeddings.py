"""
mnemos/utils/embeddings.py — Embedding providers for the Mnemos memory system.

Embeddings are the foundational representation used for:
- Surprisal calculation (semantic distance between prediction and input)
- Memory retrieval (cosine similarity search)
- Spreading activation graph construction (connecting related nodes)
- Affective routing (finding semantically similar chunks to score)

The SimpleEmbeddingProvider uses TF-IDF-inspired term weighting computed
purely with NumPy — no external model downloads or API calls needed.
This makes it ideal for development, testing, and offline scenarios.

For production, swap in a proper sentence-transformer or embedding API provider.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import threading
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

import httpx
import numpy as np

from ..observability import log_event
from .reliability import (
    MnemosConfigurationError,
    RetryPolicy,
    call_with_retry,
    is_retryable_http_exception,
)


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    All providers must implement embed() which converts text to a fixed-size
    float vector. Providers backed by deterministic models (e.g., sentence
    transformers) should produce stable embeddings for the same input.
    Providers with mutable internal state (e.g., adaptive IDF) may return
    different vectors for the same text as more documents are observed.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string into a fixed-size float vector.

        Note: Depending on the provider implementation, the returned vector
        may vary across calls for the same input text (e.g., if the provider
        maintains mutable state such as IDF statistics). See provider-specific
        documentation for determinism guarantees.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the text in embedding space.
            The vector is L2-normalized (unit length) for cosine similarity.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts efficiently.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of embeddings produced by this provider."""
        ...

    @property
    def should_offload_in_async(self) -> bool:
        """
        Whether async helpers should offload this provider to a worker thread.

        Network-backed or blocking providers should keep the default True.
        Cheap in-process providers can override to run inline and avoid
        thread-hop overhead on hot paths.
        """
        return True


def _tokenize(text: str) -> list[str]:
    """
    Simple word tokenizer: lowercase, remove punctuation, split on whitespace.

    Args:
        text: Input string.

    Returns:
        List of lowercase tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [tok for tok in text.split() if len(tok) > 1]


def _provider_name_from_base_url(base_url: str) -> str:
    lower = base_url.lower()
    if "openrouter" in lower:
        return "openrouter"
    if "openclaw" in lower:
        return "openclaw"
    return "openai"


class SimpleEmbeddingProvider(EmbeddingProvider):
    """
    TF-IDF-inspired embedding provider using only NumPy.

    Algorithm:
    1. Build a vocabulary from all texts seen so far (or from a seed corpus).
    2. For each text, compute TF (term frequency) × IDF (inverse doc frequency).
    3. Project to a fixed-size vector via hashing (hash embedding trick).
    4. L2-normalize the result.

    The hash trick maps arbitrary vocabulary to a fixed dimension without
    requiring a pre-built vocab, making it work on any text immediately.

    This is not as accurate as a proper sentence transformer but produces
    reasonable similarity scores for development and testing purposes.

    IMPORTANT: Embeddings produced by this provider are NOT stable across calls.
    Because IDF weights are updated with every call to embed() or
    embed_batch(), the same text will yield a different vector after new
    documents have been observed. This makes SimpleEmbeddingProvider suitable
    only for development and testing -- not for production workloads where
    embedding consistency is required (e.g., persisted vector indices). For
    production, use a sentence-transformer or external embedding API provider.

    Args:
        dim: Embedding dimensionality (default 384 to match config default).
        seed: Random seed for deterministic hash projections.
    """

    def __init__(self, dim: int = 384, seed: int = 42) -> None:
        self._dim = dim
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        # IDF tracking: word → document frequency count
        self._doc_freq: Counter[str] = Counter()
        self._doc_count: int = 0
        # Projection matrix for hash trick (words → dim-space via random projection)
        self._projection: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def should_offload_in_async(self) -> bool:
        """Simple local embeddings are cheap enough to run inline."""
        return False

    def _get_word_vector(self, word: str) -> np.ndarray:
        """
        Get a deterministic random projection vector for a word.

        Uses the hash trick: hash the word string to seed a local RNG,
        then draw a random unit vector. Same word always gets same vector.
        """
        if word not in self._projection:
            # Deterministic seed from stable hash + global seed
            word_seed = int(hashlib.md5(word.encode()).hexdigest()[:8], 16) ^ self._seed
            local_rng = np.random.default_rng(word_seed)
            vec = local_rng.standard_normal(self._dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._projection[word] = vec
        return self._projection[word]

    def _compute_tfidf_vector(self, tokens: list[str]) -> np.ndarray:
        """
        Compute a TF-IDF weighted embedding for a list of tokens.

        Each token's word vector is scaled by its TF-IDF weight, then summed.
        The result is L2-normalized.
        """
        if not tokens:
            return np.zeros(self._dim, dtype=np.float32)

        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        result = np.zeros(self._dim, dtype=np.float32)

        for word, count in token_counts.items():
            tf = count / total_tokens
            # IDF: log((N + 1) / (df + 1)) + 1  (smoothed)
            df = self._doc_freq.get(word, 0)
            idf = math.log((self._doc_count + 1) / (df + 1)) + 1.0
            weight = tf * idf
            result += weight * self._get_word_vector(word)

        # L2 normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm

        return result

    def _update_idf(self, tokens: list[str]) -> None:
        """Update document frequency counts for IDF calculation."""
        unique_words = set(tokens)
        for word in unique_words:
            self._doc_freq[word] += 1
        self._doc_count += 1

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Updates internal IDF statistics with this document, then computes
        the TF-IDF weighted random projection embedding.

        Args:
            text: Input text to embed.

        Returns:
            L2-normalized embedding as list of floats.
        """
        with self._lock:
            tokens = _tokenize(text)
            self._update_idf(tokens)
            vec = self._compute_tfidf_vector(tokens)
            return [float(x) for x in vec.tolist()]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts, updating IDF across the full batch first.

        First pass: update all IDF counts (so each text's IDF is computed
        with awareness of the full batch). Second pass: compute embeddings.

        Args:
            texts: List of input strings.

        Returns:
            List of L2-normalized embedding vectors.
        """
        with self._lock:
            # First pass: update IDF for the whole batch
            tokenized = [_tokenize(t) for t in texts]
            for tokens in tokenized:
                self._update_idf(tokens)

            # Second pass: compute embeddings with updated IDF
            return [
                [float(x) for x in self._compute_tfidf_vector(tokens).tolist()]
                for tokens in tokenized
            ]


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider backed by Ollama's embedding endpoints.

    Uses `/api/embed` when available and falls back to `/api/embeddings`
    for compatibility with older Ollama versions.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("Embedding dimension unknown until first embed() call.")
        return self._dim

    def _parse_vector(self, payload: dict[str, Any]) -> list[float]:
        if "embedding" in payload and isinstance(payload["embedding"], list):
            vector = payload["embedding"]
        elif "embeddings" in payload and isinstance(payload["embeddings"], list):
            embeddings = payload["embeddings"]
            if not embeddings:
                raise ValueError("Ollama returned empty embeddings list.")
            vector = embeddings[0]
        else:
            raise ValueError("Ollama response missing embedding vector.")

        vec = [float(x) for x in vector]
        self._dim = len(vec)
        return vec

    def embed(self, text: str) -> list[float]:
        # Prefer modern /api/embed endpoint.
        try:
            return call_with_retry(
                provider="ollama",
                operation="embed",
                policy=RetryPolicy(),
                should_retry=is_retryable_http_exception,
                fn=lambda: self._parse_vector(
                    _http_post_json(
                        f"{self.base_url}/api/embed",
                        payload={"model": self.model, "input": text},
                        timeout=self.timeout,
                    )
                ),
            )
        except Exception:
            # Fallback for older Ollama servers.
            try:
                return call_with_retry(
                    provider="ollama",
                    operation="embed_legacy",
                    policy=RetryPolicy(),
                    should_retry=is_retryable_http_exception,
                    fn=lambda: self._parse_vector(
                        _http_post_json(
                            f"{self.base_url}/api/embeddings",
                            payload={"model": self.model, "prompt": text},
                            timeout=self.timeout,
                        )
                    ),
                )
            except Exception as exc:
                log_event(
                    "mnemos.provider_failure",
                    level=logging.ERROR,
                    provider="ollama",
                    operation="embed",
                    error=str(exc),
                )
                raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            payload = call_with_retry(
                provider="ollama",
                operation="embed_batch",
                policy=RetryPolicy(),
                should_retry=is_retryable_http_exception,
                fn=lambda: _http_post_json(
                    f"{self.base_url}/api/embed",
                    payload={"model": self.model, "input": texts},
                    timeout=self.timeout,
                ),
            )
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                vectors = [[float(x) for x in row] for row in embeddings]
                self._dim = len(vectors[0])
                return vectors
        except Exception as exc:
            log_event(
                "mnemos.provider_failure",
                level=logging.ERROR,
                provider="ollama",
                operation="embed_batch",
                error=str(exc),
            )

        # Graceful fallback if batch endpoint is unavailable.
        return [self.embed(text) for text in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider for OpenAI-compatible `/embeddings` APIs.
    """

    def __init__(
        self,
        api_key: str,
        api_key_fallback: str | None = None,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.api_key_fallback = api_key_fallback
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("Embedding dimension unknown until first embed() call.")
        return self._dim

    def _headers(self, api_key: str | None = None) -> dict[str, str]:
        key = self.api_key if api_key is None else api_key
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _parse_single(self, payload: dict[str, Any]) -> list[float]:
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError("OpenAI embedding response missing data.")
        first = data[0]
        if not isinstance(first, dict) or "embedding" not in first:
            raise ValueError("OpenAI embedding response missing embedding vector.")
        vector = [float(x) for x in first["embedding"]]
        self._dim = len(vector)
        return vector

    def _embed_with_key(self, text: str, *, api_key: str, provider_name: str) -> list[float]:
        payload = call_with_retry(
            provider=provider_name,
            operation="embed",
            policy=RetryPolicy(),
            should_retry=is_retryable_http_exception,
            fn=lambda: _http_post_json(
                f"{self.base_url}/embeddings",
                payload={
                    "model": self.model,
                    "input": text,
                },
                timeout=self.timeout,
                headers=self._headers(api_key),
            ),
        )
        return self._parse_single(payload)

    def _embed_batch_with_key(
        self,
        texts: list[str],
        *,
        api_key: str,
        provider_name: str,
    ) -> list[list[float]]:
        payload = call_with_retry(
            provider=provider_name,
            operation="embed_batch",
            policy=RetryPolicy(),
            should_retry=is_retryable_http_exception,
            fn=lambda: _http_post_json(
                f"{self.base_url}/embeddings",
                payload={
                    "model": self.model,
                    "input": texts,
                },
                timeout=self.timeout,
                headers=self._headers(api_key),
            ),
        )
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("OpenAI embedding response missing data.")
        vectors = [[float(x) for x in item["embedding"]] for item in data]
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    def _fallback_key_for_auth_retry(self, provider_name: str) -> str | None:
        fallback_key = self.api_key_fallback
        if provider_name != "openrouter":
            return None
        if fallback_key in (None, "", self.api_key):
            return None
        return fallback_key

    def embed(self, text: str) -> list[float]:
        provider_name = _provider_name_from_base_url(self.base_url)
        try:
            return self._embed_with_key(text, api_key=self.api_key, provider_name=provider_name)
        except MnemosConfigurationError as exc:
            error: Exception = exc
            fallback_key = self._fallback_key_for_auth_retry(provider_name)
            if fallback_key is not None:
                try:
                    vector = self._embed_with_key(
                        text,
                        api_key=fallback_key,
                        provider_name=provider_name,
                    )
                    self.api_key = fallback_key
                    return vector
                except Exception as retry_exc:
                    error = retry_exc
            log_event(
                "mnemos.provider_failure",
                level=logging.ERROR,
                provider=provider_name,
                operation="embed",
                error=str(error),
            )
            raise error
        except Exception as exc:
            log_event(
                "mnemos.provider_failure",
                level=logging.ERROR,
                provider=provider_name,
                operation="embed",
                error=str(exc),
            )
            raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        provider_name = _provider_name_from_base_url(self.base_url)
        try:
            return self._embed_batch_with_key(
                texts,
                api_key=self.api_key,
                provider_name=provider_name,
            )
        except MnemosConfigurationError as exc:
            error: Exception = exc
            fallback_key = self._fallback_key_for_auth_retry(provider_name)
            if fallback_key is not None:
                try:
                    vectors = self._embed_batch_with_key(
                        texts,
                        api_key=fallback_key,
                        provider_name=provider_name,
                    )
                    self.api_key = fallback_key
                    return vectors
                except Exception as retry_exc:
                    error = retry_exc
            log_event(
                "mnemos.provider_failure",
                level=logging.ERROR,
                provider=provider_name,
                operation="embed_batch",
                error=str(error),
            )
            raise error
        except Exception as exc:
            log_event(
                "mnemos.provider_failure",
                level=logging.ERROR,
                provider=provider_name,
                operation="embed_batch",
                error=str(exc),
            )
            raise


def _http_post_json(
    url: str,
    *,
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    if headers is None:
        response = httpx.post(
            url,
            json=payload,
            timeout=timeout,
        )
    else:
        response = httpx.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, dict):
        raise ValueError("Embedding API response must be a JSON object.")
    return raw


async def embed_text_async(embedder: EmbeddingProvider, text: str) -> list[float]:
    """Offload synchronous embedding providers from the async event loop."""
    if not embedder.should_offload_in_async:
        return embedder.embed(text)
    return await asyncio.to_thread(embedder.embed, text)


async def embed_batch_async(embedder: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    """Offload synchronous batch embedding work from the async event loop."""
    if not embedder.should_offload_in_async:
        return embedder.embed_batch(texts)
    return await asyncio.to_thread(embedder.embed_batch, texts)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Cosine similarity = dot(a, b) / (|a| * |b|)
    Returns a value in [-1, 1] where 1 = identical direction, 0 = orthogonal,
    -1 = opposite direction.

    For L2-normalized vectors (as produced by SimpleEmbeddingProvider),
    this reduces to just dot(a, b).

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity in [-1, 1].

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: len(a)={len(a)}, len(b)={len(b)}")
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def cosine_distance(a: list[float], b: list[float]) -> float:
    """
    Compute cosine distance between two embedding vectors.

    Cosine distance = 1 - cosine_similarity(a, b)
    Returns a value in [0, 2] where 0 = identical, 1 = orthogonal, 2 = opposite.
    In practice for document embeddings, values cluster in [0, 1].

    This is the metric used by SurprisalGate to measure semantic divergence
    between the predicted next input and the actual input.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine distance in [0, 2].
    """
    return 1.0 - cosine_similarity(a, b)
