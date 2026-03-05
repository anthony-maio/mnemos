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

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np


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
        # IDF tracking: word → document frequency count
        self._doc_freq: Counter[str] = Counter()
        self._doc_count: int = 0
        # Projection matrix for hash trick (words → dim-space via random projection)
        self._projection: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self._dim

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
        tokens = _tokenize(text)
        self._update_idf(tokens)
        vec = self._compute_tfidf_vector(tokens)
        return vec.tolist()

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
        # First pass: update IDF for the whole batch
        tokenized = [_tokenize(t) for t in texts]
        for tokens in tokenized:
            self._update_idf(tokens)

        # Second pass: compute embeddings with updated IDF
        return [self._compute_tfidf_vector(tokens).tolist() for tokens in tokenized]


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
