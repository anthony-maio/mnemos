"""
mnemos/utils/__init__.py — Public API for the utils package.
"""

from .embeddings import (
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SimpleEmbeddingProvider,
    cosine_distance,
    cosine_similarity,
)
from .llm import LLMProvider, MockLLMProvider, OllamaProvider, OpenAIProvider
from .reliability import (
    MnemosConfigurationError,
    MnemosError,
    MnemosExternalServiceError,
    MnemosTransientError,
    RetryPolicy,
)
from .storage import InMemoryStore, MemoryStore, Neo4jStore, QdrantStore, SQLiteStore

__all__ = [
    # Embeddings
    "EmbeddingProvider",
    "SimpleEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "cosine_similarity",
    "cosine_distance",
    # LLM
    "LLMProvider",
    "MockLLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    # Reliability
    "MnemosError",
    "MnemosConfigurationError",
    "MnemosTransientError",
    "MnemosExternalServiceError",
    "RetryPolicy",
    # Storage
    "MemoryStore",
    "InMemoryStore",
    "SQLiteStore",
    "Neo4jStore",
    "QdrantStore",
]
