"""
mnemos — Biomimetic memory architectures for LLMs.

Mnemos implements five neuroscience-inspired memory modules:

1. SurprisalGate   — Predictive coding memory gate (only encodes surprising inputs)
2. MutableRAG      — Memory reconsolidation (facts evolve on retrieval, no stale data)
3. AffectiveRouter — Emotional state-dependent memory routing (amygdala filter)
4. SleepDaemon     — Hippocampal-neocortical consolidation with optional recall-gated plasticity
5. SpreadingActivation — Graph-based associative context retrieval

Quick start (zero external dependencies):

    import asyncio
    from mnemos import MnemosEngine
    from mnemos.types import Interaction

    async def main():
        engine = MnemosEngine()  # Uses MockLLM + SimpleEmbedding + InMemoryStore
        await engine.process(Interaction(role="user", content="I use Python for ML."))
        memories = await engine.retrieve("programming languages")
        for m in memories:
            print(m.content)

    asyncio.run(main())
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import (
        AffectiveConfig,
        MemoryGovernanceConfig,
        MemorySafetyConfig,
        MnemosConfig,
        MutableRAGConfig,
        SleepConfig,
        SpreadingConfig,
        SurprisalConfig,
    )
    from .engine import MnemosEngine
    from .memory_safety import MemorySafetyDecision, MemorySafetyMatch, MemoryWriteFirewall
    from .modules import (
        AffectiveRouter,
        MutableRAG,
        SleepDaemon,
        SpreadingActivation,
        SurprisalGate,
    )
    from .types import (
        ActivationNode,
        CognitiveState,
        ConsolidationResult,
        Interaction,
        MemoryChunk,
        ProcessResult,
    )
    from .utils import (
        EmbeddingProvider,
        InMemoryStore,
        LLMProvider,
        MnemosConfigurationError,
        MnemosError,
        MnemosExternalServiceError,
        MnemosTransientError,
        MemoryStore,
        MockLLMProvider,
        UnavailableLLMProvider,
        OllamaEmbeddingProvider,
        OllamaProvider,
        OpenAIEmbeddingProvider,
        OpenAIProvider,
        RetryPolicy,
        SimpleEmbeddingProvider,
        SQLiteStore,
        cosine_distance,
        cosine_similarity,
    )

# Resolve the public API only when requested; capture clients do not run inference.
_LAZY_EXPORTS = {
    "AffectiveConfig": (".config", "AffectiveConfig"),
    "MemoryGovernanceConfig": (".config", "MemoryGovernanceConfig"),
    "MemorySafetyConfig": (".config", "MemorySafetyConfig"),
    "MnemosConfig": (".config", "MnemosConfig"),
    "MutableRAGConfig": (".config", "MutableRAGConfig"),
    "SleepConfig": (".config", "SleepConfig"),
    "SpreadingConfig": (".config", "SpreadingConfig"),
    "SurprisalConfig": (".config", "SurprisalConfig"),
    "MnemosEngine": (".engine", "MnemosEngine"),
    "MemorySafetyDecision": (".memory_safety", "MemorySafetyDecision"),
    "MemorySafetyMatch": (".memory_safety", "MemorySafetyMatch"),
    "MemoryWriteFirewall": (".memory_safety", "MemoryWriteFirewall"),
    "AffectiveRouter": (".modules", "AffectiveRouter"),
    "MutableRAG": (".modules", "MutableRAG"),
    "SleepDaemon": (".modules", "SleepDaemon"),
    "SpreadingActivation": (".modules", "SpreadingActivation"),
    "SurprisalGate": (".modules", "SurprisalGate"),
    "ActivationNode": (".types", "ActivationNode"),
    "CognitiveState": (".types", "CognitiveState"),
    "ConsolidationResult": (".types", "ConsolidationResult"),
    "Interaction": (".types", "Interaction"),
    "MemoryChunk": (".types", "MemoryChunk"),
    "ProcessResult": (".types", "ProcessResult"),
    "EmbeddingProvider": (".utils", "EmbeddingProvider"),
    "InMemoryStore": (".utils", "InMemoryStore"),
    "LLMProvider": (".utils", "LLMProvider"),
    "MnemosConfigurationError": (".utils", "MnemosConfigurationError"),
    "MnemosError": (".utils", "MnemosError"),
    "MnemosExternalServiceError": (".utils", "MnemosExternalServiceError"),
    "MnemosTransientError": (".utils", "MnemosTransientError"),
    "MemoryStore": (".utils", "MemoryStore"),
    "MockLLMProvider": (".utils", "MockLLMProvider"),
    "UnavailableLLMProvider": (".utils", "UnavailableLLMProvider"),
    "OllamaEmbeddingProvider": (".utils", "OllamaEmbeddingProvider"),
    "OllamaProvider": (".utils", "OllamaProvider"),
    "OpenAIEmbeddingProvider": (".utils", "OpenAIEmbeddingProvider"),
    "OpenAIProvider": (".utils", "OpenAIProvider"),
    "RetryPolicy": (".utils", "RetryPolicy"),
    "SimpleEmbeddingProvider": (".utils", "SimpleEmbeddingProvider"),
    "SQLiteStore": (".utils", "SQLiteStore"),
    "cosine_distance": (".utils", "cosine_distance"),
    "cosine_similarity": (".utils", "cosine_similarity"),
}


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(_import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__version__ = "0.7.0b1"
__author__ = "Anthony Maio"
__license__ = "MIT"

__all__ = [
    # Version
    "__version__",
    # Engine
    "MnemosEngine",
    "MemoryWriteFirewall",
    "MemorySafetyDecision",
    "MemorySafetyMatch",
    # Config
    "MnemosConfig",
    "SurprisalConfig",
    "MutableRAGConfig",
    "AffectiveConfig",
    "MemoryGovernanceConfig",
    "MemorySafetyConfig",
    "SleepConfig",
    "SpreadingConfig",
    # Modules
    "SurprisalGate",
    "MutableRAG",
    "AffectiveRouter",
    "SleepDaemon",
    "SpreadingActivation",
    # Types
    "MemoryChunk",
    "CognitiveState",
    "Interaction",
    "ProcessResult",
    "ActivationNode",
    "ConsolidationResult",
    # Utils
    "EmbeddingProvider",
    "SimpleEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "cosine_similarity",
    "cosine_distance",
    "LLMProvider",
    "MockLLMProvider",
    "UnavailableLLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "MnemosError",
    "MnemosConfigurationError",
    "MnemosTransientError",
    "MnemosExternalServiceError",
    "RetryPolicy",
    "MemoryStore",
    "InMemoryStore",
    "SQLiteStore",
]
