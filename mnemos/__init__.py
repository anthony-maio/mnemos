"""
mnemos — Biomimetic memory architectures for LLMs.

Mnemos implements five neuroscience-inspired memory modules:

1. SurprisalGate   — Predictive coding memory gate (only encodes surprising inputs)
2. MutableRAG      — Memory reconsolidation (facts evolve on retrieval, no stale data)
3. AffectiveRouter — Emotional state-dependent memory routing (amygdala filter)
4. SleepDaemon     — Hippocampal-neocortical consolidation (semantic compression)
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

from .config import (
    AffectiveConfig,
    MnemosConfig,
    MutableRAGConfig,
    SleepConfig,
    SpreadingConfig,
    SurprisalConfig,
)
from .engine import MnemosEngine
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
    MemoryStore,
    MockLLMProvider,
    OllamaProvider,
    OpenAIProvider,
    SimpleEmbeddingProvider,
    SQLiteStore,
    cosine_distance,
    cosine_similarity,
)

__version__ = "0.1.0"
__author__ = "Anthony Maio"
__license__ = "MIT"

__all__ = [
    # Version
    "__version__",
    # Engine
    "MnemosEngine",
    # Config
    "MnemosConfig",
    "SurprisalConfig",
    "MutableRAGConfig",
    "AffectiveConfig",
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
    "cosine_similarity",
    "cosine_distance",
    "LLMProvider",
    "MockLLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "MemoryStore",
    "InMemoryStore",
    "SQLiteStore",
]
