# Mnemos Architecture

## Overview
Mnemos is a biomimetic memory architecture for LLMs. It implements five neuroscience-inspired memory modules that can be used independently or composed together.

## Package Structure
```
mnemos/
├── mnemos/
│   ├── __init__.py          # Public API exports
│   ├── config.py            # Configuration dataclasses
│   ├── types.py             # Shared types (MemoryChunk, CognitiveState, etc.)
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── surprisal.py     # Module 1: Surprisal-Triggered Encoding
│   │   ├── mutable_rag.py   # Module 2: Mutable RAG (Memory Reconsolidation)
│   │   ├── affective.py     # Module 3: State-Dependent Affective Routing
│   │   ├── sleep.py         # Module 4: Sleep Daemon (Memory Consolidation)
│   │   └── spreading.py     # Module 5: Spreading Activation (Graph RAG)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── embeddings.py    # Embedding generation utilities
│   │   ├── llm.py           # LLM interface (Ollama, OpenAI, etc.)
│   │   └── storage.py       # Storage backends (in-memory, SQLite, etc.)
│   └── engine.py            # MnemosEngine - orchestrates all modules
├── tests/
├── examples/
│   ├── basic_usage.py
│   ├── surprisal_demo.py
│   └── full_pipeline.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## Core Types

### MemoryChunk
```python
@dataclass
class MemoryChunk:
    id: str
    content: str
    embedding: list[float] | None
    metadata: dict
    salience: float          # 0.0 - 1.0, from surprisal gate
    cognitive_state: CognitiveState | None
    created_at: datetime
    updated_at: datetime
    access_count: int
    version: int             # Tracks reconsolidation mutations
```

### CognitiveState
```python
@dataclass
class CognitiveState:
    valence: float     # -1.0 (negative) to 1.0 (positive)
    arousal: float     # 0.0 (calm) to 1.0 (urgent)
    complexity: float  # 0.0 (simple) to 1.0 (complex)
```

### ActivationNode
```python
@dataclass
class ActivationNode:
    id: str
    content: str
    energy: float       # Current activation energy
    edges: list[Edge]   # Connected nodes with weights
```

## Module APIs

Each module implements a common interface:

```python
class MemoryModule(ABC):
    async def process(self, interaction: Interaction) -> ProcessResult
    async def retrieve(self, query: str, **kwargs) -> list[MemoryChunk]
    def get_stats(self) -> dict
```

### Module 1: SurprisalGate
- Uses a fast local LLM to predict user intent
- Calculates semantic divergence between prediction and actual input
- Returns salience score; only stores high-surprisal interactions

### Module 2: MutableRAG
- On retrieval, flags chunks as "labile"
- Spawns async background task to evaluate if facts changed
- Overwrites stale chunks with synthesized updates

### Module 3: AffectiveRouter
- Classifies interactions on 3 axes (valence, arousal, complexity)
- Appends CognitiveState as metadata
- Modified retrieval: similarity * 0.7 + state_match * 0.3

### Module 4: SleepDaemon
- Episodic store (fast, in-memory) for current session
- Consolidation process extracts facts, preferences, patterns
- Prunes raw episodes after consolidation
- Optional: generates tool scripts from repeated patterns

### Module 5: SpreadingActivation
- Graph-based memory with activation energy
- On retrieval, injects energy at matched node
- Energy propagates along edges with 20% decay per hop
- Returns all nodes above activation threshold

## MnemosEngine
Orchestrates all modules in a pipeline:
1. Input → SurprisalGate (filter)
2. If stored → AffectiveRouter (tag)
3. On retrieval → SpreadingActivation + AffectiveRouter scoring
4. Post-retrieval → MutableRAG (reconsolidate)
5. Idle → SleepDaemon (consolidate)

## Storage Backends
- InMemoryStore (default, for development)
- SQLiteStore (persistent, lightweight)
- Interface for Neo4j, Qdrant (optional advanced)
