# Mnemos Architecture

## Overview
Mnemos is an MCP-native memory layer with a biomimetic retrieval pipeline.
The runtime is organized around one orchestrator (`MnemosEngine`), five
memory modules, and pluggable providers for embeddings, LLMs, and storage.

## Runtime Surfaces
- `mnemos.mcp_server`: stdio MCP server (tools + resources for hosts like Claude Code/Desktop).
- `mnemos.cli`: operational CLI (`store`, `retrieve`, `consolidate`, `doctor`, `profile`, audit flows).
- `mnemos.engine.MnemosEngine`: programmatic API that composes the full pipeline.

## Package Structure
```text
mnemos/
├── mnemos/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── cli.py
│   ├── config.py
│   ├── engine.py
│   ├── health.py
│   ├── hook_autostore.py
│   ├── mcp_server.py
│   ├── memory_safety.py
│   ├── observability.py
│   ├── runtime.py
│   ├── types.py
│   ├── modules/
│   │   ├── surprisal.py
│   │   ├── mutable_rag.py
│   │   ├── affective.py
│   │   ├── sleep.py
│   │   └── spreading.py
│   └── utils/
│       ├── embeddings.py
│       ├── llm.py
│       ├── reliability.py
│       └── storage.py
├── benchmarks/
├── docs/
├── examples/
│   ├── basic_usage.py
│   ├── full_pipeline.py
│   └── mcp_agent_demo.py
└── tests/
```

## Data Model
Domain types are Pydantic models (not dataclasses):
- `MemoryChunk`: persisted memory unit with content, embedding, metadata, salience,
  optional `CognitiveState`, timestamps, access count, and reconsolidation version.
- `CognitiveState`: valence/arousal/complexity triplet for affective routing.
- `ActivationNode`: graph node used by spreading activation, with `neighbors` map
  (`node_id -> edge weight`) and transient activation `energy`.
- `Interaction`, `ProcessResult`, `ConsolidationResult`: ingest and pipeline results.

## Pipeline
1. Encode (`process`): `SurprisalGate -> AffectiveRouter -> store + spreading graph`.
2. Retrieve (`retrieve`): `SpreadingActivation + AffectiveRouter -> ranking -> MutableRAG queue`.
3. Maintain (`consolidate`): `SleepDaemon` extracts durable facts and prunes episodes.

## Storage and Persistence
- `InMemoryStore` (implemented): zero-setup, ephemeral development backend.
- `SQLiteStore` (implemented): local persistent backend with no external services.
- `QdrantStore` (implemented): vector DB backend for higher-scale retrieval.
- `Neo4jStore` (experimental): Neo4j-backed persistence for MemoryChunks. The spreading-activation graph still hydrates in-process from stored chunks today, so graph-native activation persistence remains future work.
