# Profile: Local Performance (Embedded Qdrant)

Use this for local production-grade retrieval with no external database service.

## When to Use

- Single-user or workstation-local memory
- Larger memory sets than SQLite looped cosine retrieval
- You want vector DB performance without managing remote infra

## Environment

```bash
MNEMOS_STORE_TYPE=qdrant
MNEMOS_QDRANT_PATH=.mnemos/qdrant
MNEMOS_QDRANT_COLLECTION=mnemos_memory
MNEMOS_LLM_PROVIDER=openclaw
MNEMOS_EMBEDDING_PROVIDER=openclaw
```

## Install

```bash
pip install "mnemos-memory[mcp,qdrant]"
```

## Notes

- This is the recommended upgrade path from starter SQLite.
- Plugin bootstrap automatically installs `qdrant` extra when `MNEMOS_STORE_TYPE=qdrant`.
