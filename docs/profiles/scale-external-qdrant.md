# Profile: Scale (External Qdrant)

Use this for shared/team memory and networked deployments.

## When to Use

- Shared memory service across users/agents
- Persistent vector storage outside local workstation
- Operational ownership of a remote Qdrant service

## Environment

```bash
MNEMOS_STORE_TYPE=qdrant
MNEMOS_QDRANT_URL=https://qdrant.example.com
MNEMOS_QDRANT_API_KEY=...
MNEMOS_QDRANT_COLLECTION=mnemos_memory
MNEMOS_LLM_PROVIDER=openclaw
MNEMOS_EMBEDDING_PROVIDER=openclaw
```

## Install

```bash
pip install "mnemos-memory[mcp,qdrant]"
```

## Notes

- Use `mnemos-cli doctor` and `mnemos_health` to verify dependency and profile readiness.
- Recommended for team workloads and long-running MCP servers.
