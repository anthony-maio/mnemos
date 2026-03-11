# Operations

Use this file when the user asks how Mnemos actually works in daily use, wants automation, or needs troubleshooting guidance.

## Automatic memory reality check

- Claude Code has the shipped automatic path today.
- The bundled hooks ingest high-signal user prompts and tool failures with `mnemos-cli autostore-hook`.
- The same hooks run `mnemos-cli consolidate` on `PreCompact` and `Stop`.
- Codex, Cursor, and generic MCP hosts do not get the same deterministic hook layer by default. They usually rely on:
  - the agent calling `mnemos_retrieve`, `mnemos_store`, and `mnemos_consolidate`
  - repo/user policy text such as `AGENTS.md`
  - host-specific automation added by the user

Do not hide this gap. If the user wants parity outside Claude Code, say so plainly.

## Storage guidance

- SQLite: default starter profile, simplest persistent setup.
- Qdrant: use when dataset size or latency justifies a vector database.
- Neo4jStore: experimental persistence backend for Neo4j environments. It does not yet make spreading activation graph-native.

## Retrieval quality guidance

- Use a real embedding provider for production quality.
- Treat `simple` embeddings as development-only.
- If retrieval quality looks poor, check:
  - embedding provider choice
  - scope and `scope_id`
  - whether the right store is configured
  - whether the expected memories were ever ingested

## Inspection and troubleshooting

Start with:

```bash
mnemos-cli doctor
mnemos-cli stats
mnemos inspect <chunk-id>
```

Useful MCP tools:

- `mnemos_health`
- `mnemos_stats`
- `mnemos_inspect`
- `mnemos_list`

If setup is host-related, prefer the control plane:

```bash
mnemos ui
```
