# Production Readiness Checklist

Use this checklist before promoting a release as production-ready.

## Core Quality Gates

- [ ] `pytest -q` passes on supported Python versions
- [ ] `mypy mnemos` passes with strict settings
- [ ] `black --check .` passes
- [ ] MCP startup and tool wiring smoke tests pass
- [ ] CLI startup and store/embedder wiring smoke tests pass

## Retrieval and Memory Quality

- [ ] Non-toy embedding provider configured for production (`ollama` or `openai`)
- [ ] Retrieval quality baseline documented (Recall@k/MRR) from `mnemos-benchmark`
- [ ] Reconsolidation behavior validated with contradiction/update scenarios
- [ ] Startup graph hydration validated against persisted stores

## Performance and Scale

- [ ] p95 retrieval latency budget defined and measured
- [ ] Storage growth behavior measured for representative workloads
- [ ] Store backend chosen for scale target (`sqlite` for small, `qdrant` for large)

## Operations and Safety

- [ ] Environment variable docs match implementation
- [ ] Backward-compatible env aliases validated (`MNEMOS_STORAGE`, `MNEMOS_DB_PATH`)
- [ ] Default production docs do not rely on mock providers
- [ ] Release notes list known limitations and migration notes
