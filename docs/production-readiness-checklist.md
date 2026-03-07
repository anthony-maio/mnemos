# Production Readiness Checklist

Use this checklist before promoting a release as production-ready.

## Core Quality Gates

- [ ] `pytest -q` passes on supported Python versions
- [ ] `mypy .` passes with strict settings
- [ ] `black --check .` passes
- [ ] Tier 1 MCP smoke tests pass (Claude Code, Claude Desktop config, generic stdio contract)
- [ ] Benchmark production replacement gate passes
- [ ] Profile-specific benchmark gates pass (`memory`, `sqlite`, `qdrant`)
- [ ] CLI startup and store/embedder wiring smoke tests pass

## Retrieval and Memory Quality

- [ ] Non-toy embedding provider configured for production (`ollama` or `openai`)
- [ ] Retrieval quality baseline documented (Recall@k/MRR) from `mnemos-benchmark`
- [ ] Contradiction/update, preference-drift, and cross-project-scope benchmark packs evaluated
- [ ] Startup graph hydration validated against persisted stores

## Performance and Scale

- [ ] p95 retrieval latency budget defined and measured
- [ ] Storage growth behavior measured for representative workloads
- [ ] Store backend chosen for scale target (`sqlite` for small, `qdrant` for large)
- [ ] `mnemos-cli doctor` and MCP `mnemos_health` report ready/degraded as expected for selected profile

## Operations and Safety

- [ ] Environment variable docs match implementation
- [ ] Backward-compatible env aliases validated (`MNEMOS_STORAGE`, `MNEMOS_DB_PATH`)
- [ ] Shared memory write firewall configured (`MNEMOS_MEMORY_*` secret/PII policy)
- [ ] Governance controls configured (`capture_mode`, TTL retention, per-scope cap)
- [ ] Memory audit flows validated (`mnemos-cli list/search/export/purge`)
- [ ] Default production docs do not rely on mock providers
- [ ] Structured JSON logs validated for startup, retrieval latency, and provider failures
- [ ] Tier 2 smoke checks (Cursor/Windsurf/Cline) reviewed with known caveats tracked
- [ ] Release notes list known limitations and migration notes
