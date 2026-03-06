# Mnemos Production Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship a production-grade memory MCP by replacing toy retrieval primitives, closing persistence gaps, and adding verification/observability required for real workloads.

**Architecture:** Keep the existing five-module cognitive pipeline, but harden the I/O layer: real embedding providers, scalable vector storage, persistent graph behavior, and deterministic startup/ops behavior. Treat current `SimpleEmbeddingProvider + O(N) retrieval` as a fallback dev profile only.

**Tech Stack:** Python 3.10+, Pydantic v2, NumPy, httpx, SQLite, optional Qdrant, optional sentence-transformers, optional OpenAI/Ollama embedding APIs, pytest, mypy, GitHub Actions.

---

### Task 1: Lock In Production Acceptance Criteria

**Files:**
- Create: `docs/production-readiness-checklist.md`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_engine.py`

**Step 1: Write the failing acceptance test**

Add one explicit production profile test in `tests/test_engine.py` that asserts startup with a non-mock embedder config is possible and rejects unknown provider names.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py::TestEngineProductionProfile -v`
Expected: FAIL with missing provider plumbing.

**Step 3: Add minimal production checklist + CI gate**

Add checklist sections:
- Retrieval quality baseline exists
- p95 latency budget exists
- mypy passes
- MCP and CLI smoke tests pass
- no mock defaults in production examples

Update CI to run a targeted production profile job in addition to existing unit tests.

**Step 4: Run tests and lint to verify pass**

Run: `pytest -q && mypy .`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/production-readiness-checklist.md README.md .github/workflows/ci.yml tests/test_engine.py
git commit -m "chore: define production acceptance criteria and CI gate"
```

### Task 2: Add Real Embedding Providers (Highest Impact)

**Files:**
- Modify: `mnemos/utils/embeddings.py`
- Modify: `mnemos/utils/__init__.py`
- Modify: `mnemos/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_embeddings.py`

**Step 1: Write failing provider tests**

Add tests for:
- `OllamaEmbeddingProvider.embed()` returns fixed dimension and is deterministic for same input
- `OpenAIEmbeddingProvider.embed()` parses API response correctly
- `SimpleEmbeddingProvider` is marked dev-only in docs/tests

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL because providers do not exist yet.

**Step 3: Implement minimal provider code**

Add:
- `OllamaEmbeddingProvider` (HTTP API call)
- `OpenAIEmbeddingProvider` (OpenAI-compatible embeddings endpoint)
- optional batch embedding methods for both providers

Keep `SimpleEmbeddingProvider` but clearly designated as fallback/dev.

**Step 4: Re-run embedding and full tests**

Run: `pytest tests/test_embeddings.py -v && pytest -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add mnemos/utils/embeddings.py mnemos/utils/__init__.py mnemos/__init__.py pyproject.toml tests/test_embeddings.py
git commit -m "feat: add production embedding providers for ollama and openai-compatible apis"
```

### Task 3: Wire Provider Selection in MCP + CLI (with Backward-Compatible Env Aliases)

**Files:**
- Modify: `mnemos/mcp_server.py`
- Modify: `mnemos/cli.py`
- Modify: `mnemos/config.py`
- Modify: `docs/MCP_INTEGRATION.md`
- Test: `tests/test_cli.py`
- Test: `tests/test_mcp_server.py`

**Step 1: Write failing MCP/CLI config tests**

Add tests for:
- `MNEMOS_EMBEDDING_PROVIDER` selection (`simple`, `ollama`, `openai`)
- backward-compatible aliases: `MNEMOS_STORAGE` -> `MNEMOS_STORE_TYPE`, `MNEMOS_DB_PATH` -> `MNEMOS_SQLITE_PATH`
- invalid provider values fail fast with clear errors

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py tests/test_mcp_server.py -v`
Expected: FAIL due missing env parsing behavior.

**Step 3: Implement minimal config wiring**

Add environment resolution helper used by both CLI and MCP:
- Parse canonical vars first
- fall back to compatibility aliases
- return normalized config object

**Step 4: Run tests**

Run: `pytest tests/test_cli.py tests/test_mcp_server.py -v && pytest -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add mnemos/mcp_server.py mnemos/cli.py mnemos/config.py docs/MCP_INTEGRATION.md tests/test_cli.py tests/test_mcp_server.py
git commit -m "feat: add embedding provider selection and env compatibility aliases"
```

### Task 4: Rehydrate Graph State from Persistent Store at Startup

**Files:**
- Modify: `mnemos/engine.py`
- Modify: `mnemos/modules/spreading.py`
- Test: `tests/test_engine.py`

**Step 1: Write failing restart-persistence test**

Add integration test:
- store chunks in SQLite
- create fresh `MnemosEngine` with same DB
- assert spreading graph node count reflects persisted chunks after startup

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py::TestEnginePersistence -v`
Expected: FAIL because graph currently starts empty.

**Step 3: Implement minimal hydration**

On engine init:
- load chunks via `store.get_all()`
- add missing nodes to `SpreadingActivation`
- optionally connect neighbors using existing threshold logic

Include guardrails for large stores (configurable max startup hydration count).

**Step 4: Re-run tests**

Run: `pytest tests/test_engine.py::TestEnginePersistence -v && pytest -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add mnemos/engine.py mnemos/modules/spreading.py tests/test_engine.py
git commit -m "feat: hydrate spreading activation graph from persistent store on startup"
```

### Task 5: Implement Vector DB Backend (Qdrant First)

**Files:**
- Modify: `mnemos/utils/storage.py`
- Modify: `mnemos/mcp_server.py`
- Modify: `mnemos/cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Test: `tests/test_storage_qdrant.py`

**Step 1: Write failing storage contract tests**

Add backend-agnostic tests asserting `MemoryStore` contract:
- `store/get/update/delete`
- `retrieve(top_k)` ranking semantics
- `get_stats` fields present

Parameterize tests over `InMemoryStore`, `SQLiteStore`, and new `QdrantStore`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage_qdrant.py -v`
Expected: FAIL because `QdrantStore` is not implemented.

**Step 3: Implement minimal Qdrant backend**

Add `QdrantStore(MemoryStore)`:
- collection bootstrap
- payload schema for `MemoryChunk`
- vector search using Qdrant nearest-neighbor query

Expose via env: `MNEMOS_STORE_TYPE=qdrant` + URL/collection vars.

**Step 4: Run tests**

Run: `pytest tests/test_storage_qdrant.py -v && pytest -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add mnemos/utils/storage.py mnemos/mcp_server.py mnemos/cli.py pyproject.toml README.md tests/test_storage_qdrant.py
git commit -m "feat: add qdrant vector store backend for scalable retrieval"
```

### Task 6: Make Reconsolidation Cooldown Real (Dead Config Removal or Implementation)

**Files:**
- Modify: `mnemos/modules/mutable_rag.py`
- Modify: `mnemos/config.py`
- Test: `tests/test_modules.py`

**Step 1: Write failing cooldown test**

Add test that retrieves same chunk repeatedly and verifies reconsolidation is skipped during cooldown window.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_modules.py::TestMutableRAG::test_reconsolidation_cooldown -v`
Expected: FAIL because cooldown is currently unused.

**Step 3: Implement minimal cooldown tracking**

Track last reconsolidation timestamp per chunk and enforce `reconsolidation_cooldown_seconds` before requeueing.

**Step 4: Run module tests**

Run: `pytest tests/test_modules.py::TestMutableRAG -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add mnemos/modules/mutable_rag.py mnemos/config.py tests/test_modules.py
git commit -m "fix: enforce mutable rag reconsolidation cooldown"
```

### Task 7: Benchmark Harness (Quality + Latency + Cost)

**Files:**
- Create: `benchmarks/benchmark_retrieval.py`
- Create: `benchmarks/datasets/sample_memory_eval.jsonl`
- Create: `docs/benchmarks.md`
- Modify: `README.md`

**Step 1: Write failing benchmark smoke test**

Add a small test that runs benchmark script against sample dataset and asserts JSON metrics output schema.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmarks.py -v`
Expected: FAIL because harness does not exist.

**Step 3: Implement benchmark harness**

Report at minimum:
- Recall@k / MRR
- p50 and p95 retrieval latency
- storage growth over N interactions
- reconsolidation hit/update rate

**Step 4: Run benchmark smoke test**

Run: `pytest tests/test_benchmarks.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add benchmarks/benchmark_retrieval.py benchmarks/datasets/sample_memory_eval.jsonl docs/benchmarks.md README.md tests/test_benchmarks.py
git commit -m "feat: add retrieval benchmark harness and baseline metrics"
```

### Task 8: Production Documentation and Messaging Correction

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `docs/MCP_INTEGRATION.md`
- Modify: `CHANGELOG.md`

**Step 1: Write failing docs consistency check**

Add a lightweight script/test that verifies:
- env vars in docs match code
- documented MCP resources exist
- no claim of unsupported backends as implemented

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_consistency.py -v`
Expected: FAIL on current drift.

**Step 3: Implement docs alignment**

Fix mismatches:
- `mnemos://memories` vs actual resources
- env var names and defaults
- backend support status with explicit labels (`implemented`, `planned`)
- alpha status and production caveats

**Step 4: Run consistency test**

Run: `pytest tests/test_docs_consistency.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add README.md ARCHITECTURE.md docs/MCP_INTEGRATION.md CHANGELOG.md tests/test_docs_consistency.py
git commit -m "docs: align claims with implementation and support matrix"
```

### Task 9: Final Release Gate

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `docs/release-checklist.md`

**Step 1: Write failing release checklist validation**

Add CI check requiring:
- unit tests
- type checks
- benchmark smoke
- docs consistency
- MCP/CLI integration tests

**Step 2: Run pipeline locally**

Run: `pytest -q && mypy .`
Expected: FAIL until all prior tasks complete.

**Step 3: Wire final CI pipeline**

Add separate jobs for:
- core tests
- integration tests
- release readiness checks

**Step 4: Verify green**

Run: `pytest -q && mypy .`
Expected: PASS.

**Step 5: Commit**

```bash
git add .github/workflows/ci.yml docs/release-checklist.md
git commit -m "chore: add production release gate and checklist"
```

## Execution Notes

- Execute tasks in order; Task 2 and Task 5 are highest value.
- Do not market "production-ready" until Task 7 metrics and Task 9 gate are both green.
- Keep `SimpleEmbeddingProvider` for local/dev, but never as the documented production default.
