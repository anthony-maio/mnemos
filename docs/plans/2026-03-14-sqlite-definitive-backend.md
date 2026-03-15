# SQLite Definitive Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Mnemos' multi-backend storage story with a single self-contained SQLite backend that stores chunks, graph edges, lexical index data, and vector search state in one local `mnemos.db` file.

**Architecture:** Keep Mnemos' graph semantics in the engine and persist the graph as sparse weighted adjacency inside SQLite. Use one store implementation and one runtime path everywhere; legacy Neo4j and Qdrant support become one-time import sources during migration and are then removed from user-facing config, docs, and tests.

**Tech Stack:** Python `sqlite3`, SQLite JSON + FTS5, `sqlite-vec` for in-process vector search, existing Mnemos engine + spreading activation modules.

---

### Task 1: Freeze the public storage contract around one backend

**Files:**
- Modify: `D:\Development\mnemos\mnemos\mnemos\settings.py`
- Modify: `D:\Development\mnemos\mnemos\mnemos\runtime.py`
- Modify: `D:\Development\mnemos\mnemos\mnemos\health.py`
- Test: `D:\Development\mnemos\mnemos\tests\test_settings.py`
- Test: `D:\Development\mnemos\mnemos\tests\test_runtime.py`

**Step 1: Write the failing tests**

Add tests asserting:
- `storage.type` only accepts `"memory"` and `"sqlite"` during the transition window, then `"sqlite"` only once migration code is complete.
- `build_store_from_env()` returns `SQLiteStore` for the default persistent path.
- health/doctor output no longer advertises Neo4j or Qdrant profiles.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_settings.py tests/test_runtime.py -q`
Expected: FAIL because settings/runtime still expose `qdrant` and `neo4j`.

**Step 3: Write minimal implementation**

- Remove `qdrant` and `neo4j` from the user-facing `StorageSettings.type` literal.
- Delete env parsing for `MNEMOS_QDRANT_*` and `MNEMOS_NEO4J_*` from runtime configuration loading, or gate it strictly behind migration commands.
- Simplify `build_store()` in `runtime.py` so the persistent path is always `SQLiteStore`.
- Update doctor/health messaging to describe one local backend only.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_settings.py tests/test_runtime.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add mnemos/settings.py mnemos/runtime.py mnemos/health.py tests/test_settings.py tests/test_runtime.py
git commit -m "refactor: collapse runtime storage selection to sqlite"
```

### Task 2: Upgrade SQLiteStore into the full graph store

**Files:**
- Modify: `D:\Development\mnemos\mnemos\mnemos\utils\storage.py`
- Create: `D:\Development\mnemos\mnemos\tests\test_storage_sqlite.py`
- Test: `D:\Development\mnemos\mnemos\tests\test_storage_touch.py`

**Step 1: Write the failing tests**

Add SQLite-focused tests covering:
- schema creation for `memory_chunks`, `memory_edges`, `memory_fts`, and schema version metadata
- `replace_graph_neighbors()` round-trip persistence
- cascading edge cleanup on chunk delete
- stats reporting for chunk count, edge count, and FTS/vector availability

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage_sqlite.py tests/test_storage_touch.py -q`
Expected: FAIL because SQLiteStore currently has only `memory_chunks` and no graph persistence.

**Step 3: Write minimal implementation**

Extend `SQLiteStore` with:
- `memory_edges(source_id TEXT, target_id TEXT, weight REAL, edge_type TEXT, updated_at TEXT, PRIMARY KEY(source_id, target_id, edge_type))`
- indexes on `source_id`, `target_id`, and `(edge_type, source_id)`
- delete triggers or explicit cleanup for outgoing/incoming edges when a chunk is removed
- FTS5 table plus sync triggers or explicit rebuild helpers
- `get_graph_edges()` and `replace_graph_neighbors()` implementations matching the current `MemoryStore` contract

Keep graph semantics in Python; SQLite is only persisting the adjacency and serving search primitives.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_storage_sqlite.py tests/test_storage_touch.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add mnemos/utils/storage.py tests/test_storage_sqlite.py tests/test_storage_touch.py
git commit -m "feat: persist mnemos graph state in sqlite"
```

### Task 3: Replace Python-side O(n) vector scan with sqlite-vec

**Files:**
- Modify: `D:\Development\mnemos\mnemos\mnemos\utils\storage.py`
- Modify: `D:\Development\mnemos\mnemos\pyproject.toml`
- Modify: `D:\Development\mnemos\mnemos\mnemos\health.py`
- Test: `D:\Development\mnemos\mnemos\tests\test_storage_sqlite.py`

**Step 1: Write the failing tests**

Add tests asserting:
- SQLiteStore can initialize a `sqlite-vec` virtual table/index when the extension is available
- vector retrieval returns the same ordering as the current cosine path for a fixed fixture
- health output surfaces whether vector acceleration is active

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage_sqlite.py -q`
Expected: FAIL because SQLiteStore currently computes similarity only in Python.

**Step 3: Write minimal implementation**

- Add `sqlite-vec` as a package dependency or default extra for local installs.
- Initialize the vector table/index inside SQLiteStore startup/migrations.
- Route `retrieve()` through `sqlite-vec` nearest-neighbor queries.
- Keep one short-lived fallback path only for environments where the extension is unavailable during migration work; do not expose this as a user-facing alternate backend.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_storage_sqlite.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add mnemos/utils/storage.py mnemos/health.py pyproject.toml tests/test_storage_sqlite.py
git commit -m "feat: add sqlite-vec retrieval to sqlite backend"
```

### Task 4: Add a one-way legacy import path into SQLite

**Files:**
- Modify: `D:\Development\mnemos\mnemos\mnemos\cli.py`
- Modify: `D:\Development\mnemos\mnemos\mnemos\utils\storage.py`
- Test: `D:\Development\mnemos\mnemos\tests\test_cli.py`
- Test: `D:\Development\mnemos\mnemos\tests\test_storage_neo4j.py`
- Test: `D:\Development\mnemos\mnemos\tests\test_storage_qdrant.py`

**Step 1: Write the failing tests**

Add tests for a new migration shape such as:
- `mnemos-cli migrate-legacy --from neo4j --to sqlite`
- `mnemos-cli migrate-legacy --from qdrant --to sqlite`
- `mnemos-cli migrate-legacy --from sqlite --to sqlite` for schema upgrades

Assert imported chunks and edges land in the unified SQLite schema.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py tests/test_storage_neo4j.py tests/test_storage_qdrant.py -q`
Expected: FAIL because current migration commands are symmetric multi-backend plumbing.

**Step 3: Write minimal implementation**

- Replace general-purpose source/target store migration with a one-way import command into SQLite.
- Keep Neo4j/Qdrant reader code only as import adapters.
- Ensure the command rehydrates graph edges into the new `memory_edges` table.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py tests/test_storage_neo4j.py tests/test_storage_qdrant.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add mnemos/cli.py mnemos/utils/storage.py tests/test_cli.py tests/test_storage_neo4j.py tests/test_storage_qdrant.py
git commit -m "feat: add legacy store import into sqlite"
```

### Task 5: Delete alternative backend runtime paths

**Files:**
- Modify: `D:\Development\mnemos\mnemos\mnemos\utils\storage.py`
- Modify: `D:\Development\mnemos\mnemos\mnemos\runtime.py`
- Modify: `D:\Development\mnemos\mnemos\mnemos\settings.py`
- Delete: `D:\Development\mnemos\mnemos\tests\test_storage_neo4j.py`
- Delete: `D:\Development\mnemos\mnemos\tests\test_storage_qdrant.py`

**Step 1: Write the failing tests**

Add or update tests asserting:
- runtime no longer instantiates QdrantStore or Neo4jStore
- settings no longer document/store those providers
- migration command still works using import adapters or exported files

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_settings.py tests/test_runtime.py tests/test_cli.py -q`
Expected: FAIL until references to removed backends are gone.

**Step 3: Write minimal implementation**

- Remove `QdrantStore` and `Neo4jStore` from runtime imports and user-facing store selection.
- If needed, move legacy import readers behind private migration-only helpers instead of public store classes.
- Delete obsolete backend tests once import coverage exists elsewhere.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_settings.py tests/test_runtime.py tests/test_cli.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add mnemos/utils/storage.py mnemos/runtime.py mnemos/settings.py tests/test_settings.py tests/test_runtime.py tests/test_cli.py
git rm tests/test_storage_neo4j.py tests/test_storage_qdrant.py
git commit -m "refactor: remove alternative storage backends"
```

### Task 6: Rewrite docs, install flow, and host configuration around one file

**Files:**
- Modify: `D:\Development\mnemos\mnemos\README.md`
- Modify: `D:\Development\mnemos\mnemos\docs\MCP_INTEGRATION.md`
- Modify: `D:\Development\mnemos\mnemos\docs\codex.md`
- Modify: `D:\Development\mnemos\mnemos\skills\mnemos-memory\SKILL.md`
- Modify: `D:\Development\mnemos\mnemos\skills\mnemos-codex\SKILL.md`
- Modify: `D:\Development\mnemos\mnemos\CHANGELOG.md`

**Step 1: Write the failing doc/tests**

If doc assertions exist, add or update them. Otherwise, make a checklist of text that must disappear:
- Neo4j setup
- Qdrant setup
- profile-based “upgrade” guidance
- claims that different backends yield different scale tiers

**Step 2: Verify current docs are inconsistent**

Run targeted grep/search or doc tests.
Expected: current docs still mention SQLite/Qdrant/Neo4j choices and profile branching.

**Step 3: Rewrite docs**

Make the install story:
- install Mnemos
- run one config/bootstrap command
- get one local `mnemos.db`
- optionally import old data from legacy stores

Position the differentiator as Mnemos' modules and inspectable graph behavior, not infra choices.

**Step 4: Verify docs are consistent**

Run: `pytest -q`
Expected: PASS

**Step 5: Commit**

```bash
git add README.md docs/MCP_INTEGRATION.md docs/codex.md skills/mnemos-memory/SKILL.md skills/mnemos-codex/SKILL.md CHANGELOG.md
git commit -m "docs: adopt sqlite as the definitive mnemos backend"
```

### Task 7: Run full regression and dogfood the unified store

**Files:**
- Modify as needed based on failures from earlier tasks
- Test: `D:\Development\mnemos\mnemos\tests\test_engine.py`
- Test: `D:\Development\mnemos\mnemos\tests\test_modules.py`

**Step 1: Run the full suite**

Run:
- `pytest -q`
- `black --check .`
- `mypy .`

Expected: PASS

**Step 2: Run a local end-to-end smoke**

Run:
- `python -m mnemos.cli doctor`
- `python -m mnemos.cli store --content "sqlite unification smoke test"`
- `python -m mnemos.cli retrieve --query "sqlite unification smoke test"`

Expected:
- doctor reports SQLite as the only persistent backend
- store succeeds into `mnemos.db`
- retrieve returns the stored item

**Step 3: Commit any last fixes**

```bash
git add .
git commit -m "test: verify unified sqlite backend end to end"
```
