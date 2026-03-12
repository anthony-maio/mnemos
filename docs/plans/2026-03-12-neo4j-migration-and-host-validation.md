# Neo4j Migration And Host Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Neo4j viable as the active Mnemos backend by removing noisy query warnings, adding a supported migration path from existing stores, and validating live host capture/integration flows against the Neo4j-backed setup.

**Architecture:** Keep the migration logic store-agnostic by copying `MemoryChunk` objects through the existing `MemoryStore` interface instead of writing bespoke source-specific export code. Fix Neo4j warning noise at the query layer by switching projections/stats queries away from property access patterns that trigger missing-key warnings on optional fields. Validate host flows by exercising the shipped Claude hook path and by proving Codex/Cursor host configs resolve the active Neo4j-backed config correctly.

**Tech Stack:** Python, argparse CLI, pytest, Neo4j driver, existing Mnemos runtime/settings/host integration helpers.

---

### Task 1: Add a regression test for Neo4j missing-property warning-safe projections

**Files:**
- Modify: `mnemos/utils/storage.py`
- Modify: `tests/test_storage_neo4j.py`

**Step 1: Write the failing test**

Add a fake Neo4j fixture variant that only accepts bracket-property access such as `chunk['cognitive_state_json']` and `chunk['embedding']` in `get`, `get_all`, and `get_stats` queries.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_storage_neo4j.py -k warning`
Expected: FAIL because the current Cypher still uses direct property access.

**Step 3: Write minimal implementation**

Update Neo4j projection/stat queries in `mnemos/utils/storage.py` to use warning-safe property access for optional keys.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_storage_neo4j.py -k warning`
Expected: PASS

**Step 5: Verify against live Neo4j**

Run a local smoke script against `bolt://192.168.86.41:7687` and confirm the runtime no longer emits missing-property warnings during `get`, `get_all`, and `get_stats`.

**Step 6: Commit**

```bash
git add mnemos/utils/storage.py tests/test_storage_neo4j.py
git commit -m "fix(storage): suppress neo4j optional-property warnings"
```

### Task 2: Add a generic store migration CLI path

**Files:**
- Modify: `mnemos/cli.py`
- Modify: `mnemos/runtime.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

**Step 1: Write the failing tests**

Add CLI tests for a new migration command that:
- builds a source store and target store from one config/env with explicit source/target store type overrides
- supports dry-run output
- copies chunks and reports totals
- rejects unsupported source/target combinations or missing credentials

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_cli.py -k migrate`
Expected: FAIL because the command and helpers do not exist yet.

**Step 3: Write minimal implementation**

Add a small migration helper in the CLI/runtime path that:
- clones the resolved settings
- overrides `storage.type` for source and target
- builds both stores
- iterates `source_store.get_all()`
- writes each chunk into `target_store.store()`
- prints a JSON summary with `source_store`, `target_store`, `scanned`, `migrated`, `skipped`, and `dry_run`

**Step 4: Run targeted tests**

Run: `pytest -q tests/test_cli.py -k migrate`
Expected: PASS

**Step 5: Run live migration validation**

Use the active config with `qdrant` source and `neo4j` target. First run `--dry-run`, then run a real copy if the dry-run totals look correct.

**Step 6: Document usage**

Add README CLI examples for `sqlite -> neo4j` and `qdrant -> neo4j`.

**Step 7: Commit**

```bash
git add mnemos/cli.py mnemos/runtime.py tests/test_cli.py README.md
git commit -m "feat(cli): add store migration command"
```

### Task 3: Validate live host capture and integration flows on Neo4j

**Files:**
- Modify: `docs/codex.md`
- Modify: `docs/MCP_INTEGRATION.md`
- Modify: `docs/client-compatibility-matrix.md`
- Modify: `tests/test_hook_autostore.py`
- Modify: `tests/test_settings.py`

**Step 1: Write the failing tests**

Add tests that assert:
- host-import logic respects host MCP configs that point at `MNEMOS_CONFIG_PATH`
- Claude hook auto-store behavior is unchanged when the active store is Neo4j

**Step 2: Run targeted tests to verify they fail**

Run: `pytest -q tests/test_hook_autostore.py tests/test_settings.py -k neo4j`
Expected: FAIL because the validation coverage is missing.

**Step 3: Write minimal implementation/docs**

Update docs to state:
- Claude Code hooks are the validated auto-capture path on Neo4j
- Codex and Cursor resolve the same Neo4j-backed config through MCP, but still depend on explicit Mnemos tool usage today
- host validation checklist for live setups

**Step 4: Run live validation**

Execute:
- `mnemos-cli autostore-hook UserPromptSubmit ...`
- `mnemos-cli autostore-hook PostToolUse ...`
- `mnemos-cli list --scope project --scope-id <repo>`

Confirm the chunks land in Neo4j and can be retrieved/inspected.

**Step 5: Commit**

```bash
git add docs/codex.md docs/MCP_INTEGRATION.md docs/client-compatibility-matrix.md tests/test_hook_autostore.py tests/test_settings.py
git commit -m "docs: validate host flows for neo4j-backed setups"
```

### Task 4: Final verification and push

**Files:**
- Verify: `mnemos/utils/storage.py`
- Verify: `mnemos/cli.py`
- Verify: `docs/`
- Verify: `tests/`

**Step 1: Run targeted integration checks**

Run:
- `pytest -q tests/test_storage_neo4j.py`
- `pytest -q tests/test_cli.py -k migrate`
- `pytest -q tests/test_hook_autostore.py tests/test_settings.py`

**Step 2: Run full suite**

Run: `pytest -q`
Expected: PASS

**Step 3: Push**

```bash
git push origin master
```
