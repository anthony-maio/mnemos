# Mnemos Inspectability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first real inspectability slice for Mnemos so users can see why a memory exists, how it changed, and how it relates to scope and graph context from the CLI, MCP, and control plane.

**Architecture:** Add a shared inspectability serializer over `MemoryChunk` plus engine/module state, then expose that serializer through a new CLI command, the existing MCP inspect tool, and a control-plane memory detail view. Persist lightweight provenance and reconsolidation history in chunk metadata so the inspector has real data instead of inferring from current state only.

**Tech Stack:** Python 3.10+, argparse CLI, existing MCP server, local control-plane HTTP UI, Pydantic models, pytest, mypy, black.

---

### Task 1: Add inspectability metadata and shared serializer

**Files:**
- Create: `mnemos/inspectability.py`
- Modify: `mnemos/types.py`
- Modify: `mnemos/engine.py`
- Modify: `mnemos/modules/mutable_rag.py`
- Modify: `mnemos/modules/spreading.py`
- Test: `tests/test_inspectability.py`

**Step 1: Write the failing tests**

```python
def test_build_chunk_inspection_reports_scope_provenance_and_graph_context() -> None:
    ...
    payload = build_chunk_inspection(engine, chunk.id)
    assert payload["scope"] == "project"
    assert payload["provenance"]["stored_by"] == "surprisal_gate"
    assert payload["graph"]["neighbor_count"] == 1


def test_reconsolidation_appends_revision_history() -> None:
    ...
    updated_chunk, changed = await rag.reconsolidate(chunk, "User migrated to Svelte")
    assert changed is True
    assert updated_chunk.metadata["revision_history"][0]["from_version"] == 1
    assert updated_chunk.metadata["revision_history"][0]["to_version"] == 2
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_inspectability.py`

Expected: FAIL because `build_chunk_inspection` and persistent revision history do not exist yet.

**Step 3: Write the minimal implementation**

```python
def build_chunk_inspection(engine: MnemosEngine, chunk_id: str) -> dict[str, Any] | None:
    chunk = engine.store.get(chunk_id)
    if chunk is None:
        return None
    node = engine.spreading_activation.get_node(chunk_id)
    return {
        "id": chunk.id,
        "content": chunk.content,
        "scope": ...,
        "scope_id": ...,
        "provenance": {
            "stored_by": chunk.metadata.get("source"),
            "ingest_channel": chunk.metadata.get("ingest_channel"),
            "reason": chunk.metadata.get("encoding_reason"),
        },
        "history": chunk.metadata.get("revision_history", []),
        "graph": {
            "neighbor_count": 0 if node is None else len(node.neighbors),
        },
    }
```

Also:
- stamp `encoding_reason`, `ingest_channel`, and normalized scope metadata in `MnemosEngine.process()`
- append a bounded `revision_history` entry during `MutableRAG.reconsolidate()`
- add a small graph summary helper in `SpreadingActivation` if needed

**Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/test_inspectability.py`

Expected: PASS

**Step 5: Commit**

```bash
git add mnemos/inspectability.py mnemos/types.py mnemos/engine.py mnemos/modules/mutable_rag.py mnemos/modules/spreading.py tests/test_inspectability.py
git commit -m "feat(inspectability): add chunk provenance and revision history"
```

### Task 2: Expose inspectability through CLI and MCP

**Files:**
- Modify: `mnemos/cli.py`
- Modify: `mnemos/mcp_server.py`
- Modify: `docs/mcp-transport-contract.md`
- Test: `tests/test_cli.py`
- Test: `tests/test_mcp_server.py`

**Step 1: Write the failing tests**

```python
def test_cli_inspect_outputs_chunk_details(monkeypatch, capsys) -> None:
    ...
    exit_code = main(["inspect", chunk.id])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == chunk.id
    assert payload["provenance"]["stored_by"] == "surprisal_gate"


@pytest.mark.asyncio
async def test_mcp_inspect_returns_lineage_and_graph_context() -> None:
    ...
    result = await mnemos_inspect(chunk.id, ctx=ctx)
    payload = json.loads(result)
    assert "history" in payload
    assert "graph" in payload
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_cli.py tests/test_mcp_server.py`

Expected: FAIL because CLI has no `inspect` command and MCP inspect does not return the richer payload.

**Step 3: Write the minimal implementation**

```python
inspect_parser = subparsers.add_parser("inspect", help="Inspect one stored memory chunk.")
inspect_parser.add_argument("chunk_id")
```

Implementation notes:
- route CLI `inspect` to `build_chunk_inspection()`
- keep output JSON-first for automation
- update `mnemos_inspect` to use the same serializer so CLI and MCP stay aligned
- document the richer response shape in the transport contract

**Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/test_cli.py tests/test_mcp_server.py`

Expected: PASS

**Step 5: Commit**

```bash
git add mnemos/cli.py mnemos/mcp_server.py docs/mcp-transport-contract.md tests/test_cli.py tests/test_mcp_server.py
git commit -m "feat(inspectability): expose chunk inspection in cli and mcp"
```

### Task 3: Add control-plane memory detail API and viewer

**Files:**
- Modify: `mnemos/control_plane.py`
- Modify: `mnemos/ui_server.py`
- Modify: `mnemos/ui/index.html`
- Modify: `mnemos/ui/app.js`
- Modify: `mnemos/ui/styles.css`
- Test: `tests/test_control_plane.py`
- Test: `tests/test_ui_server.py`

**Step 1: Write the failing tests**

```python
def test_control_plane_memory_detail_returns_inspection_payload(tmp_path: Path) -> None:
    ...
    payload = service.get_memory_detail(chunk.id)
    assert payload["id"] == chunk.id
    assert payload["history"] == []


def test_ui_router_serves_memory_detail_endpoint(tmp_path: Path) -> None:
    ...
    response = router.handle("GET", f"/api/memory/{chunk.id}", None)
    assert response.status == 200
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_control_plane.py tests/test_ui_server.py`

Expected: FAIL because no memory detail API exists yet.

**Step 3: Write the minimal implementation**

```python
def get_memory_detail(self, chunk_id: str) -> dict[str, Any]:
    engine = MnemosEngine(...)
    payload = build_chunk_inspection(engine, chunk_id)
    if payload is None:
        raise KeyError(chunk_id)
    return payload
```

UI behavior:
- keep the current recent-memory list
- make each memory row clickable
- add a detail panel showing:
  - content
  - scope
  - provenance
  - revision history
  - graph summary
  - raw metadata JSON

Do not add rollback or editing yet.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/test_control_plane.py tests/test_ui_server.py`

Expected: PASS

**Step 5: Commit**

```bash
git add mnemos/control_plane.py mnemos/ui_server.py mnemos/ui/index.html mnemos/ui/app.js mnemos/ui/styles.css tests/test_control_plane.py tests/test_ui_server.py
git commit -m "feat(inspectability): add control-plane memory detail view"
```

### Task 4: Final verification and docs polish

**Files:**
- Modify: `README.md`
- Modify: `docs/MCP_INTEGRATION.md`
- Modify: `docs/codex.md`
- Test: `tests/test_docs_consistency.py`

**Step 1: Write the failing docs consistency checks**

```python
def test_readme_mentions_inspectability_surface() -> None:
    ...
    assert "mnemos inspect" in readme_text
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_docs_consistency.py`

Expected: FAIL if docs do not mention the new inspectability entry points.

**Step 3: Write the minimal implementation**

Update docs to show:
- CLI `mnemos inspect <id>`
- MCP `mnemos_inspect`
- control-plane Memory detail view
- what provenance/history currently covers
- what is still intentionally out of scope (`rollback`, full trace explorer)

**Step 4: Run final verification**

Run:
- `python -m pytest -q`
- `python -m mypy .`
- `python -m black --check .`

Expected:
- all tests pass
- no mypy errors
- black reports no formatting changes needed

**Step 5: Commit**

```bash
git add README.md docs/MCP_INTEGRATION.md docs/codex.md tests/test_docs_consistency.py
git commit -m "docs(inspectability): document inspector surfaces"
```
