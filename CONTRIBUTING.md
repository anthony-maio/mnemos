# Contributing to Mnemos

## Scope

Mnemos v1 is a local-first memory layer for solo coding-agent workflows. Contributions should preserve:

- scoped memory correctness
- deterministic local startup
- stable MCP tool contracts
- additive compatibility work over speculative platform branches

## Development workflow

1. Open the relevant issue template or draft note for behavior changes.
2. Add or update tests before implementation.
3. Keep changes small enough to review by subsystem.
4. Run the local verification set before opening a PR:
   - `python -m pytest -q`
   - `python -m mypy .`
   - `python -m black --check .`

## Release bar for changes

Changes that affect memory correctness, storage, or host compatibility should include:

- regression tests
- doc updates where the public contract changed
- explicit notes for compatibility or migration impact

## Areas where rigor matters most

- scope isolation across project/workspace/global memory
- persistent store correctness
- MCP tool signatures and semantics
- safety and redaction behavior
- startup and reconnection behavior for persisted stores

## Communication

- Prefer concrete bug reports with reproduction steps.
- Use the host compatibility template for MCP or client-specific issues.
- Prefer benchmark evidence over category claims.
- Mark experimental work clearly in code and docs.
