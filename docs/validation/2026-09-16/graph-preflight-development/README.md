# Graph eligibility preflight development evidence

This directory preserves three diagnostic passes over the same five-case development fixture. The fixture is cached development evidence. It is not held-out evaluation evidence and it does not establish graph utility.

## What each pass means

- `graph_eligibility_preflight_v1` is a retrospective diagnostic. It reused persisted measurements and ran an independent bounded path check. It is preserved for history and is not a production retrieval run.
- `graph_eligibility_preflight_v2` called the production measurement helper, but its source-binding protocol was invalid because it did not require a clean, explicitly pinned source snapshot. Its zero eligible cases must not be used as a source-parity result. Its eligibility condition also treated `target_not_in_graph_results` as a prerequisite, so a successful graph retrieval could have been counted ineligible; that result is retained only as an invalid diagnostic.
- `graph_eligibility_preflight_v3` is the pinned development preflight. It ran `benchmarks.local_memory.graph_measurement.measure_graph_repetition` from the clean snapshot `.venv/graph-v4-frozen` at commit `b7a9d9d4d9d6947564696457b0e2754f7e95a801`. Imports for the production engine, storage, measurement helper, and mechanism helper resolved inside that snapshot. The source hashes matched before and after the run.

The v3 run covered 5 cases and produced 0 structurally eligible cases. Every case had a persisted graph artifact and passed the recorded vector, scope, selected-seed, and semantic-floor diagnostics. Those diagnostics are prerequisites recorded for review; they are not the eligibility definition. The production row's `eligible` field remained false because the actual graph retrieval had no valid persisted path to the target. All five cases reported `activation_valid_path=false`, `target_in_graph_results=false`, and `path_reason=no_persisted_path`. The coverage gate therefore failed, with zero observed graph path utility in this development preflight.

The authored seed label is retained separately from the query-selected vector seed. The target was outside the direct candidate union in these cases. No model or network calls were made.

## Reproduction inputs

- Fixture: `.venv/mechanism-development-work-v2/frozen-vectors.json`
- Fixture SHA256: `1370c4ad2cde46bfb9c333f75eaaf8a53bf978c0b916b5aedead530b755efc36`
- Required clean source snapshot: `.venv/graph-v4-frozen`
- Required snapshot commit: `b7a9d9d4d9d6947564696457b0e2754f7e95a801`
- Graph contract in the fixture: edge builder 4, schema 1, traversal 2

Run the v3 helper from the repository root with the pinned snapshot and an isolated work directory:

```powershell
.venv\Scripts\python.exe docs\validation\2026-09-16\graph-preflight-development\graph_eligibility_preflight_v3.py.txt `
  --snapshot .venv\graph-v4-frozen `
  --source-commit b7a9d9d4d9d6947564696457b0e2754f7e95a801 `
  --fixture .venv\mechanism-development-work-v2\frozen-vectors.json `
  --work-dir .venv\graph-eligibility-preflight-v3-work-replay `
  --output .venv\graph-eligibility-preflight-v3-replay\report.json
```

The original helpers and reports remain under `.venv`. The runner copies here use the `.py.txt` evidence convention and are byte-preserved archives. Copy a runner to a temporary `.py` path before execution, or pass the `.py.txt` path directly to Python. `evidence-index.json` records their bytes and SHA256 values, along with the fixture identity and the v3 imported source hashes.
