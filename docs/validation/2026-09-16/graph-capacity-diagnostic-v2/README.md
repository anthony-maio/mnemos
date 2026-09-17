# Graph capacity diagnostic v2

This is a development-only offline diagnostic. It uses the frozen d94ded67a6b3fe6b6f7474a74518d4efb143698e source checkout and its frozen development vectors. It makes no model or network calls and does not change production source, defaults, gates, or datasets.

The pre-registered arms were max_neighbors_per_node values 4, 8, and 16. Each arm ran the same five frozen development cases once. The run completed 15 of 15 observations and constructed 75 SQLite stores. Every store persisted the arm capacity: 25 each at capacities 4, 8, and 16.

Observed outcomes:

- Capacity 4: 0 of 5 eligible, 0 of 5 successful.
- Capacity 8: 0 of 5 eligible, 0 of 5 successful.
- Capacity 16: 1 of 5 eligible, 0 of 5 successful.
- Fourteen observations had no persisted structural path. One capacity-16 observation had both a structural and activation path and was eligible, but did not satisfy the success condition.
- All 15 activation-versus-independent-path checks matched.
- All 15 insertion-order edge checks and all 15 insertion-order result checks matched.
- Actual selected seed IDs were present in all 15 observations.

Eligibility and success are graph mechanism outcomes only. The run is not evidence of held-out quality, model quality, or a performance result. Other workloads ran concurrently, so wall-clock values in the raw observations are diagnostic context only.

The earlier v1 attempt is retained separately and excluded from conclusions. Its capacity-4 work completed, but it was interrupted during a higher arm after changing capacity after engine construction. V2 binds capacity before construction and verifies each persisted graph contract.

Run command:

    $snapshot = (Resolve-Path .venv/thinking-frozen-v1).Path
    $env:PYTHONPATH = $snapshot
    try { .venv/Scripts/python.exe .venv/graph_capacity_diagnostic_v2.py } finally { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }

The run manifest records the interpreter, command, source and vector hashes, helper hash, and imported frozen module paths and hashes. The helper is archived as graph_capacity_diagnostic_v2.py.txt so this evidence directory contains no executable duplicate.
