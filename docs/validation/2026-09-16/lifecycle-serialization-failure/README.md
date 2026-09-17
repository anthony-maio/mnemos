# Lifecycle serialization failure

This folder preserves the lifecycle failure reported by the b7 real-model queue. The coordinator stopped with exit code 1 at 2026-09-17T12:21:39.159502+00:00. The run used source commit `b7a9d9d4d9d6947564696457b0e2754f7e95a801` from the frozen `graph-v4-frozen` checkout.

The full profile had all five modules enabled and three repetitions. Its checkpoint contains 36 observations: 6 with `execution_status=complete` and 30 with `execution_status=partial`. They end at `heldout-012:r3`, with `next_session_index=11`. The next `013:r1` observation was in progress when checkpoint serialization failed. The failure is a `TypeError` from `json.dump`: `Object of type ValueError is not JSON serializable`. The manifest is marked failed; the checkpoint still says running because serialization failed while writing the next observation checkpoint.

The failed attempt database is inspected read-only and is not copied here. Its SHA256 is recorded in `failed-db-diagnostic.json`. At the saved boundary it contains one completed `remember` job and one pending `extract` job with `attempt_count=1` and `last_error=ValidationError`. The database diagnostic contains job and event status counts without conversation payloads.

The failed observation's complete in-memory model response was not preserved. The checkpoint contains the 36 observations already written to it, while the logs preserve the traceback. No quality conclusion can be drawn from the failed full run, and the full and ablation runs remain incomplete.

The baseline artifacts are separate immutable evidence. `baseline.json` and `baseline.raw-outputs.jsonl` contain the completed baseline run. The larger baseline checkpoint was deliberately not copied because it duplicates that evidence.

All copied files are indexed by relative path, byte count, and SHA256 in `evidence-index.json`. The package contains no lifecycle database, credentials, or live host configuration.
