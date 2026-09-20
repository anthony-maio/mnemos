# Reviewed-forget rejection package checkpoint

Frozen source `db217509477d92b9a82bd7e11d4a4bffb542fa29` passes 1,692 source tests
with 4 skips, all six source/package checks, 669 isolated
installed cases with 4 matching skips, and seven replay assertions.
Installed validation compares case outcomes, import origins, and package bytes
against the frozen source and wheel.

Wheel: 304,629 bytes; SHA-256 `8b1f560e98b93376fc2aa073be7c1500997d888d09ae87573d5d70ec8694ad00`.

## What changed

The [development record](../review-rejection-development-v1/README.md)
preserves the rejected-request outbox regression and the broader rejection
tests. A valid pending capture previously replayed after an invalid related
forget selection. The earlier regression used an invalid capture, so its
unchanged outbox did not prove that replay was skipped. A shared rejection
map now governs both strict response decoding and the return before replay.

Incomplete previews, duplicate selections, and dependency conflicts receive
fixed, noncommitting HTTP 409 responses. Their client and Hermes paths retain
pending captures without replay. An explicit later replay commits a valid
capture. Scope, digest, version, dependency, and deletion requirements remain
unchanged; unknown errors retain opaque HTTP 500 responses.

## Evidence limits

The [native Hermes v12 cohort](../hermes-native-cohort-v12/README.md) ran the
preceding `5155ebd` package and finished zero of three complete lifecycles.
One attempt ultimately committed target-only deletion, without establishing
reviewed-copy erasure. No attempt reached replay or final recall. Its model
behavior is not an observation of this repaired package.

No native cohort has run this checkpoint. Native continuity, complete capture,
full-module quality, Desktop, and the operational pilot remain open. Graph
development measurements remain negative. Passing these package checks does
not establish model reliability or retrieval utility.

No live host configuration or memory data was changed. The evidence index
binds the raw files; this README is outside that index.
