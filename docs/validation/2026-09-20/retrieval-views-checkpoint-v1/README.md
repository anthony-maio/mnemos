# Separate captured evidence from ordinary recall

Frozen source `e419c2aa615e037e639a74b14615889d03f4e47f` passes 1,500
source tests with 4 skips, all six source/package checks, and
648 installed contract cases with 4 matching skips.
Seven installed replay assertions also pass. Individual installed case outcomes,
module origins, and package bytes match the frozen source and wheel.

Wheel: 303,727 bytes; SHA-256 `185f3df7720a32e5ed285d9c507e1e45e7f57370942740b53c4f8178bb97a97e`.

## Behavior

Ordinary retrieval defaults to `view="memory"`, excluding only chunks marked
`source: captured_passthrough`. Explicit `view="captured_evidence"` returns those
chunks in `captured_evidence` and `evidence_context`, leaving ordinary `facts`
and `context` empty. Other permitted unknown or legacy records remain eligible memory.
The source predicate combines with scope checks before candidate limits,
lexical ranking, graph traversal, and context budgeting. Both channels retain
source evidence and the existing UTF-8 byte upper bound.

Hermes and MCP expose the view on their existing retrieve tool; the CLI has
`--view captured_evidence`. Automatic recall stays on the memory view. The
inspector retains captured records, labels them as unverified, and explains
their evidence-view retrieval. This changes retrieval behavior without
migrating data, changing capture consent, or deleting captured copies.

Regression coverage includes both storage backends, candidate starvation,
foreign profiles and scopes, legacy unknown records, graph bridges, context
budgeting, correction authority after restart, and related-forget inventory.
Authenticated service, Hermes dispatch, MCP wrapper, and CLI subprocess tests
exercise the response channels. The host adapter checks retain their assertion
and conflict labels while separately reviewing captured history. This does
not establish a native model conversation or a browser acceptance result.

## Evidence limits

The [v7 cohort](../hermes-native-cohort-v7/README.md) remains zero of three
complete lifecycles on its original runtime. No new native model cohort was
run for this checkpoint. Separating evidence addresses an API contract gap;
it does not prove why the model omitted correction, inspection, or job-status
calls. Stored-copy erasure checks, frozen model criteria, and old negative
results are unchanged. Hiding a copy from recall cannot satisfy erasure.

Development evidence includes the intended missing-field/API failures and
test-authoring mistakes (source-status label, canonical event identity, and
preview field names). Those mistakes are not product defects. The origin-check
log named `discrepancy-red-v1` passed six source cases; it did not reproduce
an import-origin failure and is not RED evidence. An evolving
development suite was interrupted and has no complete result; the frozen
source run is the basis for the counts above. Any prior frozen run is retained
separately. The first frozen run at `f49c86c` had 13 failures, 1,481 passes,
and four skips: the CLI stub lacked the new argument, three MCP roundtrips
still expected raw capture in ordinary facts, the empty forget response
expected the old shape, and eight comparison tests exercised an older engine
signature. Ordinary service calls now rely on the engine's default memory
view so older custom engines remain callable. No benchmark scoring,
protocol, dataset, or recorded outcome was edited. Source tests explicitly
review capture and retain the original scope, mutation, and replay checks.
The second frozen run at `034cbd9` passed 1,496 tests with four skips, then
stopped at a mypy override error: the benchmark-only subclass still declared
the older retrieval signature. Its signature now accepts the new argument and
explicitly rejects evidence-view experiments. Ranking, scoring, and historical
results are unchanged. Reproduce earlier experiments with their pinned source.
CLI and MCP subprocess tests also bind child imports to the parent's package
and check the origin, so their installed-wheel run cannot silently load the
source checkout. Direct inspection gained a view-aware trace lookup. No live host
configuration or live memory was changed.

The evidence index binds 81 raw files. This interpretive README is
outside that index. Native lifecycle, three-host continuity, complete capture,
model-quality acceptance, Desktop, and the two-project pilot remain open.

## Subsequent native observation

The [fixed v8 cohort](../hermes-native-cohort-v8/README.md) ran this verified
package and failed all three full lifecycles. All readiness and cleanup checks
passed. One attempt failed on restarted recall, and two passed correction and
restarted corrected recall before failing at forgetting. No successful deletion,
source replay, or final recall was reached. These outcomes do not alter the
package checks or establish a causal effect of the retrieval-view change.
