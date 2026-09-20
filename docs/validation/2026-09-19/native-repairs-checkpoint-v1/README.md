# Native failure repair package checkpoint

Frozen source `fd83fb27042c05897f29dba952b0924b15889aad` passes
1,351 source tests with 4 skips, all six source/package checks,
569 installed cases with 4 matching skips, and seven installed replay
assertions. The installed run checks each selected case/outcome, package bytes,
and imported module origins against the frozen source and built wheel.

Wheel: 299,463 bytes; SHA-256
`4958c5ef7a509fefecacc0666d87f578b46e94e1a33a896831468c0f971bf895`.

## Repairs

- The bridge writes every protocol response as explicit UTF-8 bytes. Real
  subprocess/service tests cover cp1252, ASCII, and UTF-16 text-stream settings,
  Unicode content, oversized requests, and malformed requests.
- Inspect, correct, and forget return the same `404/fact_not_found` for absent,
  deleted, and inaccessible fact IDs. Direct callers retain KeyError compatibility.
  Unexpected internal KeyError still returns a generic 500. This change does
  not alter the separate related-forget error contract.
- Automatic assistant captures quoting an explicitly forgotten fact ID are
  suppressed before persistence. Atomic fact commit rechecks source and candidate
  text to prevent a capture-before-forget/commit-after-forget race from producing
  new facts. Batched exact-token lookups use a non-destructive SQLite index.

The end-to-end fixture teaches, corrects, forgets, delivers delayed assistant
narration, reopens the database, and replays that narration. Separate project
and profile controls survive; explicit reteaching creates a new fact ID.

## Limits and prior evidence

This is a bounded exact-ID safeguard. IDs composed of ASCII letters, digits,
underscores, and hyphens, including native UUIDs, are supported. An assistant
event mixing an erased ID with unrelated notes is conservatively suppressed
as a whole. User captures and explicit writes remain eligible. A source already
captured before forgetting can remain in raw storage even when its later
fact extraction is suppressed. This does not prove arbitrary paraphrase
erasure, general raw-source erasure, or native-model reliability.

The [v4 native cohort](../hermes-native-cohort-v4/README.md) remains zero of
three full lifecycle passes on its earlier `8f8f63f` runtime. This package has
not had a new native-model lifecycle run. Model correction omission, complete
capture, three-host continuity, module-quality acceptance, and the pilot remain
open.

The preserved pre-fix `8f8f63f` source was checked retrospectively with the new
regressions: six fact-error assertions fail (three controls pass); four bridge
cases fail (two pass); the delayed-narration lifecycle fails because the old
source accepts the capture as pending. Reports bind imported paths, old source,
test hashes, commands, and outcomes. These archived runs are post-fix regression
proof, not a claim that every behavioral failure was recorded before coding.

The evidence index binds 45 files: source/installed checks, regression
reports, and exact executed helpers. This README is outside the raw index.
Live host configuration and live memory data were unchanged.
