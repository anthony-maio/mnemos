# Claude Stop replay repair

The original source20f SQLite reproducer grew from two events to three after
restart and replay. The repaired source keeps the same two event IDs and one
native assistant observation. All seven positive assertions passed against
the clean frozen `554a2517dc85bbdb6190b5e523f8511efa54cccf` checkout.

The adapter repair is commit `8d85894`; `554a251` adds the separate benchmark
instrumentation correction. The first Stop captured both native transcript
records. After reopening SQLite and constructing a fresh adapter, the same
Stop captured zero events. The replay explicitly reports degraded identity
coverage because the payload has no native prompt ID.

The fix persists a scoped, file-bound assistant fingerprint and source
reference without copying assistant text into the checkpoint. It validates
the marker, distinguishes a known empty checkpoint from unknown older state,
and refuses fallback after a rejected transcript capture. Missing stable
transcript IDs fail closed. Invalid Stop prompt IDs use degraded fallback
semantics.

Fallback and transcript records are deduplicated only when an explicit prompt
reference establishes their association. When association is unavailable,
the native UUID observation is retained with an ambiguity diagnostic. This can
preserve two observations of one message; it does not claim complete native
deduplication. Delayed same-text messages with distinct native identities must
not be silently discarded.

Focused validation passed 54 adapter and identity tests, formatting, and type
checking. The benchmark-only change passed four focused tests, including a
check through the measurement function for one metadata write per engine
query. The frozen source subsequently passed 691 tests, two optional WSL
skips, formatting, typing, JavaScript, build, and package checks. Its freshly
installed wheel also passed the seven replay assertions; see the
[full and installed verification](../claude-replay-verification/README.md).

This probe uses real SQLite and the local service contract with constant
vectors and unavailable cognition. It is a source-checkout check, not an
native Claude acceptance result. The separate installed check is linked above.
The original negative
probe remains in [claude-stop-replay](../claude-stop-replay/README.md).

The exact positive verifier, unchanged original reproducer, original report,
and their SHA256 hashes are preserved in `evidence-index.json`.
