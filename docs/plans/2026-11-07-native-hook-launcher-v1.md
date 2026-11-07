# Native hook command boundary development probe

Freeze this protocol and the executable before measuring. Current complete Stop
and prompt paths still miss the 100 ms requirement. Earlier measurements show
substantial interpreter and import costs before dispatch. Test that boundary
with an experimental compiled client of the unchanged worker protocol.

Use the verified installed product `3d1866d5` and its Python 3.12.11 validation
environment for the Python arm. Verify all installed package bytes against
wheel SHA-256
`253ca806dba5aa817bd7ddcba1af123535b29853aa211fee61a14f71799a62b8`.
Compile the candidate from `benchmarks/native_hook` with Rust 1.95.0 and the
locked dependency graph. Freeze both artifacts, compiler version, experiment
source, helper and protocol hashes before the first measurement.

Use four arms: empty Python command, empty native command, Python dispatch and
native dispatch. The empty commands read `{}` with an empty event argument and
must emit no output. Dispatch commands receive the same synthetic UTF-8 payload
and contact one disposable authenticated loopback responder. It checks the
complete envelope, environment binding, request identity, event, working
directory and exact payload, then returns a fixed context through the original
worker response schema. Both arms must return identical decoded host output.

The responder is a protocol fixture, not the memory worker. It does not capture,
retrieve, start a service or open a memory database. Use a private descriptor
and a copied environment with all original Mnemos settings removed. Preserve
only the experiment descriptor setting. Record environment data as a hash,
never as raw values. Do not alter normal host settings.

Run two warmup rounds and 20 measured rounds. Rotate the four-arm order by round
so each arm occupies every position five times in measured rounds. There are
88 commands, 80 measured. Save every command result and the responder's bound
request metadata; do not selectively rerun, subtract startup time or omit an
error. Complete elapsed time runs from process launch through exit and output
collection. Warmups remain separate. Use nearest-rank p95 and report paired
native-minus-Python changes for empty and dispatch paths.

Verify exactly 44 authenticated dispatches, no fallback, identical full output,
zero empty-arm requests, exited command processes and responder shutdown. Check
source, executable and installed-package hashes after the run. Independently
recompute summaries and pairing from raw records without executing inference.
Preserve the first build's integer precision failure and its correction as
development evidence, without folding test time into benchmark timing.

This probe only determines whether eliminating Python from the client boundary
is worth a complete installed comparison. It cannot satisfy the original 100 ms
hook budget or 500 ms recall budget. The next gate uses the same accepted
10,000-fact seed, real embeddings, original worker checks, durable capture,
restart verification and complete context in both arms. Native-host acceptance,
fresh quality, canonical delivery and actual private pilot use remain open.
