# Complete installed native launcher comparison

Freeze this protocol, both command paths and all inputs before measurement.
The earlier loopback probe reduces median command-boundary time, but it does
not exercise capture, retrieval or project resolution. This comparison measures
the complete installed paths with those operations intact.

Both arms use product `3d1866d5`, the same Python 3.12.11 validation environment,
96 verified package files, the same 57 dependency versions and 17 unchanged
local BGE assets. Wheel SHA-256 is
`253ca806dba5aa817bd7ddcba1af123535b29853aa211fee61a14f71799a62b8`.
Baseline runs the original resident Python entry with reviewed environment
setup and guarded fallback. Candidate runs the experimental native executable
with a hash-pinned profile derived from that exact environment setup. Its
fallback invokes the same reviewed launcher; it cannot bypass the guard.

Retain the 131 passing subprocess/related checks, two Rust deadline checks and
the 24-command private policy comparison. The latter uses the exact original
launcher and settings parser. Both paths agree on valid settings, cleared
inherited selectors, explicit global/project LLM settings and malformed
global/project configuration. These are preparation checks, not latency samples
or native-host acceptance.

Use the original pristine accepted-fact seed, SHA-256
`7d362132eb196e099ac63e2578602c80196ac0b07c486a2682b911d6d23de5ac`,
with 10,000 accepted facts, source evidence and vectors. Each block starts with
its own copy, unchanged accepted projection, declared scope and input manifest.
No candidate prefiltering, context reduction, query change, embedding substitute,
disabled maintenance or removed worker check is permitted. Preserve the current
Git identity resolver and current registry/settings checks on every hook event.

Run baseline-0, candidate-0, candidate-1, baseline-1, with only one owned
service/worker pair active at a time. Each block retains the original three
recall warmups, 25 timed recalls, one warmup and 25 timed commands for each of
Stop and UserPromptSubmit. This gives 320 observations, 300 timed, and 208
captured source/job records across the four independent stores. Keep the
original query and event indexes so each arm has 50 timed pairs per path.

Both worker wrappers record the same bounded dispatch metadata and output hash.
Each hook must match exactly one handled worker dispatch with the expected
host, event, environment and payload hash. This checks the native path without
adding a Python bootstrap to its measured command. Both wrappers include the
same receipt-write work inside worker dispatch. Preserve full host output and
compare decoded responses and context exactly across arms. No successful
observation may use fallback or suppress capture.

The complete hook timer includes command launch, native profile or Python
environment preparation, transport, worker checks, outbox durability, service
operations and command exit. Do not subtract startup or component timing.
Retain the prior comparison's interpolated p95 convention and unchanged
requirements: 100 ms complete hooks and 500 ms recall. Report failures,
maxima, matched differences and block-level variation as well as medians.

Run normal maintenance, stop the owned processes, restart the service and verify
every capture remains durable. Check completion status separately. Verify the
accepted facts, sources and vectors remain unchanged, SQLite integrity and
foreign keys pass, and model/source/artifact bindings remain identical. Stop
only processes whose recorded identities still match. Poll the original handle
until terminal; an observation timeout never justifies restarting the run.

Independently audit all raw observations, worker dispatches, responses, database
records, pairing and cleanup. Preserve any failed attempt without replacing or
relabeling it. This is a synthetic installed comparison, not fresh semantic
quality evaluation, native Codex/Claude acceptance or actual pilot use. The
inactive private pilot stays at zero days. No live host or publication changes
are authorized by this protocol.
