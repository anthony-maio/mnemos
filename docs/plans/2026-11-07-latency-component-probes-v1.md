# Git launch and evidence-index component probes

The current-worker diagnosis identifies fresh Git resolution and evidence lookup
as concrete costs. Freeze these separate development probes before measurement.
Neither changes product code, live configuration, frozen scores or latency gates.
Run them sequentially so they do not compete during timing.

## Git executable boundary

Use installed product `3d1866d5` and its original `canonical_project_root`.
Compare the selected `C:/Program Files/Git/cmd/git.exe` with
`C:/Program Files/Git/mingw64/bin/git.exe`; both report 2.51.0.windows.1 and
have valid Authenticode signatures. Pin their bytes. Substitute only the first
command argument at the original subprocess call. Preserve all other arguments,
environment filtering, filesystem checks, timeouts, retry behavior and results.
Do not cache project identity or bypass Git.

Create private offline fixtures: a plain folder, an ordinary repository, its
subdirectory, a linked worktree and a bare repository. Disable fixture hooks,
signing, templates and automatic maintenance. No remote is configured. Each
scenario gets one warmup pair and ten measured pairs. Rotate scenario order
and alternate which executable runs first. Retain all 110 calls, including
warmups, without retrying a failed harness observation.

Record complete resolver time, underlying subprocess time and count, exit status,
resolved identity and errors. Compare full subprocess output and final identity
within every pair. Environment values stay private and are recorded only as a
hash. Separately check inherited Git selectors and a broken Git marker with both
executables. These four control calls are not latency samples. Preserve setup
commands separately from measurement and verify no owned Git process remains.

## Fact-evidence index

Use two fresh copies of the original pristine 10,000-fact seed, SHA-256
`7d362132eb196e099ac63e2578602c80196ac0b07c486a2682b911d6d23de5ac`.
Open both with the same installed `3d1866d5` SQLiteStore. Add only
`fact_evidence(fact_id, created_at)` in the candidate copy. Call the original
`list_fact_evidence` method, including parsing, with identical fact IDs.

Select 100 fact IDs at evenly spaced positions in the sorted accepted IDs,
plus 20 absent IDs. Run ten warmup pairs and three rounds over all 120 IDs,
alternating arm order. Retain 740 observations, including 20 warmup observations,
and all 370 exact-result pairs. Report timings separately for present and absent
IDs, with query plans and index-build time. No query, row, evidence or source
content may be omitted from result equality checks.

Use a separate small fixture to compare full evidence rows before and after
index creation, including equal timestamps inserted in an order different from
identifier order, multiple facts, absent facts and an insert after index creation.
Do not modify the original seed. Verify both performance copies retain the exact
accepted projection, including evidence and vectors, and pass integrity and
foreign-key checks after closing.

## Interpretation

Independently recompute ordering, complete outputs, timings, pair counts and
file bindings. Preserve any failure before revising a harness. Keep raw local
evidence in a verified private archive. These are component probes on known
development inputs, without model calls, service startup or native-host use.
A useful component result still needs a reviewed implementation, relevant
regression checks and a complete installed comparison. The 100 ms complete-hook
gate and inactive zero-day pilot remain unchanged.
