# Complete native comparison after Windows environment repair

Freeze this protocol, candidate source, executable and drivers before running
the comparison. The v1 run remains failed: all 104 native hooks missed the
resident worker. A separate regression reproduces Python's omission of Windows
hidden drive-directory entries and the native client's conflicting fingerprint.
The candidate excludes these entries on Windows while retaining ordinary
environment checks. It does not change product Python code or worker behavior.

The candidate passes 137 subprocess/related checks, two Rust deadline tests,
strict Clippy, Rustfmt, Black and 12 documentation checks. Its exact private
guard passes 12 paired cases, or 24 commands, without starting a service or
model. Executable SHA-256 is
`ec52c79a0ae613ae336836b3e82ddf853a601a0de3f7cd3aa0047318d06ee12b`.

Repeat the full
[v1 protocol](2026-11-07-native-launcher-performance-v1.md) in a new output
directory, retaining v1 without edits. Keep its original pristine seed, installed
`3d1866d5` wheel, 96 package files, 57 dependencies, 17 BGE assets, input manifest,
four-block order, per-block warmups and measurements, full output and worker
receipt requirements. This is a development regression comparison on existing
inputs, not fresh semantic-quality evaluation.

The parent launches commands with implicit environment inheritance, preserving
the condition that exposed the bug. Do not normalize the benchmark's child
environment to hide the mismatch. Both command modes and both hosts must pass
the new inherited-drive regression, and ordinary environment mismatches must
still refuse dispatch. Preserve the original four failures and two passing
controls. Rerun the exact private guard parity checks with the new executable
before measurement.

Keep the complete command timers and interpolated p95, 100 ms complete-hook and
500 ms recall gates. Every successful hook must have exactly one handled,
nondegraded resident-worker receipt and durable capture. Verify all 208 captures
after restart, completed jobs, unchanged accepted facts/evidence/vectors,
database integrity, source/model/artifact bindings and owned-process cleanup.
No retry, omitted observation, relaxed threshold or fallback may count as a
successful native path.

Retain failures as failures in the independent audit. Report timing differences
only alongside route acceptance and output equality. The pilot remains inactive
at zero days. Native-host acceptance, live configuration, release and publication
remain outside this synthetic installed comparison.
