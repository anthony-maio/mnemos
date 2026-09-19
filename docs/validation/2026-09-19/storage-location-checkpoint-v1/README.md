# Explicit storage location checkpoint

Source `7bcabde63ba02947ad8f240a1adec9278c275b01` adds a fresh explicit work
directory to the performance harness, records database/outbox locations and
filesystem device IDs, retains explicit fixtures, and refuses to overwrite an
existing work directory or report. Defaults and acceptance thresholds remain
unchanged. Eight new cases cover path selection, cleanup, failure retention,
CLI forwarding, and evidence protection.

The clean snapshot passes 1,131 source tests with four skips, Black, strict
mypy, JavaScript syntax, package build, and Twine. All six checks and their
commands are retained under `source-checks`. The website hash is unchanged.

Runtime/build Git objects for `mnemos`, `agents`, and `pyproject.toml` match
`39b9c79`. The newly built wheel is byte-identical to that checkpoint's wheel
(SHA256 `00428a439b2e29d756da54b2d092e59c99c589653bd872bf3d5476c1fe80ba59`).
The previous 428 installed contract passes, two skips, and seven replay
assertions are reused through that exact-byte binding; they were not rerun.
The binding references their archived reports and verifies their hashes.
The source distribution includes changed benchmark/documentation content and
is not claimed byte-identical. Build artifacts remain at their recorded paths.

The [capture placement diagnostic](../capture-placement-v1/README.md) records
three passing SSD repetitions and three failing HDD repetitions with the
unchanged 100 ms gate. Recall latency, native-host capture, real-model quality,
and operational acceptance remain open. The original failures are preserved.
The active model evaluation and queued comparisons remain on frozen `fc72cc2`.
