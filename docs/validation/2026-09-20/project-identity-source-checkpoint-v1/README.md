# Project identity source/package checkpoint

Frozen checkpoint cc940ac passes 2,034 source tests with four skips, formatting,
Python 3.10 compatibility type checking, JavaScript syntax, wheel/sdist build
and twine metadata checks. All six checks have exit code zero. Source, runner,
external inputs, individual JUnit outcomes and package hashes are preserved.

Git lookup retries once and does not silently assign a folder identity to a
repository after discovery failure. Inherited Git repository path overrides
cannot select another project's identity. Tests cover ordinary directories,
worktrees, unrelated clones, bare repositories, malformed output, persistent
failure, recovery after transient failure and a dangling Git marker.

Installed verification is separate. The earlier bee8357 installed MCP failure
is preserved. This checkpoint does not close real-model extraction quality,
native-host recovery, full-module benefit or the daily-use pilot.
