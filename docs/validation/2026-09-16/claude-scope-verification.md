# Scoped capture package verification

Source `20f271d93bbf02990a78b51aa74694deaf5a9bf7` passed 669 tests, with
two explicit WSL integration skips, in 1542.39 seconds. Black, strict mypy,
JavaScript syntax, package build, and Twine checks also passed. The first
attempt ran from 06:35 to 07:02 UTC on September 17.

That attempt overlapped an unintended shared-environment synchronization at
06:47 UTC. The [incident audit](environment-sync-incident/README.md) records
what changed and the limits of the subsequent checks. A second full run on
the same frozen source began at 07:02:43 UTC, without further synchronization.
Its result remains pending. Neither attempt establishes real-model quality.

The first build's wheel is 262591 bytes, SHA256
`0ecaa634fe447621d6366b93accad7abfa2495063d9b425e6043973ea7b86663`.
The source distribution is 2228367 bytes, SHA256
`a389695a8c1a427fa02abb5348192de78550223a64a22234498c1a76f72a5cfd`.

An installed-wheel check from outside the checkout passed scoped Claude
capture for two projects with colliding native identifiers, independent
checkpoints, restart persistence, legacy forget/replay suppression, and
durable queuing of the versioned payload. All 53 tracked package files match
the source after line-ending normalization. Installed files match the wheel
bytes exactly, except the installer-rewritten distribution RECORD. Packaged
resource files are included in that byte check.

This check used simple embeddings and unavailable cognition. It validates the
adapter and local service; it did not run a native Claude conversation or
automatic extraction. It installed into a private disposable target and did
not alter the shared environment or live memory data.

The [evidence index](claude-scope-verification/attempt-v1/evidence-index.json)
binds the logs, report, and exact verification and installed-wheel runners.
