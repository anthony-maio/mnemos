# Normal hook review prepared; user review is pending

The frozen `f5a944ea` wheel now has a disposable Codex profile containing the
six current pilot hook definitions. Three metadata-only native processes
completed and exited. Their hook hashes agree, and all six hooks remain
untrusted. No credentials were copied, service started or model turn requested.
Thirteen original file paths stayed unchanged. The private pilot remains
inactive at zero days.

The first inventory exposed a parser defect: Codex reports camelCase event
names, while its configuration uses PascalCase. That failed attempt is retained.
The repaired validator accepts the native names and rejects altered commands,
timeouts, sources, hashes, missing or additional hooks and untrusted definitions
when trust is required. The focused suite has 39 passing checks. Two source
files pass strict typing and three pass formatting. An initial typing command
needed explicit package bases; its diagnostic is retained.

The user-facing launcher passed its check-only run using the installed Python,
wheel and pinned Codex executable. The private review instructions contain the
exact command, event settings and isolated data paths. Interactive review must
still happen through the normal `/hooks` UI. Afterward, the launcher checks the
same hashes, observed trust state, copied-login removal, service cleanup and
unchanged original files. Any recorded model turn requires separate inspection.

The [receipt](native-codex-hook-review-v1-receipt.json) binds 47 private archive
members, SHA-256
`b3c0c78d659077a94e4e991f02386f97a35a210281d51439f7f814cfe2a70724`.
An earlier archive verification tripped over Windows path separators; that
archive was retained and the corrected archive was read back successfully.
This is preparation evidence. It does not establish persisted trust, hook
execution, capture, quality, timing or pilot readiness. The next steps follow
the [review protocol](../../plans/2026-11-07-native-codex-hook-review-v1.md).
