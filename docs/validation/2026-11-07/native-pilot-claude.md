# Native Claude pilot integration

One Claude Code print-mode session loaded the private pilot's drafted project
MCP and local hook settings on installed product `6d0e5ba6`. It restarted the
disposable store, retrieved the controller's explicit seed, inspected that exact
fact and its source, and returned the requested guide path and marker. The
native startup advertised exactly the two permitted tools. Both calls passed
strict correlation under the actual project MCP namespace.

The run passed the [v2 protocol](../../plans/2026-11-07-native-pilot-claude-v2.md),
but capture identity remains degraded. SessionStart, UserPromptSubmit and Stop
executed and exited successfully. Five raw records persisted under the correct
project, profile and native session: two user records and three assistant
records. The prompt hook lacked a stable occurrence ID, so one user record is
provisional and the later transcript record has its own native UUID. Mnemos
does not merge them based on matching text alone.

The provisional event is deferred with a pending capture-only job. Four normal
transcript records have pending extraction jobs. The explicit controller seed
is the only fact; no automatic facts were accepted. The visible diagnostics
remain `identity_reference_unavailable` and `ambiguous_transcript_association`,
consistent with the earlier documented host limitation. Hook exit code zero
does not mean the adapter reported complete identity coverage.

The real pilot remained inactive. The installed launcher bytes and command
structure were used unchanged, with only the disposable launcher location
substituted in the project settings. No real project settings were applied.
Both source credentials and the persistent pilot files stayed byte-for-byte
unchanged. The copied credential and runtime descriptor were removed, all three
observed native/service PIDs were absent, and SQLite integrity and foreign keys
passed. The session took 17.39 seconds and reported $0.0358131.

The [v2 receipt](native-pilot-claude-v2-receipt.json) binds runner `7c80ba3e`,
Claude Code 2.1.191, `claude-sonnet-4-6`, the installed wheel, and 53 verified
private archive members. Archive SHA-256:
`6d4b7a26794635e7fa05b30bcf9e0c5b5ab4e048029464ed88ba1bef7868d097`.
Raw settings, transcripts, database and identity details remain private.

The [first attempt](native-pilot-claude-v1-receipt.json) failed before any native
submission. Closing the seed MCP client terminated the service process while
leaving its descriptor, so the later shutdown command could not reach it. The
failed report and 41 archive members were preserved; follow-up cleanup verified
the PID absent and port closed before removing that exact stale descriptor.
V2 shuts down the seed owner while the client context is still open and checks
the correct public `chunk_id` inspection argument. These are runner corrections,
not a service-lifecycle repair. The first independent audit also confused
deferred event status with pending job status; its original helper is preserved
and the corrected audit checks both fields and their linkage. No native retry
was needed for that audit correction.

The project-namespace parser has 70 focused passing checks, including 11 new
cases. Formatting and strict type checks pass. These changes postdate the
2,845-test source regression; that earlier full-suite result retains its
original source binding.

Interactive project approval, Codex pilot hooks, rollback application,
current-artifact outage recovery, quality gates and actual daily use remain
open. This session supplies no pilot day, reliability rate, extraction-quality
pass, live activation or publication evidence.
