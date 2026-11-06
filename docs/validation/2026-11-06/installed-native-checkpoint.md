# Installed and native explicit-memory checkpoint

The wheel from product `24887ef3` passed its installed smoke check. The native
Claude lifecycle attempt is failed and incomplete: teaching passed, then the
first recalled-memory session failed authentication before inference. No
native correction, forgetting or source replay was attempted in this run.

## Installed package

Both wheel and source archive built successfully and passed Twine metadata
checks. The wheel SHA-256 is
`bdc64792d918293ca1e01c482531c518e02c1aca9c5ccf6603e035a067ac8273`.
It remains the unpublished `0.7.0b1` development candidate.

The [installed receipt](installed-smoke-24887ef3.json) records an environment
outside the source checkout, matching package/runtime versions, packaged
resources and CLI entrypoints, MCP 1.30.0 and FastEmbed 0.9.1. The example used
real BGE small English v1.5 embeddings, completed an explicit remember job,
restarted the service process, recalled the same fact through CLI and MCP,
inspected retained evidence, excluded a foreign project and stopped its service.
Loading the Hermes entrypoint was an installation check, not native acceptance.

## Native attempt v1

Runner `c3a14fd3` bound the installed package to that wheel and source, copied
the committed plugin into the disposable project, and retained its input
hashes before inference. Claude Code 2.1.191 used the observed model
`claude-sonnet-4-6`. All cognition modules were disabled. The service observer
recorded arrivals without replacing the installed dispatcher or scheduler.

The teaching session made three correlated calls: store, maintenance and job
status. The write committed and its actual processing job completed. Independent
SQLite inspection found the asserted fact and unchanged retained/foreign
controls. All three observed teaching hooks returned successful exit statuses;
complete capture coverage was not evaluated.

After a clean service restart, a new Claude session returned
`authentication_failed`, exit code 1, zero inference output tokens and no memory
tool calls. The controller stopped the attempt. The native CLI reported about
$0.052 for teaching and zero for the rejected session; those are host-reported
usage figures, not an independent billing statement.

Cleanup passed: the owned service stopped, its descriptor disappeared, copied
authentication was removed, and no process referring to this attempt remained.
The original report has an overall failed status but leaves the last phase
labeled running. Its terminal native process record establishes failure; the
[derived receipt](native-explicit-v1-receipt.json) records that interpretation
without modifying the original evidence. The next runner fixes that report field.

Raw streams, arrivals, database state and the exact executed source bytes remain
in a private local archive. The committed receipt contains their hashes. The
original failed run is not resumed or pooled with a later attempt.

## Follow-up

Read-only inspection found a November 1 access-token expiry in the source
credential file. The normal CLI's cached auth-status command reported logged
in, which does not prove request authentication. The run did not establish
whether credential rotation caused the second-session failure. Normal
credentials were not changed or refreshed by the controller.

The [authentication amendment](../../plans/2026-11-06-native-explicit-auth-v2.md)
requires a separate validation login and at least one hour of access-token
lifetime before the next attempt. Interactive sign-in is pending. No second
inference attempt has run.

The original runner and reused acceptance checks passed 77 offline tests.
After the authentication guard, 38 focused tests and strict mypy passed.
These tests validate the harness; they do not complete the native lifecycle.
The two-week/two-project daily-use pilot and publication remain open.
