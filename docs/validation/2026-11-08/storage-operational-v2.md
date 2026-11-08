# Storage operational checks and failure logging

The installed private 0.6.1 wheel at source `37e942f8` reports doctor status,
legacy unscoped counts, MCP health, startup and retrieval latency. A mocked HTTP
400 also produces a provider-failure event. Mock-provider status is correctly
degraded; this does not establish a live provider connection or native-host acceptance.
Wheel SHA256: `2568133cfa5e2d68d54bedc94f3f0764bfc7a6d2c3892ae112da4c0af628f79c`.
Installed package bytes matched the wheel before and after the completed probe.
All databases and home/configuration files were disposable synthetic fixtures.

The first probe failed because it requested project and global memories, then
incorrectly required every result to belong to the project. The second probe
requests project-only retrieval and saves the result. The first failure remains
preserved; it is a harness error, not a changed product scope policy.

The completed probe reproduced a missing diagnostic: an injected graph insertion
failure propagated IntegrityError and left no open transaction, but emitted zero
structured storage-failure events. Source inspection confirmed the missing log.

Release candidate `1319e1c62ca21d46f5f38e57dd485f39c0d6f46f` adds
`mnemos.storage_failure` after successful graph savepoint rollback and release.
Fields contain backend, operation, exception type and completed replacement
rollback status. They exclude content, identifiers, database paths and exception
messages. Savepoint creation and rollback/connection failures are outside this
event's coverage. It is not a general claim that every storage failure is logged.

Two exact-field regressions failed before the change. Thirteen graph, FTS and
access-update tests then passed; the initial full suite passed 327 tests.
Review identified that a logging handler could mask the original storage error.
The final change catches ordinary logging exceptions and preserves the original
error, with two additional standalone/caller-transaction regressions. Final
source suite: 329 passed in 27.83 seconds. Black accepts both changed Python
files; mypy accepts the changed storage module; git diff --check passes.
Review of the revised diff found no remaining scoped findings; the reviewer did
not run tests. These are local source results, not hosted CI or installed results
for the new code. Test environment: the improvements worktree's Python environment.

Private evidence: `.venv/storage-operational-evidence-v1.zip`, 27 evidence files
plus a SHA256 manifest. Archive SHA256:
`2c3ec46ac8a655d081d3b5abd591afb8446dfd38811520c585c2224594e7241b`.
Every member listed in the manifest was read back and verified. It includes both
probe helpers, probe outputs and red/intermediate/final test logs/XML. Formatting,
typing and review results were observed separately and are not raw archive logs.

The source commit is pushed to `wt/storage-release-merged-20261108`. The prior
private wheel is unchanged and lacks this repair. Next: rebuild from the new
commit, verify package/source parity and installed operational behavior, then
assess the remaining release checklist. No tag, package publication, live memory
change, semantic gate pass or pilot completion is established here.
