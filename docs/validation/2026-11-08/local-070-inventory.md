# Existing local 0.7.0 work

The original checkout `D:/Development/mnemos/mnemos` is version 0.7.0 at local
master `5fcc925d`. It contains 11 commits absent from origin/master, followed by
17 modified tracked files and an untracked tests/test_sleep_settings.py. The
committed work includes auto-priming hooks, the retrieve-hook CLI, FastEmbed,
first-run UI and durable-memory repairs. This is distinct from the clean
0.7.0b1 delivery worktree at `D:/Development/mnemos-delivery`, head `54cb50e7`.
The original checkout's HEAD is not an ancestor of the delivery HEAD; version
numbers alone cannot establish that one contains the other.

The uncommitted source was copied over an archive of its exact base into an
isolated audit directory. All 34 tests in test_sleep_settings.py and
test_durable_memory_repair.py pass there. This is targeted validation, not a
full-suite, package, native or release-readiness result. Source hashes were
checked before and after copying. Original files, index and live databases were
not changed. The private snapshot/patch, bindings and JUnit are retained in
`.venv/uncommitted-070-audit-v1`.

Following the user's existing request to save the work remotely, the 17 changed
files and untracked test were committed in a separate worktree. Backup branch
`wt/backup-20261108-local-070` is pushed at
`cc5ecf681c261409aad30c99e4fa52a1c47786dd`. It preserves the original 11 local
commits as history and adds an 18-file checkpoint. It does not merge the work
into master, PR #9 or the delivery branch. Database WAL/SHM files and two
untracked local planning notes were excluded; those remain in the original
checkout. This is a source backup, not a claim that all local files are remote.

## Integration decisions

| Existing work | Current disposition |
| --- | --- |
| Complete SleepConfig loading, environment overrides, runtime propagation and multiline TOML persistence | Relevant and missing from canonical delivery's partial SleepSettings implementation. Port deliberately, preserving its extraction_mode field and existing precedence. Coordinate overlap with open PR #6. |
| Interaction identity and append-to-buffer only after durable deduplication | Relevant and missing from canonical experimental SleepDaemon. Reproduce against canonical before porting; preserve profile partitioning and the write firewall. The local Claude-hook branch still deduplicates by content, so this is not proof of native occurrence identity. |
| MCP 1.x constraint and content-only FTS updates | Overlap with accepted patch work; compare behavior rather than duplicate commits. |
| Fixed chronological benchmark timestamps | Existing benchmark correction overlaps PR #6 and prior chronology work. Keep earlier benchmark scores frozen; any changed benchmark needs its own source/data binding. |
| FastEmbed, auto-priming, first-run UI and durable pipeline commits | Inventory as migration inputs, not an automatically eligible replacement for canonical scoped memory. Native hook trust, provenance, queued completion and quality/default gates still apply. |
| Website-test skip and release prose | Review for honest claims; a skipped external-site check is not website verification. Do not copy old 0.7.0 release claims into the unpublished candidate. |

The next bounded canonical integration should reproduce the two missing sleep
and episode fixes with current profile/firewall behavior, then port only the
needed implementation and tests. Keep enrichment experimental. PR #9 remains
the smaller 0.6.1 affected-user patch; the 0.7 work is part of the broader goal.
