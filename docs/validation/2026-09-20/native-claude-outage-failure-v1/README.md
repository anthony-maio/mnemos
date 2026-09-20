# Failed native Claude service-outage attempt

The attempt beginning September 20 at 14:21:44 UTC failed. Native Claude
completed its prompt, but Mnemos started a service despite the experiment's
`MNEMOS_AUTO_START=false` setting. The runner stopped at the assertion that
the runtime descriptor must remain absent. Recovery, replay, and restart
acceptance did not run. This is not evidence of outage recovery.

The frozen runner is `fc5ed255c271c01d96a948064b0e5b1473ae924f`; the installed
product is the earlier `e6887de` wheel. The original [report](report.json)
and [protocol](../../../plans/2026-09-20-native-claude-service-outage-v1.md)
remain unchanged. The report's failed cleanup observation predates the
separate incident response described below.

Two defects combined. `MnemosClient` never read the autostart environment
setting. The experiment isolated its descriptor and injected dispatcher,
but its host-side configuration did not pin the SQLite path. The unexpected
default service therefore opened `C:/Users/Ant/AppData/Roaming/Mnemos/memory.sqlite`.
The experiment violated its intended storage isolation.

Read-only inspection restricted to profile `native-claude-outage-v1` found
three synthetic episodic events, three jobs, and zero facts. The events are
the native user message, its provisional prompt capture, and the assistant
response. These rows are preserved in the [filtered inspection](unexpected-default-store-probe-rows.json).
No pre-attempt snapshot establishes that unrelated database contents were
unchanged. The original injected dispatch trace did not observe this owner's
writes and cannot stand in for the database inspection.

The unexpected owner, PID 49336, was identified by its experiment-specific
descriptor command line and stopped through authenticated direct HTTP.
Its process exited and descriptor disappeared. That shutdown did not use a
client capable of replaying an outbox. A verified SQLite backup was then made
through a read-only source connection; its [metadata](incident-backup.json)
records an integrity result of `ok`. The full backup remains private in the
temporary experiment directory and is deliberately excluded from this archive.
No cleanup deletion or migration has been performed on the default database.

The corrective work must honor the autostart flag in every default client,
independently pin disposable storage/runtime paths, and validate a new installed
package before a separately versioned native attempt. Keep this failure in
the final research packet and article claim notes. It supersedes any blanket
claim that this experiment never touched the default store.

The [index](evidence-index.json) binds eleven copied evidence/source files.
This README and the index itself are outside that list. Native authentication,
full private database contents, and unrelated conversations are not included.
