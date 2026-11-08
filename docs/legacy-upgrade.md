# Reviewing the 0.6.1rc1 legacy upgrade

This candidate is unpublished. It includes the PR #7 base plus graph savepoints,
content-recency tracking, explicit Windows plugin Python resolution and array
typing repairs. It is separate from the canonical 0.7 development line and its
inactive pilot. Do not substitute one line's artifact or host configuration for
the other.

## Database boundary

Opening a legacy database with this candidate migrates it to schema 3. A nullable
content_updated_at column records known rewrites. Old activity timestamps are
not copied into that field, because reads also changed them. Facts with unknown
revision history retain null. Creation time is a ranking fallback, not evidence
that a contradictory assertion is current.

Older clients deserialize rows positionally. They cannot safely share a migrated
database. Stop every client that can open the database before a later approved
upgrade. A package reinstall alone does not reverse the schema change.

## Rehearsal before any live change

1. Identify the actual configured database and every client using it. Stop the
   clients and retain their configuration and exact package versions.
2. Take a consistent SQLite backup, including committed WAL contents. Do not
   copy only the main file while writers are running. Preserve the original
   backup unchanged and use a second copy for rehearsal.
3. Install the reviewed wheel by its exact file and SHA-256 in a separate
   environment. Confirm its runtime version is 0.6.1rc1. Do not replace a normal
   host environment or activate capture hooks during the rehearsal.
4. Point only the isolated candidate at the rehearsal copy. Verify row counts,
   text, vectors, source metadata, graph edges, feedback and known/unknown revision
   times, then run SQLite integrity and foreign-key checks. Verify recall and
   another write after closing and reopening the candidate.
5. Retain the results and approve the concrete live upgrade separately. Local
   source tests and a rehearsal do not establish native-host or pilot acceptance.

## Rollback boundary

Before any new writes, the untouched original database can be paired with the
original stopped clients. Keep the migrated copy separate; do not open it with
an older client. Once the candidate has accepted new writes, corrections or
forget requests, restoring an old backup would lose those changes and could
resurrect forgotten material. Stop and reconcile those changes before considering
rollback. There is no automatic downgrade or replay guarantee in this candidate.

The original legacy benchmark failures remain open. Controlled chronology
results concern already-known questions and do not qualify general answer
support, abstention, complete-command latency or a daily-use pilot.
