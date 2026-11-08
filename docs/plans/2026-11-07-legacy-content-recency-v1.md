# Separate content recency from legacy access time

The completed legacy diagnosis shows that reading a fact can make it rank as
the newest content. Repair that mechanism on a separate branch from release
candidate `9ea3852a` and frozen product `f5a944ea`. Keep both candidates, their
artifacts, the original contradiction fixture and all recorded scores unchanged.
No model run or production benchmark rerun belongs to this increment.

Add an optional `content_updated_at` field to the legacy `MemoryChunk`. A known
content rewrite records its own time; access does not change it. Keep the old
`updated_at` and access counter behavior for compatibility with activity-based
retention. Recency ranking uses the explicit content time, or the original
creation time when no content revision is known. That fallback is a ranking
heuristic, not proof of which contradictory assertion is current. Unknown
revision history remains null, including legacy rows that have been rewritten.
Never infer a content revision from the old access-influenced `updated_at`.

Persist the optional field in SQLite schema version 3. Migrate under a write
transaction so concurrent clients cannot race the column addition, and retain
existing rows and timestamps without fabricating a backfill. A failed migration
must roll back. Only disposable databases may be used. Schema 3 is not claimed
compatible with simultaneous older clients that deserialize `SELECT *` rows.

Before implementation, record regressions on the unchanged source for:

- Prior access to stale facts must not reverse content-recency order in either
  store, including the current top-one and multi-result paths.
- Explicit same-ID revision time must take precedence over original creation;
  touching that revision must preserve its time and content.
- Equal content times must not acquire a false temporal ordering from access.
- Missing revision history must remain visible as unknown through serialization,
  migration, restart and subsequent reads.
- New schema creation, old-row preservation, migration rollback, simultaneous
  opening and both stores' persistence of revision time must be covered.

Use fixed synthetic timestamps, explicit scoped fixtures, simple local
embeddings and the mock LLM. Keep reconstruction of old fixture outcomes out
of the regression labels. These cases verify a causal storage/ranking contract;
they are not new semantic-acceptance questions or replacement benchmark scores.

After the focused regressions pass, run the legacy source suite, strict typing
and formatting on the exact repair commit. Preserve the initial failing tests
and bind source, checks and the isolated migration evidence in a report. Do not
merge this repair into the three-file storage backport, the 0.6.1 candidate or
frozen f5. A later benchmark needs a separate, prospective chronology protocol
with explicit source ordering and prior-access cases. Full quality, timing,
native configuration, publication and actual pilot requirements stay open.
