# Content recency in the legacy library

This development repair separates content revision from access in the legacy
`MnemosEngine` path. It is separate from the prepared storage release and the
frozen functional-memory candidate. It has not been published or applied to
live databases.

`MemoryChunk.updated_at` retains its historical activity behavior: a read may
update it. Access counts and activity-based retention keep that behavior.
`content_updated_at` records a known content rewrite and is unaffected by
`MemoryChunk.touch`, store access, or retrieval. `reconsolidate(new_content)`
sets the new content time, preserves the ID and increments the version. As
before, the returned content needs a new embedding before vector retrieval.
Callers that rewrite content directly must supply the corresponding content
revision time, or leave it null when that time is unknown.

Recency ranking uses `content_recency_at`: the explicit content revision if
known, otherwise the original creation time. This prevents a read from making
an old fact appear newly written. It does not establish the effective date of
an assertion, resolve contradictory facts with tied dates, or reconstruct
edits made before the new field existed. Creation is a fallback ranking signal,
not evidence that an assertion is current. Unknown revision history remains
null in serialized and stored records, even when the version exceeds one.

SQLite schema 3 adds a nullable `content_updated_at` column. Schema setup uses a
write transaction so existing clients opening the same legacy database cannot
race the migration. Existing row values are preserved; the old activity time
is never copied into the new field. A failed migration rolls back and closes
its connection. Tests cover simultaneous opening of an existing WAL database,
which matches the earlier release's database configuration. Concurrent first
creation and WAL conversion remain outside that guarantee.

Older clients deserialize all columns positionally and cannot share a schema 3
database safely. Any later deployment must stop older clients and retain the
original database backup. This repair has only been exercised on disposable
databases and copies. It does not make the existing failed benchmark pass,
and the original fixture and results remain unchanged. A later benchmark must
declare chronology and prior-access cases before measurement.
