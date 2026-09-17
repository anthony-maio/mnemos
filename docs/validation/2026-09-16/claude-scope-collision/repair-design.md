# Claude scope identity repair design

This is a design note from the offline collision probe. It does not change
the adapter, service, checkpoints, or existing runtime data.

## Evidence

The probe used `profile-a`, one Claude session ID, two enabled projects, and
the same explicit native `prompt_id` (`native-event-1`). Both adapter event
IDs were identical. Project A committed one event and job. Project B raised
`ValueError: Source event identity already has a different payload`. Durable
state remained one event and one job, both scoped to project A. There was no
duplicate record or cross-project read leak. The probe made zero model calls.

Probe source SHA256: `7269f44302742048968f7ca263744763c92050ec96c5c1f72a64629c2159c7ac`.
The final report has `explicit_prompt_id: true` and is retained at
`.venv/claude-scope-collision-report.json`.

## Repair boundary

Introduce version 2 identities. The canonical scope key is the normalized
tuple `(user_profile_id, project_id, workspace_id)`. The event identity input
must contain that scope key, host, session ID, native event ID or transcript
source reference, role, and normalized captured content. Keep the existing
`claude:` prefix and hash encoding so the value remains safe for the service
firewall.

The service idempotency key should continue to include profile, host, session,
and the adapter event ID. Once the adapter event ID is scope-bound, this key is
isolated across projects. Do not solve the collision by allowing an existing
unscoped key to accept a different context; that would make the first
project's event authoritative for the second project.

## Old event deduplication

Old captures have unscoped adapter IDs. A v2 capture should carry a bounded,
internal `legacy_event_id` alias only when the adapter can derive the exact
old identity from the same session, source reference, role, and content. The
service can use the alias for a read-only existing-event lookup:

1. If the old key exists with the same resolved profile, project, workspace,
   role, and content, return the existing durable result and do not create a
   v2 duplicate.
2. If the old key exists with a different scope or payload, reject the alias
   and retain the new scoped event as a separate candidate only after the
   normal scope checks. Never return the old event as the new project's
   result.
3. If the old key does not exist, accept the v2 event normally.

The alias is migration metadata, not a replacement event ID. It must not be
stored in user-visible content or used to bypass suppression or scope checks.
An absence of recent legacy traffic does not make old replays impossible.
Keep the deterministic legacy suppression check unless an explicit migration
also preserves those tombstones under every supported future identity. This
branch is unreleased; deployment and any migration of live data remain outside
the current authorization.

Forgotten-source suppression needs a source-key check before either alias
acceptance or v2 event insertion. The current `memory_suppressions` table has
separate `canonical_key` and `source_event_id` columns. `forget_fact` writes a
source suppression row before deleting the episodic event, its evidence, and
the fact, so that source tombstone remains available even after the old source
record has been erased. The stored source key is the service identity hash,
not the adapter's raw event ID. A migration path must therefore derive the
old service key from the old identity tuple `(profile, host, session, old
adapter event ID)` and query `is_suppressed(source_event_id=old_service_key)`
before accepting a legacy alias or creating a v2 event.
If that lookup finds an explicit-forget tombstone, return a suppressed,
durable result and create no v2 record, even when the old event and fact are
gone.

The old source key does not include project or workspace, and a tombstone may
outlive the metadata that could establish that scope. Such a match must remain
conservatively suppressed or explicitly degraded. It cannot be reassigned to
another project by guessing. The resulting possible false suppression needs
a documented recovery path; it is preferable to resurrecting forgotten data.

Canonical suppression is supplementary. It can suppress a new event only when
its normalized captured content produces the same canonical key. A replayed
conversation can differ from the extracted fact text, so canonical matching
cannot protect that replay by itself. If the old source tombstone is absent,
the service may perform the bounded legacy-event lookup described above. It
must require an exact resolved scope, role, and payload match for an existing
old event; a caller-supplied alias must never be trusted to select an
arbitrary tombstone. A scope or payload mismatch rejects the alias and then
uses the normal scoped v2 path as a separate candidate. Source tombstones
created by explicit forget should be retained independently of source-event
retention and should not be deleted during alias migration.

## Checkpoint isolation and migration

The v1 checkpoint filename hashes only `session_id`, so it cannot safely be
shared by two projects. V2 checkpoint paths should hash a versioned tuple of
`user_profile_id`, `project_id`, `workspace_id`, and `session_id`. The JSON
checkpoint should also include `identity_version: 2` and the same scope key;
the read path must verify both before applying offset, transcript identity, or
pending dedup records.

Read the v2 path first. A v1 checkpoint with no scope claim is ambiguous and
must not be silently copied into a project-specific path. The safe choices are
an explicit migration command that binds the file to one project after user
review, or a one-time migration manifest containing the old path, scope key,
and transcript identity. The adapter should return a degraded result and keep
the v1 file unchanged when no binding is present. This can cause replayed
captures after an operator-approved fresh start, but it prevents silently
skipping one project's transcript or sharing offsets across projects.

Once bound, copy the checkpoint atomically to the v2 path, record the scope
key and identity version, and retain a tombstone for the old path. A second
project must see the tombstone and fail closed rather than claim the same v1
file. Legacy string entries in `pending_dedup` remain readable; new entries
are records containing fingerprint and source reference. Checkpoint writes
must continue to occur only after committed or queued capture acknowledgement.

## Native event IDs and missing IDs

When Claude provides a native event ID, preserve it in the v2 identity. When
it does not, a content hash is not an event identity: two independent,
identical prompts can otherwise merge. Prefer a native transcript UUID or
prompt reference when available. If no stable native reference exists, the
adapter should report a degraded identity status and require an explicit
host-provided event nonce or operator-approved per-session sequence. A local
counter alone is unsuitable for replay because it changes after restart.

## Error and rollout rules

Capture rejection or transport failure must produce a degraded, uncommitted
result and must not advance a transcript offset. Malformed checkpoint JSON,
scope claims, or offsets should fail closed; only the documented v1 string
pending format is accepted as legacy input. Do not log event content, token
values, transcript paths, or checkpoint contents in migration diagnostics.

The implementation should add tests for same native event ID across projects,
same ID across profiles, old exact replay, old cross-scope collision, bound
and unbound v1 checkpoints, v2 path isolation, forgotten canonical replay,
missing native event IDs, and rejected capture without offset advancement.
