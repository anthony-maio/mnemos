# Hermes compression revisions through the installed adapter and service

Six capture phases use the actual isolated SessionDB and ContextCompressor
carrier merge with the current installed Mnemos provider: original capture,
child compression, live-payload edit, unchanged edit replay, database reopen,
and grandchild recompression. Exactly three unique outbox records remain.
The original source keeps its identity and content. The edit gets a separate
source identity in the child session and remains stable on replay and recopy.
The summary retains provisional-unassociated classification and its identity.

The exact outbox then passes through the authenticated loopback service into
private SQLite. There are three episodes and three jobs: two extract jobs and
one capture_only summary job. The summary episode is deferred. Service outage
keeps all three queued; restart acknowledges them without duplicate episodes
or jobs. The service descriptors are removed, and the source outbox is intact.

All 55 installed Mnemos Python modules match the 05c71d2 wheel. The host patch
and source-snapshot helper hashes are recorded by the capture probe. Both
SQLite backups pass integrity and foreign-key checks.

This combines actual compression/edit projection, adapter capture, and raw
service delivery. It is not a full interactive CLI run. All five cognition
modules are disabled, the model provider is unavailable, no background
maintenance runs, and the configured embedder is SimpleEmbeddingProvider.
No extracted-fact correction, semantic quality, forgetting, or module benefit
is claimed by these three queued source records. The host remains an isolated
candidate with the previously documented broader validation gates open.

Distinct hybrid edits retain separate inactive source snapshots as history.
This preserves prior source references; long-term retention bounds and storage
growth for those snapshots have not been validated by this bounded check.
