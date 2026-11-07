# Scoped index probe: checkpoint preparation

Retain the [V1 protocol](2026-11-07-scoped-index-v1.md), index, SQL, data and sample
counts. V1 committed index creation but left its setup connection open. The index
remained in WAL, so the immutable reader could not see it. The attempt retained
one baseline observation and no indexed observation. Process exit then closed
that connection and checkpointed the index; its prepared-file hash consequently
differs from the earlier freeze. The original seed stayed unchanged.

V2 explicitly checkpoints and closes the setup connection before opening either
read-only reader or recording database hashes. Use a new directory and preserve
V1 as a preparation failure. Product code and query semantics remain unchanged.
