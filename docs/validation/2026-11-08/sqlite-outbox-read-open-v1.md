# Existing outbox inspection avoids schema writes

A fresh SQLiteOutbox instance previously started BEGIN IMMEDIATE and rewrote user_version even after validating an existing schema. If another connection held a write transaction, pending() could therefore fail with database is locked despite WAL readers being able to inspect the committed backlog.

The new regression commits one queued event, closes the original connection, then starts a separate uncommitted deletion. Before the patch, a newly opened reader fails at the schema write lock. After the patch, it sees the committed event while the deletion is uncommitted, then sees an empty queue after the writer commits. This verifies both concurrent inspection and isolation from uncommitted changes.

The isolated capture candidate now creates the schema only when no tables exist. It still validates existing table/version structure, rejects foreign objects, checks WAL mode and FULL synchronization, and protects private paths. Append and acknowledgment transactions retain their FULL synchronous commits. No acknowledgment durability setting changed.

All 22 selected SQLite outbox, worker and real-service lifecycle tests passed in 18.38 seconds. One Windows test skipped because the operating system prevents renaming an open SQLite database. The retained tests include cross-process writes and abrupt-exit WAL recovery, duplicate replay, correction and forgetting. JUnit is retained at `.venv/outbox-read-green-v1.xml` with its hash in the adjacent receipt.

This is source validation of a concurrency repair, not an installed artifact or a measured latency improvement. The previously validated v5 package, canonical runtime and live optional pilot are unchanged. A new build is required before testing installed behavior. Original quality, timing and daily-use gates remain open.
