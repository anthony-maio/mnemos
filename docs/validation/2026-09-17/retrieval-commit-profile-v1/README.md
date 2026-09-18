# Retrieval commit profile

This archive preserves one completed, source-bound SQLite write-stage profile.
It created a disposable synthetic fixture with 10,000 production-schema rows,
then measured 33 `mark_retrieved_many` calls with five IDs per call. The
production SQLite settings remained WAL with `synchronous=FULL`; no explicit
checkpoint was issued. Each call recorded exactly one access INSERT, one access
UPDATE, and one commit. The five updated access counts were 33 after closing
and reopening the store.

The isolated profile's total p95 was 86.372 ms. Its commit p95 was 86.213 ms,
while access INSERT p95 was 0.134 ms, access UPDATE p95 was 0.042 ms, and
lock-wait p95 was 0.001 ms. This run did not reproduce the approximately 796
ms metadata tail from the original end-to-end report. That observation does
not establish a fix or eliminate environmental, WAL, filesystem, scheduler,
or cross-process causes.

The original end-to-end report remains referenced in `performance-reference.json`
without copying lifecycle or held-out quality artifacts. It recorded 33 warm
queries, a recall p95 of 1088.094 ms, and a failed 500 ms gate. Its p95 row
was query index 12, with 220.545 ms embedding, 63.409 ms vector SQL, 0.416 ms
lexical work, and 796.927 ms metadata update plus refresh. The end-to-end gate
therefore remains failed; this archive is a write-stage diagnostic only.

Source commit: `b752ffbe07e2e32f5ddbf54cc689b8dbbad8c9e0`. The executed profiler helper and original profiler report are copied byte-for-byte; the stage summary and performance reference are derived records. The disposable fixture at
`D:\WindowsTemp\mnemos-profile-retrieval-commit-v1` is intentionally not
archived.
