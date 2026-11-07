# SQL ranking probe: vector serialization repair

Retain the [V1 protocol](2026-11-07-sql-ranking-probe-v1.md), SQL, data, order and
sample counts. V1 stopped before its first query because the harness called an
instance method as a static function. Its zero-observation failure is preserved.

V2 uses `sqlite_vec.serialize_float32`, the same serializer called by the installed
store method after its extension-enabled check. The read-only probe has already
loaded that extension. Use a new output directory and retain V1's files and hashes.
This changes only harness serialization, not product code or ranking behavior.
