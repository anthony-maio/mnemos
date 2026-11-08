# Installed outbox inspection repair

The fresh v6 capture-candidate installation passed all 22 selected checks, with one expected Windows skip for renaming an open SQLite database. The suite covers the repaired concurrent reader, SQLite durability and cross-process behavior, worker backend selection and compatibility replay, and real-service correction/forgetting without resurrection. Execution took 28.70 seconds. This is focused installed regression validation, not the full repository suite or a timing benchmark.

The regression opens a new reader while another connection holds an uncommitted deletion. It sees the committed queue entry before the writer commits and an empty queue afterward. Previously, connection initialization tried to acquire a schema write lock and failed instead. Existing-schema inspection now avoids that write while retaining schema validation, private-path protection, WAL checks and FULL synchronous append/acknowledgment transactions.

Built from 5b48e0387923ce0a84ac790a5696aa05b2af075b using hatchling 1.33.0. All 99 wheel source/resource members and 106 sdist members matched source. The fresh Python 3.12.11 environment retained the prior dependency versions, and all installed files matched after tests. Six copied Python test/fixture files matched source hashes. The runner used the artifact directory as its working directory to keep subprocess imports on the installed package.

Only mnemos/sqlite_outbox.py differs among the 99 package files compared with the previously validated v5 wheel; the other 98 are identical. The prior wheel and sdist hashes still match their frozen receipt. That comparison supports the focused verification scope without treating v5's broader tests as a new v6 full-suite result.

Wheel SHA256: fea10802ee2c5b43d7ea63b78a1afab1d99cd02cfc5b6d10cbb3dd30f36edfea.

Sdist SHA256: df9d9e7ade81537c21c7e46ced95d8affa4b94baa30b2926882c7485cb9b1f80.

The verified 21-member evidence archive retains artifacts, constraints, test copies, JUnit, source/installed parity, the prior-package comparison and helpers. SHA256: 7dde70ee82fb21464239d5998e57ec4db4f741789e6d2c23c2c870a5f25f108f. The adjacent JSON records exact counts and hashes.

No latency improvement is claimed. This removes unnecessary schema-write contention on connection opening; it does not remove identity-resolution or durable-append costs from every warm capture. The 100 ms capture requirement remains open. Canonical runtime, live pilot configuration, databases and publication state are unchanged.
