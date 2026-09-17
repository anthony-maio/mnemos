# Corrected 10,000-fact performance measurement

This is an immutable copy of the completed performance phase from the live
v4 evaluation queue. Source d94ded6, model configuration, timings, and code
provenance are recorded in the raw report. Every copied artifact was checked
against the completed phase hashes before copying. The queue continues in
its original location.

Warm recall p95 is 1,088.094 ms and fails the unchanged 500 ms gate. Durable
capture p95 is 64.279 ms and passes the 100 ms gate. There were no duplicate
follow-up diagnostic writes between timed queries. SQLite remained WAL/FULL.
Metadata-update p95 is 795.750 ms; this identifies a slow stage, not its cause.
The fixture is synthetic with real query embeddings, not an extraction-quality
or native-host acceptance measurement.
