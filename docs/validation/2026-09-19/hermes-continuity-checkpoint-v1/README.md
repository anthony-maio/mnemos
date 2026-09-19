# Hermes continuity package checkpoint

Verified source: `78c63ed36942848c95c1acd93b805fe0e0d13a67`.
The full source suite passed 1,172 tests with four skips. All six checks passed:
pytest, Black, configured strict mypy, JavaScript syntax, build, and Twine.
The isolated installed wheel passed 484 selected contract cases with two
matching platform skips and seven replay assertions.

Wheel: `mnemos_memory-0.7.0-py3-none-any.whl`, 292,974 bytes.
SHA-256: `4a8d364d69e69c64d43f876ffeb7b9b9b413efc997ad2b4cb8759ca0e15a9625`.

Repeated Hermes callbacks now retain the same occurrence timestamp within
the provider lifetime. Ordinary prefetch reads the service again so an external
correction or forgetting operation can change the next result. The archived
old-wheel regression fails on stale recall; the repaired source passes the
real HTTP service and SQLite correction/forget test. Cross-process recovery
of callback timestamps remains unverified.

The [native v4 run](../native-hermes-local-v4/README.md) passes its 16 bounded
checks using this wheel and an isolated Hermes CLI patch. That establishes
single-turn scoped recall and raw capture. Full native lifecycle acceptance,
automatic mutation mapping, model quality, and the pilot remain open.

`evidence-index.json` binds 35 raw artifacts by byte count and SHA-256.
This README is explanatory and outside that raw index. Earlier failures and
their original source bindings remain in the preceding checkpoint archives.
