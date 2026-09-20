# Versioned correction collision follow-up

The Hermes revision lifecycle v34 exposed a correction whose new text already
exists as a separate captured-evidence record. Storage's canonical upsert keeps
the existing destination ID. The versioned correction guard rejects that ID
change and rolls the transaction back. This guard predates the follow-up.

The follow-up introduces CorrectionCanonicalCollisionError and the durable
maintenance status correction_canonical_collision. It preserves rollback and
retry behavior. Two regression cases exercise both expected_version and
expected_versions, then reopen SQLite and verify both original rows, versions,
evidence, revisions, source state and pending job state. These are new tests,
not evidence that the complete collision behavior was previously covered.

This diagnostic change does not make the correction succeed. The remaining
repair must preserve the correction target ID while resolving a duplicate
captured-evidence destination atomically. It needs evidence transfer, source
identity preservation, version history, graph invalidation, rollback, and
reviewed forgetting/replay tests. An asserted or unresolved-conflict destination
cannot be treated as unverified captured evidence. Old completed jobs and any
references to the removed captured row also need an explicit lifecycle contract.

The failed installed-wheel run remains unchanged in
[hermes-revision-lifecycle-v34](hermes-revision-lifecycle-v34/README.md).
No live memory or host configuration was changed.

Validation: 29 tests pass across test_conflict_canonical_collision.py,
test_maintenance.py and test_memory_service.py. Black formatted the changed
files; git diff --check passes. Mypy passes for both changed production files
with --python-version 3.13. The default configured 3.10 check is blocked by the
installed NumPy stub's Python 3.12 type syntax; this override does not establish
Python 3.10 compatibility. Full-suite and rebuilt-wheel verification of this
follow-up have not run.
