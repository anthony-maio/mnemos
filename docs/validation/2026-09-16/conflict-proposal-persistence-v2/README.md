# Offline replay of an unsupported conflict proposal

This injected replay uses the recorded requirement-case output, with its
evidence ID mechanically rebound to the newly persisted event. It makes no
model call. The wrapper verifies frozen d94ded6, the actual imported package,
all 49 Python source hashes before/after, and the unchanged replay helper.

The existing SQLite fact keeps its ID and version 1. Canonical deduplication
merges the proposal into it, changes assertion status from asserted to conflict,
and adds the Redis requirement source as a second evidence record. The job
completes; no Redis fact is created. The chosen query returns no facts with the
simple embedding fixture, so this replay alone does not demonstrate what an
agent would see on successful retrieval.

A separate regression test exercises a known retrievable conflict after SQLite
restart. It reproduced missing assertion status in structured output and host
context. Commit 6ba17f5 exposes that status, labels unresolved conflicts, and
includes the label in the context budget. Fifty-nine affected tests and strict
production mypy pass. That presentation repair does not validate the model's
unsupported citation or change conflict persistence in this frozen replay.
