# Hermes durable evidence callback development

The adapter adds a negotiated v1 evidence input to ordinary turn and session
end callbacks. An explicit empty list is authoritative; the model-facing
transcript and callback text cannot substitute for it. Native identity,
origin, and summary classification checks still apply before enqueueing.

Thirteen original tests failed because the capability and callback keyword
were absent. The first candidate run had 33 passes and one test-fixture error:
the normal payload omits identity_status, which the service defaults to normal.
The corrected assertion reads that existing contract. A further malformed
classification test reproduced an unhashable-dict error; the candidate now
rejects that field with ValueError before capture. The final focused run
passes 90 cases, including 14 new tests. Strict type checking also passes.

The outbox tests preserve canonical payloads through session switches and
provider restart, keep summaries capture-only, reject duplicate IDs, and
check both callback paths. They do not prove that a real host supplies the
required projection or that hybrid components retain separate durable origins.
Full source/package checks and isolated host integration are separate gates.

Original red inputs, all test reports, and candidate source are indexed.
This README is outside the index.
