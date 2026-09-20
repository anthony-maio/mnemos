# Historical assertion contract review

This is a prospective development contract review. It preserves the T18
negative results and does not change prompts, runtime behavior, thresholds,
datasets, tests, or acceptance status. It is not a held-out evaluation and its
expectations are not independently human validated.

## Why this review is needed

The preserved lifecycle archive records partial observations whose correction
failed with `missing_accepted_target` after automatic extraction returned
`decision="no_durable_facts"`. The diagnosis in
[`lifecycle-extraction-diagnosis.md`](../validation/2026-09-19/lifecycle-extraction-diagnosis.md)
shows that this pattern accounts for the 50 lifecycle partials. The same
missing-target category appears in the 34 partial observations of the terminal
encoding comparison ([comparison diagnostics](../validation/2026-09-19/comparison-diagnostics.md)).

The representative held-out case uses an event saying, "The earlier project
note said ..." followed by an explicit current correction. Its fixture labels
the earlier statement as superseded and the current statement as supported,
but the runtime has no accepted fact to target after the first event is
rejected as historical context. The raw archived reports preserve the exact
case and response; they do not establish which product policy is desired.

The current extraction contract in
[`sleep.py`](../../mnemos/modules/sleep.py) requires direct source support,
preserves assertion provenance, and permits `no_durable_facts` when no durable
fact is supported. The service's correction path in
[`memory_service.py`](../../mnemos/memory_service.py) is target based. The
adapter reports `missing_accepted_target` when accepted evidence cannot resolve
exactly one target in
[`ollama_adapter.py`](../../benchmarks/local_memory/ollama_adapter.py).
Those behaviors make refusal and target absence observable. They do not answer
whether a historical statement should be retained as provenance, indexed as a
durable historical fact, or discarded from durable fact storage.

## Unambiguous contract expectations

These are the minimum behaviors that follow from the current documented
contract. They are development expectations for a future review, not claims
about an independently validated product policy.

| Case | Plain-language input | Expected storage result | Expected correction result | Rationale |
| --- | --- | --- | --- | --- |
| 1 | The user says, "The project cache uses Redis for session tokens." | Store one asserted fact with the source event as evidence. | The fact is available as a versioned target. | A direct current assertion is source-supported. |
| 2 | The user says, "For this project, the deployment target is AWS ECS." | Store one project-scoped asserted fact with its project scope and source evidence. | The fact is available as a versioned target. | Scope and assertion status are explicit in the source. |
| 3 | The user says, "My preferred editor is VS Code for this project." | Store one preference with preference provenance, scope, and source evidence. | The preference is available as a versioned target. | A stated preference is explicitly included by the extraction contract. |
| 4 | The user says, "The earlier project note said the cache used Redis." | Do not silently store it as a current asserted fact. Retain the source event. | No correction may guess a target from this event. | The wording is attributed historical context, and the current prompt rejects unsupported current fact status. |
| 5 | The user says, "A previous manager required weekly releases." | Do not silently convert the report into the user's current requirement. Retain the source event. | No correction may guess a target from this event. | Attribution and time qualification make current authority unresolved. |
| 6 | The assistant says, "The earlier plan said we would use PostgreSQL." | Retain the assistant event as conversation evidence only; create no user fact. | No correction target may be derived from the assistant event. | The prompt explicitly rejects assistant claims as proof of user preference or completion. |
| 7 | An accepted fact says, "The cache uses Redis," and the user says, "Correction, use PostgreSQL." | Keep the accepted fact until the correction commits; retain both source events. | Apply one version-checked correction to the accepted fact. | The current mutable path requires an existing target and expected version. |
| 8 | An accepted project fact says, "Deploy to ECS," and the user says, "Correction, deploy to Kubernetes." | Preserve project scope and source evidence while preparing the revision. | Commit one version increment and preserve the previous content as a revision. | Correction is source-backed and target identity is available. |
| 9 | An accepted preference says, "Use VS Code," and the user says, "Correction, use Zed." | Preserve preference identity, scope, and evidence while preparing the revision. | Commit one version-checked preference correction. | A direct preference correction has a resolvable accepted target. |
| 10 | No accepted fact exists, and the user says, "Correction, use PostgreSQL instead of Redis." | Do not fabricate a target or silently claim a revision occurred. Retain the event. | Return an explicit unavailable-target outcome. | The current correction API is target based and must not guess fact identity. |
| 11 | Only an attributed historical event exists, and the user says, "Correction, the current value is Kubernetes." | Preserve the correction event and report that no accepted target exists. | Do not silently bootstrap a revision under an invented identity. | Target absence is safe and observable, but bootstrap policy is unresolved. |
| 12 | A prior fact was rejected as `no_durable_facts`, then the user submits a direct current assertion without calling it a correction. | Store the new direct assertion as a new fact with its own evidence. | It becomes a target for later correction. | A direct current assertion is distinct from a target-dependent correction. |

## Product choices still unresolved

The cases above deliberately leave two choices open.

First, the product has not decided whether an attributed historical assertion
should be represented as a durable historical fact with temporal and attribution
metadata, or retained only as episodic provenance. The current extraction prompt
does not establish that policy. A decision needs a product-level definition of
what retrieval, correction, and answer generation should do with historical
assertions. The T18 fixtures cannot make that decision because their labels are
synthetic and their `human_acceptance` status is `not_run`.

Second, the product has not decided whether an explicit correction with no
accepted target may bootstrap a new fact. The current runtime correctly refuses
to guess a target. Allowing bootstrap would require a separately defined
operation and evidence contract, including how the new identity, revision
history, and assertion status are recorded. It should not be inferred from the
current `correct` operation or repaired by relabeling the archived partials.

## Minimum independent review before implementation

A prospective review should have a product owner or independent reviewer assign
the expected outcome for all 12 cases, including the historical-provenance and
correction-bootstrap choices. The review should then freeze the resulting
contract before any implementation or evaluation work. Until that happens, the
archived lifecycle and encoding negatives remain valid observations of the
current contract and model behavior, not evidence for a runtime fix or an
acceptance result.
