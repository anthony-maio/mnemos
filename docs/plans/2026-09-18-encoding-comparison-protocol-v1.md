# T18 applied encoding comparison protocol v1

This is a separate comparison from the standard lifecycle profile matrix. It
uses the existing `local-memory-heldout-v3` rows and labels. The amended v3
fixture has already been observed in the earlier invalid run and is reused
here; this comparison does not claim a newly unseen held-out dataset.

The question is whether the existing surprisal gate and a deterministic
length/noise rule produce different extraction and lifecycle outcomes. The
three arms are:

1. `full`: all five existing modules, including the production surprisal gate.
2. `ablate-surprisal_gating`: the same four other modules with no surprisal
   gate. This is the ungated control.
3. `length-noise`: the same four other modules with the applied replacement
   rule and no surprisal gate.

The primary replacement comparison is `full` versus `length-noise`.
`ablate-surprisal_gating` is retained to separate the effect of either gate
from the four-module pipeline. The normal lifecycle runner and its output
directories are not reused.

## Frozen encoding rule

The applied rule runs after the persisted capture has passed the write
firewall and only at automatic extraction-job processing. It keeps text when
the stripped content has at least 50 characters and its case-folded content is
not exactly one of `ok`, `okay`, `thanks`, `thanks!`, or `got it`. Rejection is
recorded as `length_noise_filtered` in the completed durable job. The episodic
source event and job remain durable, and no extraction model, embedding, or
fact write is attempted for the rejected job.

Explicit `remember`, `correct`, and `forget` operations bypass this rule.
Their lifecycle behavior, target resolution, expected versions, and
denominators remain unchanged. Automatic `capture`, `noise`, and `response`
operations are represented by their durable extraction jobs. The fixture's
`write_mode` remains a reported source field; it is not silently treated as a
service-side filter switch.

The three-arm runner intentionally preserves the `full` arm as the existing
surprisal-only behavior and uses `length-noise` as the replacement arm. It
does not add a common length filter to `full`, because that would change the
question. A future common-filter study would be a separately named protocol
and artifact namespace.

## Measurement and denominators

Every arm uses the same ordered sessions, events, repetitions, top-k,
context-token limit, dataset bytes, policy bytes, protocol bytes, and source
code binding. A filtered event stays in the raw operation trace and in the
session observation. The event-level records include the dataset event ID,
runtime job ID, operation, fixture write mode, job status, applied policy,
decision, and encoding diagnostics.

For each arm, report the all-event denominator, the automatic-extraction
denominator, the applicable policy denominator, and the number of completed
job decisions. A length rejection is not a surprisal rejection. In the
length-noise arm, the applicable policy denominator is the automatic
extraction jobs. In the full arm, the applicable policy is the existing
surprisal gate and no common length eligibility is imposed. Missing, pending,
failed, and unavailable jobs remain unjudged and are not converted to false or
kept.

The paired lifecycle metrics preserve one observation per session and
repetition. Paired differences are treatment minus comparison and are
cluster-bootstrapped by session with the existing harness. Filter counts are
reported separately from retention, retrieval, precision, and lifecycle
metrics. No coverage or lifecycle denominator is reduced because an event was
filtered.

## Provenance and token accounting

The runner records the exact dataset, protocol, policy, model, embedding,
generation options, and source-code hashes. Real execution requires the live
model and embedding digests from the configured Ollama endpoint. `--think
false` is an explicit supported option. Preparation writes a `NOT RUN`
artifact and performs no provider or worker call.

When the applied length filter rejects an event before provider work, token
counters are null and the record says `model_call: not_requested`. Other rows
without an event-level attribution say `model_call: not_attributed`; their
session-level provider response metadata remains authoritative. A missing
provider counter is not changed to zero, and no billing cost is inferred.

Resumption is allowed only when the manifest, checkpoint, protocol, policy,
dataset, options, model digests, embedding digest, and source-code binding
match exactly. Existing output or work directories with a different binding
are rejected before any checkpoint or report is touched. Failed attempts are
retained in numbered attempt directories.

No new labels, thresholds, dataset content, tuning, or semantic-quality claim
is introduced. Real execution remains separate from human acceptance and
host workflow evidence. It must wait for the existing real-model queue and
endpoint availability.
