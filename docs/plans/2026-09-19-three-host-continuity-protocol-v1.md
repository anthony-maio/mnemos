# Three-host native continuity protocol v1

Status: prospective; NOT RUN. This extends T14/T17 to the missing three-host
workflow. The earlier Codex/Claude result and the separate Hermes lifecycle
probe remain distinct evidence. A Hermes-only pass cannot close this protocol.

## Bindings and isolation

Run after the supplemental GPU queue is terminal and its coordinator and
worker processes have exited, and after the Hermes-only native lifecycle
probe has passed its bounded contract. Verify actual processes rather than
assuming a stale state file establishes exit. Do not compete with another
Ollama job or selectively repeat unsuccessful phases.

Use one verified Mnemos wheel and source checkpoint across every host. The
current candidate is `3730fd96c3c50c7ef6179b2b9f70414ab5a8fb17`, wheel SHA-256
`3772398a8e11b92d2cffd285b0571a733a9fd63cbdf932c30519d933cf0a6dd2`.
Record exact installed Codex and Claude versions and Hermes revision. Hermes
must use an isolated checkout with a recorded compatibility patch; never
present patched-host evidence as evidence for the unmodified installation.
Changes to this binding require a new manifest and output directory.

Use a disposable Windows service, SQLite database, scope registry, outbox,
host homes, and project directories. Register one profile with projects A and
B and explicit Windows/WSL aliases for A. All three hosts share A's identity.
No live configuration, source installation, or memory data is changed. Keep
only temporary copies of required authentication, omit them from artifacts,
and remove those copies during cleanup. Model-provided tools may access only
this service and the disposable project. Disable unrelated integrations.

Use the baseline module configuration and real local Nomic embeddings.
Automatic extraction cognition is unavailable. Conversation capture remains
enabled and explicitly qualified as captured evidence. This tests native
cross-host transport, identity, and lifecycle with real query embeddings;
it does not prove automatic fact extraction or biomimetic quality. Hermes
uses the same pinned Qwen digest and generation settings as the preceding
native lifecycle protocol. Record provider configuration and digests.

Each native phase starts a fresh conversation, without carrying earlier
answers in its prompt. Direct controller inspection can establish database
state but cannot substitute for a host-issued memory operation. The controller
may seed only unrelated and foreign controls. It must not teach, correct,
or forget the target on behalf of a host.

## Native phases

1. Codex teaches a project release-label convention, with an explicit durable
   remember request. Record the model-issued tool call, actual result, source
   evidence, fact ID/version, and completed job. Keep raw capture enabled.
2. Claude teaches an independent project review-label convention. Its prompt
   contains that convention only, never the Codex target value. Record the same
   operation and persistence evidence. This fact becomes the retained control.
3. Restart the service. A fresh native Hermes conversation retrieves both
   conventions by topic, without either value in its prompt. Require both
   original IDs, contents, scopes, and versions in real tool results. This
   explicitly tests Hermes recall of facts taught by each other host.
4. Hermes retrieves and inspects the release-label fact, then corrects its
   exact ID using the observed version. The new label is supplied only for
   this correction phase. Require the original ID at version 2, a completed
   job, and no change to the review-label or foreign control.
5. Restart the service. Fresh Codex, Claude, and Hermes conversations each
   retrieve the release-label fact by topic. Require the same version-2 target
   in each native tool result. Preserve any captured old mentions with their
   actual authority labels; an answer repeating the superseded value fails
   answer consistency even if a tool also returned the current value.
6. A fresh Claude conversation reviews the target and its retained
   conversational copies, then forgets the exact selected records with
   inspected IDs/versions. No controller mutation is permitted. Require the
   target's removal and preservation of the independent review-label fact,
   unrelated project control, and all foreign-project controls.
7. Restart and replay the original teaching requests and captured source
   events using their original identities. This is explicitly scripted API
   replay evidence, not a native host action. Require suppression without
   resurrecting target facts, raw sources, derived edges, or new jobs that
   can restore the forgotten values.
8. Fresh Codex, Claude, and Hermes conversations retrieve both topics. None
   receives either forgotten value in its prompt or injected context. Require
   no target/value in tool results, supplied model context, or final answers;
   the review-label control must remain available. Foreign-project canaries
   must never appear in project A's model requests or results in any phase.

Use unique synthetic values chosen before execution and recorded in the
manifest. Do not choose them after seeing retrieval results. Maintain exact
request/source identities for retries; do not turn a replay into new teaching.

## Evidence and failure rules

Correlate each successful operation with a model-issued native call, its exact
arguments, and its actual result. Reject controller-injected tool successes,
unissued calls, duplicate dispatch, malformed payloads, and tool errors.
Inspection must precede correction and targeted forgetting. Job status must
establish completion rather than treating a durable receipt as processing.

Use the existing host budgets: 180 seconds for a Codex/Claude phase; for
Hermes, eight model turns, a 180-second host budget and 240-second outer
deadline plus five-second termination grace. Keep maintenance at its existing
60-second observation budget and 45-second dispatch deadline. Stop dependent
phases on failure, retain raw synthetic outputs, and keep timeout/degraded
capture separate from successful fact operations. No deadline widening or
selective successful reruns may replace the original attempt.

Bind the reports to executed helper hashes, wheel bytes, source commit,
host versions and patches, model digests, registry/project identities, and
synthetic scenario bytes. Verify imports originate in the installed package.
Record startup, capture, callback, tool, shutdown, and process-exit outcomes.
Verify native tools completed before cleanup and inspect SQLite read-only
after service shutdown. A missing required phase is NOT RUN, not passed.

Passing closes only this bounded three-host native lifecycle observation.
Codex Desktop, automatic extraction quality, complete capture under arbitrary
interruptions, unindexed paraphrase erasure, automatic daemon restoration,
and the two-week daily-use pilot remain separate gates.
