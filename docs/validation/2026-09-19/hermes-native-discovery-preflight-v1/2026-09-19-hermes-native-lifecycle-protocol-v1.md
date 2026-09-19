# Native Hermes model-driven lifecycle protocol

This probe extends the passing single-turn native recall check and the separate
fixture-provider MemoryManager lifecycle/recovery checks. The remaining claim
requires the real Hermes model to issue the lifecycle tools itself. A scripted
tool call or a plausible final answer does not satisfy that claim.

Use the verified Mnemos wheel on the replacement local RTX 4080 machine, the
pinned Hermes revision `14a346345486418aa85ed6526549412ee2665ade`, and the
previously verified isolated CLI checkout with only its cwd propagation patch.
Keep the live host source, home, configuration, and database unchanged. Bind
every report to the runtime source commit, wheel digest, runner commit, exact
runner/bootstrap/evaluator/protocol hashes, host revision and patch, model
digests, provider options, phase outputs, and selected synthetic database rows.

The model is local Qwen3.5:9b, digest
`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`.
Embeddings are local Nomic, digest
`0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
Use the existing native host profile: 65,536 context tokens and reasoning
disabled. Record effective requests and cold loading separately. Run no new
model work until the current supplemental queue is terminal and its verified
coordinator and worker processes have exited. The runner rejects a live prior
process or an uncompleted queue record; an observation timeout is not exit.

## Phases and required evidence

Use a fresh disposable home, project/foreign-project registry, service database,
descriptor, occurrence ledger, and outbox. The controller may seed one unrelated
project control and one foreign-project control. It must not seed the target
fact, correct it, or forget it. Those actions belong to the model-driven host.

1. Teach: a native Hermes CLI conversation calls `mnemos_remember` for exactly
   "The project release label is cedar-ridge-731." under a fresh logical request
   identity. Require a committed event/job response, a native job-status check
   for that returned job ID, and one persisted asserted target after maintenance.
   The immediate job check may report pending, leased, or completed; only the
   controller's subsequent maintenance establishes completed processing.
2. Restart and recall: a fresh native conversation must call `mnemos_retrieve`
   and return the same target ID, exact content, and version 1. The prompt does
   not supply the label or fact ID.
3. Correct: a fresh conversation retrieves and inspects the asserted target,
   then issues `mnemos_correct` with that exact ID, version 1, a new request
   identity, and exactly "The project release label is maple-ridge-842."
   Maintenance must preserve the target ID and advance it to version 2.
4. Restart and corrected recall: another conversation retrieves the same ID,
   corrected content, and version 2 without receiving the label in its prompt.
5. Forget: authorize removal of the project's release label and its retained
   conversational copies. A fresh conversation retrieves matching facts,
   inspects each exact target, and issues `mnemos_forget` using the returned
   version. This must include the asserted target at version 2. Require absence
   and replay suppression; the unrelated and foreign controls must retain their
   IDs/content/versions. Every removal requires its own preceding exact inspect.
6. Restart, replay, and final native recall: a separate MemoryManager API
   process replays the original remember request and actual teaching transcript
   under the original host session identity, through the real provider and
   bridge. Label this scripted replay as API evidence. A fresh model-driven CLI
   conversation then retrieves the project label. Require a structurally valid
   response with no forgotten target or original/corrected label, and no leaked
   label in the model's supplied context. Retained conversational copies that
   expose the forgotten label fail this gate; do not disable capture to avoid it.

Every model-driven tool dispatch must match an earlier model response by
response-scoped call identity, tool name, and exact arguments. Preserve raw
provider call IDs separately. Reject unissued, mismatched, duplicate, failed,
malformed, or undispatched calls. Only the six `mnemos_*` provider tools count.
Inspect must precede correction/forget. Model answers and controller API writes
cannot substitute for these traces. Foreign controls must never appear in
model requests or native tool results. Native callback errors fail acceptance.

The controller runs ordinary service maintenance between phases and inspects
selected persisted rows. It checks the original controls, their source rows,
and evidence without allowing duplicate control assertions. Maintenance has
a 60-second observation budget and a 45-second service dispatch deadline;
processing errors stop the probe. Automatic capture remains enabled. This host probe
uses the baseline module configuration, real host inference and real embeddings,
with automatic extraction cognition disabled. It does not evaluate biomimetic
quality or replace the separate seven-profile ablation. Captured copies must
remain distinguishable from asserted facts.

## Execution and failure handling

First run `--preflight-only` using the installed wheel Python from outside the
checkout. It imports the actual patched Hermes CLI, verifies module origins,
discovers the verified wheel's provider through Hermes's actual plugin loader,
initializes its MemoryManager/provider with the disposable scope and shared
Hermes identity home, and checks tool discovery. The API replay must initialize
with that same home so original occurrence timestamps remain stable. Preflight
must make no model request, service request, or
memory mutation. Its status is `preflight_passed`, never lifecycle `passed`.

Commit the tested runner and protocol before model execution. Each native
run copies and executes the bound bootstrap/evaluator bytes, so subsequent
checkout edits cannot silently change the probe. Each native
conversation permits eight model turns, a 180-second host run budget, and a
240-second outer host deadline followed by a five-second termination grace.
Preflight permits 60 seconds. Preserve failed outputs and stop dependent phases
at the first failed gate. Do not widen deadlines, inject tool results, replace
the model answer, or selectively retry a failed phase to make the same run pass.
Any successor attempt requires a new output directory and a separate report.

Route bridge autostart attempts to a deliberately invalid disposable config.
The controller owns service restoration; a failed bridge cannot open a live
runtime. Exclude authentication descriptors, credentials, and live user content
from archives. Keep raw synthetic responses and negative results.

Passing establishes this bounded native lifecycle on an isolated patched host.
It does not establish unmodified CLI compatibility, model-quality gains,
hard-kill recovery, complete cross-host capture, or a two-week operational pilot.
