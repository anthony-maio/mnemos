# Real-model evaluation status

## Current checkpoint: September 17, 2026, 20:50 UTC

The resumed held-out mechanism worker completed with exit code zero at
19:48 UTC. The coordinator then started lifecycle worker PID 47588, observed
live at this checkpoint. The seven-profile lifecycle matrix is running; the
baseline checkpoint contains 138 completed observations with next session index
45. Ollama on `.35` responds with both originally pinned model digests.
The completed mechanism phase has not been used to tune the frozen settings,
and no quality conclusion is accepted before the complete comparison.

The v4 queue resumed at 18:29 UTC after the local machine's power interruption
and restart at 18:09 UTC. The original process handle and worker PIDs were
absent. Recovery first timed out reaching Ollama without changing any files.
Once the endpoint returned, preflight verified the original model digests,
clean frozen source, runner hash, and every completed-phase artifact digest.
The stale lock and queue state were preserved before retiring that lock.
The unchanged coordinator resumed with `--resume`; PIDs 26488 and 35548 were
observed live. The earlier phase entry for PID 66348 is historical despite its
retained `running` label. The new phase entry identifies the resumed worker.
The [interruption snapshot](../2026-09-17/queue-interruption-v1/README.md) and
[recovery record](../2026-09-17/queue-recovery-v1/README.md) preserve both states.

The interrupted run uses frozen `d94ded6`, `qwen3.5:9b`
with thinking disabled and `nomic-embed-text:latest`. Development mechanism,
lifecycle preflight, and 10,000-fact performance phases completed. The
held-out mechanism phase finished; the seven-profile lifecycle matrix
is continuing. Recovery retained 62 completed case/pair IDs, 112 graph
observation rows, and 150 affective observation rows, including partial cases.
Those counts are execution progress, not quality outcomes.
No held-out quality conclusion is accepted yet.

The corrected performance measurement recorded recall p95 1,088.094 ms
against a 500 ms gate and capture p95 64.279 ms against 100 ms. See the
[preserved performance evidence](performance-corrected-v4/README.md).
The later [metadata-write comparison](access-metadata-comparison-v1/README.md)
does not establish a reliable speedup or close the recall gate.

The current coordinator is `.venv/run_thinking_evaluations_v1.py`;
status and outputs are under `real-model-v4/`, with persistent checkpoints
under `.venv/real-model-work-v4/`. The queue retains its original source,
dataset, model-digest, and option bindings. Later product changes, including
embedding roles, conflict collisions, concurrent retrieval, Claude startup,
and lost capture acknowledgments, are not part of this run. The latest complete
package checkpoint is separate source `dedfcfd`, with 914 tests passed and
four skipped, followed by formatting, typing, JavaScript, build, and Twine
checks. Its fresh installed wheel passed 120 selected contract cases with two
matching platform skips and
the synthetic Stop replay assertions on C: SSD fixtures. These checks do not
promote the frozen evaluation's quality results or resolve the earlier D:
storage timeouts. See the [package checkpoint](../2026-09-17/claude-startup-package-checkpoint-v1/README.md).

Current model pins:

- Cognition: `qwen3.5:9b`, digest
  `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`.
- Embeddings: `nomic-embed-text:latest`, digest
  `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
- Temperature 0, output budget 512, provider timeout 120 seconds,
  three repetitions, `think=false`, no explicit embedding prefixes.

The v3 queue terminated with a lifecycle serialization failure. Its
[failure archive](lifecycle-serialization-failure/README.md) and earlier
results are preserved. It is not the active evaluation. The sections below
retain historical status and estimates, not current readiness claims.
See the [claim and evidence ledger](claim-evidence-ledger.md) for remaining
acceptance gates.

## Historical v3 checkpoint

At this earlier checkpoint, the fresh v3 evaluation queue was running from frozen
`b7a9d9d`. All four integrated lifecycle preflight cases passed. The 10,000-fact
performance phase completed: capture passed at 57.410 ms p95; recall failed at
1,314.583 ms p95 with two stalls inside durable retrieval metadata updates.
The held-out mechanism phase is now active; lifecycle follows it. The repaired
mechanism development run completed all observations and passed its graph
correctness controls; it established no graph recall benefit. See the
[development observations](mechanism-development-v2.md).
Complete repository verification passed: 624 tests passed,
2 skipped, followed by formatting, typing, JavaScript syntax, packaging, and
installed-wheel lifecycle checks. The [verification record](graph-v4-verification.md)
preserves the logs and artifact hashes.

The serialized queue runs lifecycle preflight, 10,000-fact performance,
held-out mechanism evaluation, then the seven-profile lifecycle matrix. Its
status and logs are in `real-model-v3/`; persistent fixtures and checkpoints
are in `.venv/real-model-work-v3/`. The coordinator is
`.venv/run_graph_v4_evaluations.py`, with its SHA-256 recorded in
`real-model-v3/queue-status.json`. It requires matching model digests, a clean
frozen checkout, passing broad/wheel checks, and completed development
correctness controls. Resume rejects a still-active prior coordinator or
child process. Negative quality and coverage results are retained separately
from execution and correctness failures.

The superseded v2 lifecycle matrix is stopped. It preserved 600 baseline
observations and 12 full-profile observations before a target-mapping defect
was confirmed. All 60 baseline correction attempts and 60 forget attempts
failed in the evaluator before reaching the corresponding service operation.
The replacement contract will name the prior source event and separate
replacement content from the operation instruction. Every profile must be
rerun under a new dataset/source binding; no existing result will be rewritten.

Baseline provider callbacks verified real embeddings in all 600 observations
and cognition in 150. Baseline deliberately disables cognition-dependent
modules, so the other 450 observations have embedding-only evidence. The
artifact has 480 complete and 120 partial observations and correctly reports
an incomplete run. Its 4 percent exact supported-fact proxy, 20 percent
answerable recall, and zero no-answer accuracy are not accepted quality
results: the correction/forget mapping defect contaminates lifecycle outcomes,
and exact frozen-text matching is narrower than semantic correctness.

Processes 47644 and 55876 were stopped intentionally after command-line
identity verification. Do not resume this v2 run or its old coordinator. The
files below are retained diagnostic evidence. The replacement lifecycle run
has not started. The original mechanism development run was also stopped after
a production graph correctness failure. A repaired development run is now
complete from detached checkpoint `b7a9d9d`.

The completed development run uses graph builder v4 and the complete
`2026-09-17-module-evaluation-protocol-v2.md` amendment. It retains five graph
cases, five affective pairs, and three repetitions, with unchanged authored
labels and model pins. Its artifacts are in `mechanism-development-v2/` and
persistent fixtures in `.venv/mechanism-development-work-v2/`. The binding is
`ac58f07c79c20e91bee95394c7e5f47348a4d51fe897861abe465211fff272b5`.
The frozen-vector digest is
`1370c4ad2cde46bfb9c333f75eaaf8a53bf978c0b916b5aedead530b755efc36`.
Full check logs and pytest XML are preserved in `graph-v4-verification/`.

The mechanism development run used detached source `e9d8ca2` in
`.venv/mechanism-frozen-v1`, five development graph cases and five development
affective pairs, with three repetitions. Its artifacts are in
`mechanism-development-v1/`; persistent fixtures are in
`.venv/mechanism-development-work-v1/`. Live provider metadata matches the
model pins below. The manifest binding is
`4738f773f7a56919c459ba1ea36c69d9893085ba16a7775f527305b8545109b9`.
Its first completed graph repetition changed both persisted edges and ranked
results when the fact insertion order was reversed. The energy oracle still
matched production. Processes 29796 and 31636 were stopped after command-line
identity verification; `mechanism-development-v1/stopped.json` records the
completed measurement hash. The databases and partial next repetition remain
intact. No held-out mechanism case was executed.

This development run cannot meet the held-out coverage minimums and cannot
establish quality acceptance. A preceding offline fixture completed three
production SQLite graph repetitions with synthetic embeddings and passed the
graph correctness controls. It made no model calls.

The lifecycle v3 amendment preserves every original v2 event, fact, and query.
It adds source-event target identifiers and separates correction content from
the action instruction. Since the v2 labels were already observed, v3 is a
repaired rerun, not new unseen held-out evidence. A new development fixture
now exercises actual correction and forget dispatch before the matrix restarts.

The first real v3 development preflight used source `634b460` and attempt
`8ce175be0c894d83928e8309ecb3724b`. Three of four runs passed: baseline
correction preserved the corrected fact ID and version after restart, and
both baseline and full profiles passed forget, replay suppression, and restart.
Full-profile correction was blocked before correction dispatch: its extraction
response repeated `{"operation":"add"}` objects without content or evidence
until the unchanged 512-token budget was exhausted. JSON parsing failed and
the durable extraction job remained pending. The evaluator correctly reported
no accepted target instead of guessing a fact to correct. Raw responses and
manifests are preserved in `lifecycle-target-development-v3/`. This is a
development result, not quality acceptance.

The extraction schema repair is committed in `b5a2834`. Every operation
alternative now carries all required fields. The old schema combined common
object constraints with partial `oneOf` branches. The llama.cpp schema reader
selects union alternatives before reading sibling object constraints, which
explains the observed incomplete objects; see its
[schema reader](https://github.com/ggml-org/llama.cpp/blob/master/common/json-schema.cpp)
and [grammar converter](https://github.com/ggml-org/llama.cpp/blob/master/common/json-schema-to-grammar.cpp).
These upstream sources corroborate the failure mechanism; the lab server's
embedded converter revision was not independently identified.

A fresh source-bound run from detached `b5a2834` passed all four real Ollama
development cases under attempt `b3fb69233d634934949a2432df377b59`. Both profiles
preserved corrected identity and version across restart, and both passed forget,
replay suppression, and restart. Model pins, token budget, prompts, and scenario
labels remained unchanged. Runtime evidence and operation validation remain in
place. Artifacts are preserved in `lifecycle-target-development-v4/`; this
development gate does not establish held-out quality or graph correctness.
The matrix remains stopped while the repaired graph checkpoint is validated.

The stopped run used frozen source commit `5e852e7` in
`.venv/eval-frozen-v2`. It started on September 16, 2026 local time, after
source-hash preflight confirmed 118 Python/configuration files. Its source
hash is `dd0631c56f5270679a84eee0d3f33c2c6f9c86f6cd7dcd4e4dcc98fde196d949`.

The lifecycle experiment ran against the lab Ollama endpoint, preserving
independent databases and checkpoints:

1. Lifecycle: 200 held-out synthetic sessions, seven profiles, three
   repetitions, 4,200 observations.
2. Competing-memory supplement v1: held before execution after a design audit.
   Its 25 candidates all fit inside the production direct candidate limit,
   and it lacks independently labeled state-appropriate targets. It cannot
   establish graph or affective benefit. A valid preregistered replacement
   remains required; no supplemental acceptance result is claimed.

The initial lifecycle estimate was about 42 hours. The withdrawn supplement's
35 to 70 hour estimate is no longer a scheduled commitment. The machine and Ollama
endpoint need to remain available. Completed checkpoints can be resumed with
the same dataset, source, model, and option hashes after interruption.

The coordinator status is `real-model-v2/queue-status.json`; per-phase logs
and completed reports are in `real-model-v2/`. Per-observation checkpoints
and isolated SQLite files are in `.venv/real-model-work-v2/` relative to the
implementation worktree. The coordinator was first stopped to prevent
its automatic handoff to the invalid supplement. The lifecycle worker
continued until the later target-mapping failure described above.
Do not relaunch the old queue script, which still includes the withdrawn
supplement. No task scheduler or system startup entry was installed.

Provider pins recorded in every profile manifest:

- Cognition: `gemma3:4b`, digest
  `a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`.
- Embeddings: `nomic-embed-text:latest`, digest
  `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
- Temperature 0, output budget 512, provider timeout 120 seconds,
  top-k 5, context budget 1,200 tokens.

Execution completion and quality acceptance are separate. A complete
observation can contain failed quality outcomes. No interim result justifies
promoting the biomimetic retrieval profile over the baseline default. The
supplement records whether graph targets were already direct semantic
candidates; such cases cannot establish graph utility. The datasets remain
synthetic and do not replace the two-week daily-use pilot.

The initial five-observation attempt under `real-model/` has invalid source
provenance and is excluded. See the [progress record](progress.md). Frozen
package evidence is in [frozen-build.json](frozen-build.json).
