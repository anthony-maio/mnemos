# Real-model evaluation status

Current status: the v2 lifecycle matrix is stopped. It preserved 600 baseline
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
has not started. The mechanism development run was also stopped after a
production graph correctness failure. No model experiment is currently running.

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

The extraction schema is being repaired so every operation alternative carries
all required fields. The old schema combined common object constraints with
partial `oneOf` branches. The replacement keeps runtime evidence and operation
validation and does not raise the token budget. It requires a fresh source-bound
development run before the full matrix restarts.

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
