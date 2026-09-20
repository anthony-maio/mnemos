# Native Hermes lifecycle v10: stale-preview rejection repair

This prospective successor tests the verified
[stale-preview package](../validation/2026-09-20/stale-preview-checkpoint-v1/README.md)
without changing the tasks or acceptance requirements. Native v9 remains
zero of three complete lifecycles. Its first deletion apply in attempt 3
copied the reviewed IDs and digest but changed the preview limit. The repair
preserves rejection while exposing an actionable, noncommitted HTTP 409.
It has package evidence and no native recovery observation yet.

## Frozen inputs

Use source `f2f741443e60f838b392ae8ea1d1a0b4d8c6c1f2` and wheel SHA-256
`8552345d2c17c4c037347961eb3043d13b20f790f54a9ccc86a096add6197b0f`.
Use the isolated installed environment validated with that wheel. Bind the
source/package and installed-check reports, installed bytes, runner inputs,
and unchanged host patch before inference.

Keep the v9 bootstrap, evaluator, request observer, readiness helper,
context-pinned model alias/digest, Nomic digest, sampling, thinking setting,
host source, live interpreter revision and dependency inventory unchanged.
Local Ollama remains `http://127.0.0.1:11434`; the failed remote machine is
not used. No live host configuration, source, or memory is changed.

Keep all lifecycle prompts, exact fixtures, eight tools and schemas, raw
capture, scope controls, restart boundaries, source replay, mutation and
stored-copy erasure checks, turn limits, deadlines, and cleanup requirements
from [v9](2026-09-20-hermes-native-lifecycle-protocol-v9.md). All five cognition
modules remain disabled for this transport/lifecycle baseline. Do not fix
model arguments or skip a required tool call on the model's behalf.

Commit `hermes_lifecycle_run_v10.py`, `hermes_cohort_run_v8.py`, this protocol,
and their unchanged dependencies before the import-only preflight. The
preflight must discover all eight tools without model or service requests.
Readiness uses the OpenAI-compatible endpoint and requires the matching
alias/digest, 65,536 context tokens, and positive finite GPU residency.
Preserve outgoing request bytes and finish reasons, and verify the loaded
profile after each observed model response.

## Schedule and interpretation

Run exactly three fresh attempts. Stop dependent phases at the first failure
and stop the cohort if cleanup is uncertain. Keep the eight host turns,
180-second model budget, 240-second outer deadline, five-second kill grace,
and 1,800-second attempt deadline. Preserve readiness failures, missing
actions, rejected mutations, unsuccessful deletion, and all other outcomes.
Do not add attempts or repeat selected failed phases.

Report whether stale-preview rejection occurs, whether the model acts on
it correctly, and whether the full stored-copy erasure and replay checks
pass. A successful typed rejection alone is not a successful deletion or
lifecycle. This comparison does not establish a causal reliability gain;
identifiers and generated conversations vary between fresh attempts.

Three-host continuity, complete capture, independent module quality,
Desktop, and the operational pilot remain separate gates. Earlier native
cohorts and T18 negatives retain their bindings and results.
