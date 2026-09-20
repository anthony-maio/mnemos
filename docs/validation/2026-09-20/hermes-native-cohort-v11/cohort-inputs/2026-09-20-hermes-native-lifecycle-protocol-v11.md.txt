# Native Hermes lifecycle v11: new guidance package prerequisite

This prospective successor tests the new guidance package bound to the
frozen source and wheel checkpoints below, without changing the tasks or acceptance requirements. Native v10 remains
one of three complete lifecycles, and that earlier result is retained as a
prior outcome rather than treated as a causal comparison. This successor
checks the new package against the same lifecycle and acceptance gates.
The Hermes-only setup does not install the shared Codex project instructions,
so it cannot measure the effect of the instruction repair. That remains a
separate three-host observation.

## Frozen inputs

Use source `697b6720b201d578c03ce33d98dff3dda6d2def8` and wheel SHA-256
`49c7d791cd6b5d7645125cec72371b453607543b371e6834c6a2e122774f42b0`.
Use the isolated installed environment validated with that wheel. Bind the
source/package and installed-check reports, installed bytes, runner inputs,
and unchanged host patch before inference.

Keep the v10 bootstrap, evaluator, request observer, readiness helper,
context-pinned model alias/digest, Nomic digest, sampling, thinking setting,
host source, live interpreter revision and dependency inventory unchanged.
Local Ollama remains `http://127.0.0.1:11434`; the failed remote machine is
not used. No live host configuration, source, or memory is changed.

Keep all lifecycle prompts, exact fixtures, eight tools and schemas, raw
capture, scope controls, restart boundaries, source replay, mutation and
stored-copy erasure checks, turn limits, deadlines, and cleanup requirements
from [v10](2026-09-20-hermes-native-lifecycle-protocol-v10.md). All five cognition
modules remain disabled for this transport/lifecycle baseline. Do not fix
model arguments or skip a required tool call on the model's behalf.

Commit `hermes_lifecycle_run_v11.py`, `hermes_cohort_run_v9.py`, this protocol,
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
