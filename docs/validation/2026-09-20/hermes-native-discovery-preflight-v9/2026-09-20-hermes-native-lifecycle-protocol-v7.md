# Native Hermes lifecycle v7: deletion coverage receipts

This prospective successor tests the verified deletion receipt package in
the actual Hermes host with automatic capture enabled. The
[v6 cohort](../validation/2026-09-19/hermes-native-cohort-v6/README.md)
remains zero of three completed lifecycles on `db932be`. Two attempts selected
only the target and retained conversational copies; the third omitted source
events required by its selected copies. No v6 outcome is rewritten.

Use runtime `1a110583a18408cbd3de4a695a9962109c4d02e7`, wheel SHA-256
`4e3d16d00f7b9ab54fe1e5350f7cf0fbe6209dc881bfc6d7d9f974f1c3f697a5`.
Its [checkpoint](../validation/2026-09-19/forget-coverage-checkpoint-v1/README.md)
passes 1,439 source tests, 608 installed contract cases, four matching skips
in each suite, six source/package checks, and seven replay assertions.

The package adds bounded deletion coverage receipts and guidance to report
them. It also includes the earlier typed rejection for missing required
source selections. Neither change chooses records for the model or proves
that the model will use the receipt correctly. The separate direct development
comparison on `e454a85` does not establish native-host acceptance.

## Frozen workflow and criteria

Use `hermes_lifecycle_run_v7.py`, `hermes_cohort_run_v5.py`, the unchanged
`hermes_native_bootstrap_v4.py`, and unchanged `hermes_trace_v2.py` evaluator.
Commit the runner, readiness helper, and this protocol before inference.
Bind them, the earlier v3-v6 protocols, and the model profile by hash.

Retain all v6 prompts, fixture values, scopes, controls, eight provider tools,
automatic capture, restart sequence, original-source replay, evaluator gates,
retained-label scans, and required inspection/source selection. The model
must retrieve and inspect the target, observe its preview before applying it,
inspect selected additional facts, and explicitly select required sources.
The controller supplies no fact IDs, repairs no arguments, and performs no
model-requested deletion itself. Keep the old substring check on final model
requests and the scoped scan across facts, events, evidence, revisions, and
jobs. A receipt reporting zero retained preview records cannot override any
failed erasure gate. The new receipt fields are not substitutes for state
inspection. No evaluator threshold or gate changes.

Retain the local Qwen3.5:9b greedy-v1 alias and its existing digest, Nomic
embedding digest, 65,536-token context, no reasoning, temperature zero,
seed 42, and presence penalty zero. Check model digests and sampling profile
before the run and each phase. All five cognition modules stay disabled for
this host-lifecycle baseline; their quality experiments remain separate.

The pinned Hermes checkout and isolated cwd-only compatibility patch remain
unchanged. Use a disposable home, project, registry, service, and database.
Do not change live host configuration, source, credentials, or memory.

## Recorded readiness, before workflow inference

V6 already made an empty-prompt generate request to load the model. V7
replaces that load call with one recorded, minimal chat request using
`ollama_readiness_v1.py`, after control seeding and before the teach phase.
It has no tools or memory content, uses the same model and context, and
allows at most 16 generated tokens. The single chat request has a 180-second
timeout; metadata calls have 10-second timeouts. There are no readiness
retries. This does not lengthen any native workflow phase deadline or the
total attempt deadline.

Readiness requires HTTP 200, a structurally complete assistant response,
nonempty text, no error or tool call, an unchanged pinned digest before and
after, and a matching loaded model with positive reported GPU residency.
Punctuation such as `READY.` is valid; this is not an answer-accuracy test.
Record request bytes, raw responses, timings, server version, model metadata,
and loaded-model state. Positive VRAM usage does not prove every layer was
offloaded. On failure preserve the evidence, abort dependent native phases,
and count that scheduled attempt as failed. The import-only preflight must
not call readiness or any model endpoint.

The runner also decodes checkpoint files and native trace JSON as UTF-8
explicitly. This preserves emitted text without rewriting the trace or
changing the evaluator's interpretation.

## Fixed schedule and interpretation

Run exactly three independent fresh attempts. Keep eight host turns,
180 seconds per host model run, the 240-second outer host deadline,
five-second kill grace, and the 1,800-second total attempt deadline.
Stop dependent phases after failure. Start the next scheduled attempt only
after established cleanup. Do not selectively repeat phases, add attempts,
change inputs during the cohort, or omit startup failures from the denominator.

Report all three outcomes, unreached phases, readiness results, cleanup,
and exact runtime/model/input bindings. This tests a newer package and
readiness procedure together; it is not a controlled causal estimate of
either change. A pass establishes only that bounded attempt. Three-host
continuity still requires its separate stronger test after a bound Hermes
prerequisite passes. Complete capture, real-model quality, Desktop, and the
pilot remain separate gates. Retain all earlier negative results and update
the research handoff after the cohort reaches a terminal state.
