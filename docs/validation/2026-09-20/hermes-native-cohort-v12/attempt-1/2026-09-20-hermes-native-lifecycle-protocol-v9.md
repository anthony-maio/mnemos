# Native Hermes lifecycle v9: verify effective serving context

This prospective successor repairs the verified
[endpoint context mismatch](../validation/2026-09-20/ollama-context-boundary-v1/README.md).
The old alias loaded 65,536 tokens through native Ollama readiness, but
16,384 through the OpenAI-compatible endpoint used by Hermes. The new
context-pinned alias loaded 65,536 through that endpoint in the bounded probe.
This does not establish the cause of earlier tool omissions or change any
earlier failed lifecycle result.

## Frozen inputs

Use runtime `e419c2aa615e037e639a74b14615889d03f4e47f` and wheel SHA-256
`185f3df7720a32e5ed285d9c507e1e45e7f57370942740b53c4f8178bb97a97e`.
Keep all v8 lifecycle prompts, fixture values, eight tools and their schema,
capture, scope controls, restart behavior, source replay, mutation checks,
stored-copy erasure scans, and success criteria unchanged. No schema or
rejection-message experiment is included.

Use `mnemos-native-qwen35-9b:greedy-ctx64k-v2`, digest
`87bc5156d86918e0435431ab0286167a4c79c4a4a259b82fba5f3f75022006e2`.
The [v2 profile](../validation/2026-09-20/native-model-profile-v2/profile.json)
adds only `num_ctx: 65536` to the old alias's parameters and preserves its
weights, template, capabilities, and sampling settings. Keep local Ollama,
the pinned Nomic digest, thinking disabled, temperature zero, seed 42,
presence penalty zero, top-k 20, and top-p 0.95. All five cognition modules
remain disabled for this native transport/lifecycle baseline.

Commit and bind `hermes_lifecycle_run_v9.py`, `hermes_cohort_run_v7.py`,
`hermes_native_bootstrap_v6.py`, `hermes_request_observer.py`,
`ollama_readiness_v2.py`, the unchanged `hermes_trace_v2.py` evaluator, model
profile, and protocols before inference. Use the unchanged isolated Hermes
revision and cwd-only patch. No live host configuration, source, credentials,
or memory is changed.

The first import-only preflight stopped before importing Hermes because the
live checkout advanced to `ebe3fcb2f85ebdbe0e469e9d295636ac930246b4`.
Preserve that failed report. The isolated source remains pinned at
`14a346345486418aa85ed6526549412ee2665ade`; the successor independently binds
the live installation's revision and sorted `pip freeze` SHA-256, checks both
before and after each attempt, and records the dependency inventory. Existing
module-origin checks must still prove Hermes imports came from the isolated
checkout. No host downgrade is performed. This is another environment binding
and limits comparison with earlier cohorts; it does not change memory tasks.

## Serving-context checks

Readiness v2 sends one tool-free, 16-output-token request through
`/v1/chat/completions`, with a 180-second deadline and no retry. Verify model
digest before and after, the alias's pinned parameters, a nonempty assistant
completion without tool calls, positive finite GPU residency, and exactly
65,536 loaded context tokens in `/api/ps`. Do not infer readiness from an
ignored request option or a native `/api/chat` warmup.

The bootstrap preserves outgoing request bodies and response finish reasons.
Immediately after each observed model response, record `/api/ps` and require
one unambiguous matching alias/digest with the correct context and positive
GPU residency. A missing, invalid, or drifting profile fails the phase.
The runner adds a gate requiring one successful loaded-profile check for
each observed response. Existing lifecycle gates are retained in full.

The import-only preflight must load the helpers and discover all eight tools
without readiness, inference, or service mutation. Request recording remains
limited to the synthetic disposable session and omits authentication headers.

## Schedule and interpretation

Run exactly three fresh attempts with the existing eight host turns,
180-second host model budget, 240-second outer deadline, five-second kill
grace, and 1,800-second total attempt deadline. Stop dependent phases at the
first failure. Start the next scheduled attempt only after cleanup is
established. Preserve every result, including readiness or configuration
failure; do not add attempts, change inputs, or repeat selected phases.

Report model outcomes separately from verified effective-context checks.
Even three passes would establish only this bounded workflow; complete
capture, three-host continuity, independent biomimetic quality, Desktop, and
the operational pilot remain separate. The preceding single request
diagnostic and v3-v8 cohorts retain their own bindings and outcomes. This is
not a controlled causal estimate of context size's effect on tool selection.
