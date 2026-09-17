# Native Codex generated-config check

Codex CLI 0.146.0 used the installed `20f271d` wheel and the configuration
written by `apply_host_integration`. The test did not insert a command wrapper,
write hooks.json, or pass `--enable hooks`. It used a private temporary home,
project, scope registry, service, and database. The generated MCP command
resolved from the separate installed-wheel environment.

The native turn completed in 36.915 seconds and returned code zero. Its answer
contained the random canary stored only in the fixture memory:
`The private release codename is mnemos-a92513ffc99c.` The prompt did not
contain that value, and the complete JSON event stream recorded no tool call.
Both user and assistant events had the `codex_hook` source kind and survived
service restart. This supplies evidence that the generated native hook path
delivered the canary and durably captured the turn.

There was no wrapper trace, so individual hook invocation counts, durations,
and native additionalContext payloads are unavailable. The successful direct
SessionStart warm-up is recorded separately and excluded from native capture.
The host reported 19,189 input tokens, including 10,624 cached input tokens,
and 19 output tokens. It did not report a model name. Its three error-labeled
items contain two hook-trust notices and one shortened skill-description
notice; no other error item was observed. Global skills remained discoverable.

Every fixture text maps to the same constant unit vector. This intentionally
tests hook delivery and persistence without measuring semantic selection.
Cognition was unavailable and all biomimetic modules were disabled. No
extraction, retrieval-quality, performance, full lifecycle, or Desktop claim
follows from this result. The prior SimpleEmbeddingProvider preflight failure
is retained under `native-codex-generated-v7`.

An offline preview also confirmed that an existing `features.hooks = false`
setting remains false after rendering. That preview is not a native disabled-
feature test. Users who explicitly disable hooks do not get automatic capture.

The wheel SHA256 is
`0ecaa634fe447621d6366b93accad7abfa2495063d9b425e6043973ea7b86663`.
All 53 tracked package files match the pinned source after line-ending
normalization; installed package bytes match the wheel. The exact runner,
generated configuration, reviewed synthetic host output, durable events, and
cleanup results are indexed in `evidence-index.json`. The temporary credential
copy and runtime descriptor are absent. No matching fixture process remained
at cleanup. The private fixture is preserved for diagnosis.
