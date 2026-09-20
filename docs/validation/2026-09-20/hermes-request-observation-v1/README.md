# Native Hermes request observation v1

The one scheduled development attempt captured two exact outgoing HTTP
request bodies and stopped at teaching. The model remembered the fact, then
reported its pending receipt without calling `mnemos_job_status`. The teach
gate failed. Its second response had finish reason `stop`, not `length`.
No recall, correction, forget, source replay, or final recall phase ran.

Both captured requests contain the complete `mnemos_forget_related` tool
schema, including `selected_source_event_ids`, its array type, string item
schema, maximum size, and empty default. The schema requires only `fact_id`,
`expected_version`, and `preview_digest`. Thus source selection is present
but optional in this model-facing schema. The service still requires the
appropriate source selections when additional facts are selected.

Neither request explicitly sets `tool_choice`. Both contain `think: false`,
`reasoning_effort: none`, and `options.num_ctx: 65536`. These are observed
request fields, not an inference from configuration. The observer preserves
the exact bytes, length, and SHA-256 before dispatch and records no headers.
It does not establish how Ollama internally processed those fields.

## Scope and evidence

- Runtime: verified `e419c2aa615e037e639a74b14615889d03f4e47f` wheel, unchanged from native v8.
- Committed diagnostic inputs: `5d130c36c5741ad1dce73694c7209dd323c1708f`.
- The [prospective protocol](../../../plans/2026-09-20-hermes-request-observation-protocol-v1.md) allowed one fresh attempt, with unchanged prompts, model profile, capture, deadlines, and lifecycle gates. It was not rerun after failing.
- Model readiness passed. The native teach phase exited 0 after 27.267 seconds, but its required job-status gate failed. The runner exited 1, closed the service, and removed its runtime descriptor; no cleanup error was recorded.
- The [preflight](../hermes-request-observation-preflight-v1/README.md) imported the observer without activation or inference. Separate offline reports pass 10 observer cases and 63 surrounding harness cases.
- `request-analysis.json` identifies the exact body files and relevant schema/settings. `evidence-index.json` binds 52 raw files; this README is outside the index.

This proves schema forwarding only for the two requests observed here. It
does not retrospectively establish the v8 wire bodies, and this attempt
never sent a forget preview. The separate [persistence diagnostic](../hermes-v8-preview-persistence-v1/README.md)
shows that v8 attempts 2 and 3 saved intact previews, which is a different
boundary. The [v8 cohort](../hermes-native-cohort-v8/README.md) remains zero
of three complete lifecycles.

The evidence narrows a proposed schema-loss explanation, but does not prove
one cause for all omissions or justify treating a schema edit as a model
reliability fix. Any proposed contract or model change needs a separate
development comparison before another acceptance cohort. Native reliability,
three-host continuity, complete capture, biomimetic quality, Desktop, and
the pilot remain open. Live configuration, source, and memory were unchanged.
