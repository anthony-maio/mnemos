# Native Codex compact recall screen v1

Status: prospective. Commit the fixture, runner, grading and protocol before inference.

Compare full and compact recall in six fresh disposable Codex sessions, three per
format, in this order: full, compact, compact, full, full, compact. Use the pinned
Codex 0.164.0-alpha.11.1 executable and gpt-6-astra with high reasoning. No hooks,
shell, browsing, other memory tools or automatic capture are enabled. Preserve
normal credentials and host configuration. This screen does not activate the pilot.

The fixture and expected typed JSON fields are frozen in
`benchmarks/native_hosts/compact_recall.py`. Five accepted facts cover production
and staging log retention, archive encryption and retention, conflicting release
reviewer counts, and a staging queue with no stated production equivalent. Both
reviewer records carry conflict status. A separate project contains a forbidden
answer. Use deterministic identical synthetic vectors, baseline retrieval, all
experimental modules disabled, and the same SQLite fact and evidence snapshot.
Retrieval access metadata may advance; accepted facts, versions, evidence and
rendered context must remain identical. These vectors establish fixture coverage,
not semantic retrieval quality. No local cognition model runs.

Each prompt requests exactly one retrieval with the same query, five results and
project-only scope. The prompt template differs only in the response_format
selector. It asks for eight typed JSON fields and null for unsupported or
conflicted values. It does not disclose expected values. Grade every field
against the frozen fixture; wrong types, missing keys, duplicate keys, extra
keys or additional prose prevent an overall pass. Retain all failed attempts.

Do not force Code Mode to print the full MCP payload. Retain its actual code,
raw MCP reply, model-visible tool output and native inference requests. Correlate
the model call, code cell and runtime MCP call through native trace IDs. Confirm
exact tool arguments, fact identities, literal context and foreign-scope exclusion.
Record whether actual model input contains the complete context. If the host
selects only context or reformats the result, report that behavior explicitly.

Report actual terminal input, cached-input and output tokens, native duration,
eight-field answer agreement, raw reply bytes and model-visible output bytes.
Native usage includes host instructions, schemas, reasoning and repeated inputs;
it is not an arbitrary-text tokenizer, isolated context cost or billing estimate.
Small sample size, treatment order, fresh session IDs, cache behavior and model
variation limit causal and latency claims. No significance or general superiority
claim is supported by three repetitions of one synthetic scenario.

Allow at most 180 seconds per native process. Do not repeat a completed model
request to repair a harness or audit failure. Keep original responses and audit
them offline. On model/transport failure, retain the failed planned session and
continue other planned sessions only when cleanup and integrity checks permit.
Stop for source, scope, credential or fixture-integrity drift. Hash bound source
before and after the run, independently reconstruct reported metrics from original
evidence, remove temporary credential copies, verify owned processes and service
are stopped, and preserve the private archive. No release or live change occurs.
