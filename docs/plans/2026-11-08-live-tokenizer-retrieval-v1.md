# Live tokenizer callback during retrieval

The previous context-density measurement supplied precomputed native counts to
the packer. This increment measures a live local Qwen tokenizer callback inside
`MnemosEngine.retrieve` and verifies that `LocalMemoryService.retrieve` preserves
the resulting context and metadata. It changes no selection policy or model.

Use canonical product source at `c77933402a8bb4629211f24be22c3555fc872a76` and the
68 known questions and 32 accepted scenario snapshots from the closed
retrieval/support comparison. Preserve all original snapshots and scores.
Freeze source, model, tokenizer, embedding and dependency hashes before running.
Copy the complete scenario database separately for every observation; never
reuse an access-mutated copy across arms. Keep scope, baseline retrieval,
accepted facts, real BGE embeddings, candidate visibility and top-five ordering
identical. No extraction or answer-support inference runs.

Compare three arms at each numeric budget, 64 and 2048:

- An identified live native Qwen tokenizer callback, with literal special-token
  handling and no count cache.
- Ordinary UTF-8 byte fallback with no callback.
- A callback that fails on its first count, exercising complete byte repacking.

The same number denotes different units across native and byte arms. Do not
compare their raw budget numbers as equivalent token capacity. The schedule is
408 observations: 68 questions, two budgets, three arms, one pass. Rotate arm
order by question index and keep the two budgets separate in latency summaries.
This is a known-data integration experiment, not fresh quality acceptance.

The native runner is separately owned and loopback-only, using the previously
verified Ollama/Qwen artifacts. It may perform empty-prompt loading but no answer
generation. Do not stop existing applications or download a model. Bound each
callback request by a two-second timeout; invalid replies and network failures
must enter the existing byte fallback. Record every callback input, raw reply,
count and elapsed time. No credentials or live project data are sent.

Time complete local-service retrieval, including actual embedding, ranking,
packing and callback work. This excludes host launch and client transport and
must not be called complete host-command timing. Report callback calls and time
separately. Native tokenization of final contexts and exact detokenization are
audited outside the retrieval timer; do not use their counts to change selection.

Check that emitted records are whole, ranked, scoped and source-backed; the final
native count must match metadata and stay within its native budget. Byte arms
must report bytes, with null exact-token metadata. Injected-failure and ordinary
fallback must return identical context and selected IDs for every pair, including
the same evidence references and versions. The service's compact envelope must
not be described as covered by the context-only count.

Report supported facts per 1,000 actual Qwen context tokens alongside expected
fact retention, unsupported selections and per-budget local-service latency.
Keep empty-context density undefined. Do not turn operational failure into a
correct abstention or use repeated measurements as independent questions.
Expected labels remain unchanged; critical-fact retention remains unmeasured.

Commit the runner and its deterministic boundary checks before execution. If
integration or provenance checks fail, retain the failed attempt and repair the
specific defect before any permitted repeat. Quality or latency failure is not
a reason to tune budgets or rerun this protocol. Close and archive the result,
unload only the owned model, and verify owned-process exit and endpoint closure.

This can establish live Qwen callback integration. It cannot establish Codex or
Claude tokenizer compatibility, full host-envelope budgeting, 10,000-fact timing,
fresh quality, automatic enrichment benefit or pilot use. No activation or
publication is included.
