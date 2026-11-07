# Reconciliation development v1

Status: prospective. Commit source, fixture and protocol before inference.

The isolated extraction v2 study classified a correction correctly but left it
unresolved despite a uniquely observed target. Two possible contributors are the
word-substituted relationship instructions and a false requirement-revision flag
on a correction candidate. Neither explanation is established by that failure.

Compare four arms: `legacy_full`, `legacy_relevant`, `direct_full` and
`direct_relevant`. The direct instructions explain matching the subject,
attribute and qualifiers without requiring the revised value to equal the old
value. They contain no fixture-specific example. The relevant flag variant omits
`revises_existing_requirement` only for non-requirement candidates. Requirement
flags, source text, targets, schemas, runtime guards and eligible operations stay
identical. The default remains legacy/full during this study.

Freeze `benchmarks/local_memory/fixtures/reconciliation-development-v1.json` at
SHA-256 `366ae35e01f517b88a0826dc2f109d51775065d552fcd0eb992033196776fef0`.
It contains 20 authored development cases in six clusters. Supplied, validated
classifications isolate reconciliation; this study does not measure extraction.
The first candidate and target reproduce the original correction input. Controls
cover absent, unrelated, duplicate and ambiguous targets; a matching target
among distractors; environment and attribute mismatches; numeric settings; new
facts; exact support; prospective requirements; and assistant history. The
assistant-disagreement case accepts either the observed target/version or null
for both, as permitted by the existing conflict contract. All other cases have
one prespecified operation/target/version alternative.

Run all four arms for three repetitions per case: 240 observations, with one
logical provider call per observation. Shuffle the full schedule with seed
2026110701 and preserve it. Pair results by original case and repetition; verify
the exact input hashes and schemas across arms. Repetitions are not independent
cases. Report per-case outcomes, exact operation/target/version agreement,
operational failures, unresolved controls, wrong support/correction targets,
latency and model token accounting. Reasons are retained for inspection, not
treated as proof that a target was ambiguous. No case or failure is excluded.

Use the existing local Qwen3:14b Q4_K_M model, digest
`bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`,
Ollama 0.40.9, temperature 0, thinking disabled, one parallel request, context
8192 and output limit 2048. Count each exact prompt plus decoder schema with the
verified model's native tokenizer before inference; reject above 5632 tokens.
Keep the 30-second provider and 60-second observation deadlines. Preserve raw
responses, retries, errors and all timings. Use a private loopback runtime and
the existing model files, then unload the model and stop only owned processes.

This is a development comparison using already studied families. No database,
retrieval, embedding, native host, live configuration or pilot is involved. A
variant is eligible for a separate full-pipeline regression only if it resolves
the known correction in all three repetitions, makes no wrong support or
correction target choices, has no operational failures, and loses no control
case that legacy/full passes in all repetitions. Rank eligible arms by cases
correct in all repetitions, then total correct observations. Prefer the arm
with fewer changes if tied; if both single-change arms tie, prefer direct/full
because it preserves the payload. Do not select an arm on latency alone.

Independently reconstruct every grade from preserved original inputs and raw
decisions. Verify source hashes, model/runtime identity, call accounting and
cleanup. Freeze the chosen arm before a separate rerun of the original full
extraction corpus. A small development win does not establish production
quality; fresh evaluation and the daily-use pilot remain required.
