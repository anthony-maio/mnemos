# Target identity development v2

Status: prospective. Commit source, data binding and protocol before inference.

The [v1 comparison](../validation/2026-11-07/reconciliation-v1.md) found that
direct relationship instructions fixed matching corrections but also selected
wrong targets. Its 240 original observations and scores remain frozen. This
study separates identity comparison from the decision to correct a fact.

Compare three arms on the unchanged 20-case corpus, three repetitions each:
`legacy_full`, `direct_full` and `identity_v1`. Use
`benchmarks/local_memory/fixtures/reconciliation-development-v1.json`, SHA-256
`366ae35e01f517b88a0826dc2f109d51775065d552fcd0eb992033196776fef0`.
These are 180 observations over 20 authored development cases in six clusters,
not 180 independent cases or a fresh evaluation. Classifications remain supplied.
All arms include the separately tested duplicate-identity repair at `c30ed298`;
compare their current results and preserve v1's earlier source binding.

For user corrections and revision requirements, identity_v1 asks about the
subject, attribute and identifying qualifiers of every visible target. Each
dimension is same, different or unspecified, with literal quotes from both
sides where the wording exists. Changed values are separate from fact identity.
Missing qualifiers are unspecified; incompatible qualifiers are different.
Both sides lacking a qualifier can be same with null quotes. Quotes are checked
against the exact candidate and target, but literal occurrence does not prove
the model interpreted their meaning correctly or identified every qualifier.

Code permits correction only when exactly one target matches all dimensions
and every other target is ruled out by at least one different dimension. A
second possible target, an incomplete match or duplicate normalized identities
requires review. Complete prospective requirements with no possible old rule
can be added under the existing contract. Empty targets and exact duplicate
user candidates are handled mechanically. Other source kinds use legacy/full.
All target comparisons must name observed IDs and current versions exactly
once; malformed, missing, stale, invented-quote or mutated-input responses fail
closed. Batches of at most eight targets are combined before any decision.
Existing authority, provenance and scope validation still applies at the end.

The comparison changes the decomposition, prompts and decision rule together;
it does not isolate any one causal contributor. Inputs and visibility remain
paired by original case and repetition. Freeze the full randomized schedule
with seed 2026110701. Count every actual request, response, retry and failure;
identity's deterministic cases can require zero calls. There is no equal-call
assumption. Report exact operation/target agreement, all-repetition case counts,
unresolved controls, wrong-target corrections, quote failures, latency and
provider token accounting. Reasons and quotes are inspection evidence, not
semantic ground truth. Retain every observation regardless of outcome.

Use the existing Qwen3:14b Q4_K_M model, digest
`bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`,
Ollama 0.40.9, temperature zero, thinking disabled, context 8192 and output
limit 2048. Use the native tokenizer to count each prompt plus decoder schema
before inference and reject above 5632 tokens. Keep 30-second provider and
60-second observation deadlines, one parallel request and the private loopback
runtime. No downloads, database writes, retrieval, embeddings, live settings,
native host, installed pilot artifact or publication are part of this study.

Identity_v1 qualifies for a separately frozen full-pipeline regression only if
all 20 cases pass in all three repetitions, with zero wrong-target choices and
zero operational failures. This stricter small-development gate is feasible
before broader testing and does not replace the existing fresh-evaluation
thresholds. If it fails, preserve and diagnose the failure before deciding the
next action. Do not tune or remove cases during this run. No treatment becomes
a default based on this comparison.

Independently reconstruct each identity decision from the original comparison
fields, literal evidence, target versions and fixed aggregation rule. Verify
all input pairs, source hashes and per-call schemas, archive the source and
original outputs, and unload and stop only the owned runtime. A successful
development result would still leave full-pipeline behavior, persisted
extraction, fresh quality and the actual daily-use pilot open.
